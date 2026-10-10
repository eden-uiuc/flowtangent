from __future__ import annotations

import abc
from typing import Any, List, Literal, Optional, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..utils import Module, static_field
from ..utils.data import DataLoader, slice_data
from ..utils.typing import LoaderType, _
from .kernels import Kernel
from .nn import TransformerBlock, fit_neural_model

# --- Module-Level Registries ---
_MANIFOLD_REGISTRY = {}


def register_manifold(name: str):
    def decorator(cls):
        _MANIFOLD_REGISTRY[name.upper()] = cls
        return cls

    return decorator


# --- DataLoader Compatibility Helpers ---
def _get_full_X(data: Union[Float[Array, "N D"], LoaderType]) -> Float[Array, "N D"]:
    """Drains a DataLoader into a single full-batch array for classical matrix inversion."""
    if hasattr(data, "dataset") and hasattr(data, "__iter__"):
        X_list = []
        for batch in data:
            # Depending on collate_fn, batch might be (X, Y) or just X
            X_list.append(batch[0] if isinstance(batch, (tuple, list)) else batch)
        return jnp.concatenate(X_list, axis=0)
    return data  # type: ignore


def _unpack_batch(batch: Union[Float[Array, "B D"], Tuple]) -> Float[Array, "B D"]:
    """Safely unpacks X_batch from a DataLoader (X, Y) tuple for unsupervised AE training."""
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


# --- Base Class ---
class Manifold(Module):
    input_dim: int = static_field(_)
    latent_dim: int = static_field(_)

    def __new__(cls, method: Optional[str] = None, *args, **kwargs):
        if cls is not Manifold:
            return super().__new__(cls)
        if method is None:
            raise ValueError("Must provide a method string when instantiating directly.")

        method = method.upper()
        if method not in _MANIFOLD_REGISTRY:
            raise ValueError(f"Unknown Manifold: {method}. Available: {list(_MANIFOLD_REGISTRY.keys())}")

        return _MANIFOLD_REGISTRY[method](*args, **kwargs)

    @abc.abstractmethod
    def encode(self, x: Float[Array, " D"]) -> Float[Array, " L"]:
        pass

    @abc.abstractmethod
    def decode(self, z: Float[Array, " L"]) -> Float[Array, " D"]:
        pass

    @abc.abstractmethod
    def fit(self, X: Union[Float[Array, "N D"], LoaderType], **kwargs) -> Manifold:
        pass

    def __add__(self, other: "Manifold") -> "Manifold":
        if not isinstance(other, Manifold):
            return NotImplemented

        # Flatten nested combinations cleanly
        left = self.manifolds if isinstance(self, CompositeManifold) else [self]
        right = other.manifolds if isinstance(other, CompositeManifold) else [other]

        return CompositeManifold(left + right)


@register_manifold("Composite")
class CompositeManifold(Manifold):
    """Adjoins multiple manifolds side-by-side."""

    manifolds: List[Manifold] = _
    _input_splits: Tuple[int, ...] = static_field(())
    _latent_splits: Tuple[int, ...] = static_field(())

    def __init__(self, manifolds: List[Manifold]):
        self.manifolds = manifolds
        self.input_dim = sum(m.input_dim for m in manifolds)
        self.latent_dim = sum(m.latent_dim for m in manifolds)

        # Pre-compute static split indices for JAX (e.g., sizes [2, 3, 4] -> indices [2, 5])
        in_sizes = [m.input_dim for m in manifolds[:-1]]
        self._input_splits = tuple(jnp.cumsum(jnp.array(in_sizes)).tolist())

        out_sizes = [m.latent_dim for m in manifolds[:-1]]
        self._latent_splits = tuple(jnp.cumsum(jnp.array(out_sizes)).tolist())

    def encode(self, x: Float[Array, " D"]) -> Float[Array, " L"]:
        x_parts = jnp.split(x, self._input_splits, axis=-1)
        z_parts = [m.encode(x_p) for m, x_p in zip(self.manifolds, x_parts)]
        return jnp.concatenate(z_parts, axis=-1)

    def decode(self, z: Float[Array, " L"]) -> Float[Array, " D"]:
        z_parts = jnp.split(z, self._latent_splits, axis=-1)
        x_parts = [m.decode(z_p) for m, z_p in zip(self.manifolds, z_parts)]
        return jnp.concatenate(x_parts, axis=-1)

    def fit(self, data: Union[Float[Array, "N D"], DataLoader], **kwargs) -> "CompositeManifold":
        new_manifolds = []
        for i, m in enumerate(self.manifolds):
            sliced_data = slice_data(data, self._input_splits, i)
            new_manifolds.append(m.fit(sliced_data, **kwargs))  # type: ignore

        return eqx.tree_at(lambda model: model.manifolds, self, new_manifolds)


@register_manifold("Passthrough")
class Passthrough(Manifold):
    """Passthrough manifold for direct methods"""

    def __init__(self, dim: int):
        self.input_dim = dim
        self.latent_dim = dim

    def encode(self, x: Float[Array, " D"]) -> Float[Array, " L"]:
        return x

    def decode(self, z: Float[Array, " L"]) -> Float[Array, " D"]:
        return z

    def fit(self, X: Union[Float[Array, "N D"], DataLoader], **kwargs) -> "Passthrough":
        return self


@register_manifold("SCALING")
class ScalingManifold(Manifold):
    """
    Unified manifold for transforming physical variables into an unbounded,
    normalized [-inf, inf] latent space suitable for gradient-based optimization.
    """

    method: Literal["linear", "log_bounded", "algebraic_bounded", "softplus", "logarithmic"] = static_field(
        "log_bounded"
    )

    scale_factors: jax.Array = _
    bounds_min: jax.Array = _
    bounds_max: jax.Array = _

    def __init__(
        self,
        dim: int,
        method: Literal["linear", "log_bounded", "algebraic_bounded", "softplus", "logarithmic"] = "log_bounded",
    ):
        self.input_dim = dim
        self.latent_dim = dim
        self.method = method
        self.scale_factors = jnp.ones(dim)
        self.bounds_min = jnp.zeros(dim)
        self.bounds_max = jnp.ones(dim)

    # ==========================================================================
    # 1. Routing
    # ==========================================================================

    def encode(self, x: Float[Array, " D"]) -> Float[Array, " L"]:
        func = getattr(self, f"_{self.method}_encode")
        return func(x)

    def decode(self, z: Float[Array, " L"]) -> Float[Array, " D"]:
        func = getattr(self, f"_{self.method}_decode")
        return func(z)

    # ==========================================================================
    # 2. Linear Scaling
    # ==========================================================================
    def _linear_encode(self, x):
        return x / self.scale_factors

    def _linear_decode(self, z):
        return z * self.scale_factors

    # ==========================================================================
    # 3. Log-Bounded (Logit)
    # ==========================================================================
    def _log_bounded_encode(self, x):
        norm = jnp.clip((x - self.bounds_min) / (self.bounds_max - self.bounds_min), 1e-6, 1.0 - 1e-6)
        return jnp.log(norm / (1.0 - norm))

    def _log_bounded_decode(self, z):
        return self.bounds_min + (self.bounds_max - self.bounds_min) * jax.nn.sigmoid(z)

    # ==========================================================================
    # 4. Algebraic-Bounded
    # ==========================================================================
    def _algebraic_bounded_encode(self, x):
        norm = jnp.clip((x - self.bounds_min) / (self.bounds_max - self.bounds_min), 1e-6, 1.0 - 1e-6)
        m = norm - 0.5
        return (2.0 * m) / jnp.sqrt(1.0 - 4.0 * (m**2))

    def _algebraic_bounded_decode(self, z):
        alg_sig = (z / jnp.sqrt(1.0 + z**2)) / 2.0
        return self.bounds_min + (self.bounds_max - self.bounds_min) * (alg_sig + 0.5)

    # ==========================================================================
    # 5. Softplus
    # ==========================================================================
    def _softplus_encode(self, x):
        return jnp.log(jnp.expm1(jnp.clip(x, 1e-6, None)))

    def _softplus_decode(self, z):
        return jax.nn.softplus(z)

    # ==========================================================================
    # 6. Logarithmic (Base 10)
    # ==========================================================================
    def _logarithmic_encode(self, x):
        return jnp.log10(jnp.clip(x, 1e-6, None))

    def _logarithmic_decode(self, z):
        return 10.0**z

    # ==========================================================================
    # 7. Data-Driven Fitting
    # ==========================================================================
    def fit(self, data: Union[Float[Array, "N D"], Any], **kwargs) -> "ScalingManifold":
        # Helper to extract full arrays if `data` is a DataLoader
        X_full = (
            data[0]
            if isinstance(data, (tuple, list))
            else (
                data
                if not hasattr(data, "__iter__")
                else jnp.concatenate(
                    [batch[0] if isinstance(batch, (tuple, list)) else batch for batch in data], axis=0
                )
            )
        )

        d_min = jnp.min(X_full, axis=0)
        d_max = jnp.max(X_full, axis=0)

        # Add 1% padding so the extrema don't map to absolute infinity
        padding = jnp.maximum((d_max - d_min) * 0.01, 1e-6)
        padded_min = d_min - padding
        padded_max = d_max + padding

        # For linear scaling, find the max absolute magnitude
        scale_factors = jnp.maximum(jnp.abs(padded_max), jnp.abs(padded_min))
        scale_factors = jnp.where(scale_factors == 0, 1.0, scale_factors)

        return eqx.tree_at(
            lambda m: (m.bounds_min, m.bounds_max, m.scale_factors), self, (padded_min, padded_max, scale_factors)
        )


# ==========================================
# 1. Statistical & Kernel Methods
# ==========================================


@register_manifold("PCA")
class LinearPCA(Manifold):
    """Classic Principal Component Analysis."""

    W_enc: jax.Array = _
    b_enc: jax.Array = _
    W_dec: jax.Array = _
    b_dec: jax.Array = _

    def __init__(self, physical_dim: int, latent_dim: int, key=None):
        self.latent_dim = latent_dim
        self.W_enc = jnp.zeros((latent_dim, physical_dim))
        self.b_enc = jnp.zeros((latent_dim,))
        self.W_dec = jnp.zeros((physical_dim, latent_dim))
        self.b_dec = jnp.zeros((physical_dim,))

    def encode(self, x: Float[Array, " D"]) -> Float[Array, " L"]:
        return self.W_enc @ x + self.b_enc

    def decode(self, z: Float[Array, " L"]) -> Float[Array, " D"]:
        return self.W_dec @ z + self.b_dec

    def fit(self, X: Union[Float[Array, "N D"], LoaderType], **kwargs) -> "LinearPCA":
        X_full = _get_full_X(X)

        # Exact PCA via Eigendecomposition
        mu = jnp.mean(X_full, axis=0)
        X_centered = X_full - mu
        cov = (X_centered.T @ X_centered) / (X_full.shape[0] - 1)

        eigenvalues, eigenvectors = jnp.linalg.eigh(cov)

        # Sort descending
        idx = jnp.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        W_enc = eigenvectors[:, : self.latent_dim].T
        b_enc = -W_enc @ mu
        W_dec = eigenvectors[:, : self.latent_dim]
        b_dec = mu

        return eqx.tree_at(lambda m: (m.W_enc, m.b_enc, m.W_dec, m.b_dec), self, (W_enc, b_enc, W_dec, b_dec))


@register_manifold("KPCA")
class KernelPCA(Manifold):
    """Non-Linear PCA utilizing the generalized Kernel class."""

    X_fit: jax.Array = _
    eigenvectors: jax.Array = _
    eigenvalues: jax.Array = _
    kernel: Kernel = _
    pre_image_decoder: eqx.nn.MLP = _

    def __init__(self, key, input_dim: int, latent_dim: int, kernel: Kernel):
        self.latent_dim = latent_dim
        self.kernel = kernel
        self.X_fit = jnp.zeros((1, input_dim))
        self.eigenvectors = jnp.zeros((1, latent_dim))
        self.eigenvalues = jnp.ones(latent_dim)
        self.pre_image_decoder = eqx.nn.MLP(latent_dim, input_dim, 64, 2, key=key)

    def encode(self, x: Float[Array, " D"]) -> Float[Array, " L"]:
        K_x = jax.vmap(lambda x_train: self.kernel(x, x_train))(self.X_fit)
        return (K_x @ self.eigenvectors) / jnp.sqrt(self.eigenvalues)

    def decode(self, z: Float[Array, " L"]) -> Float[Array, " D"]:
        return self.pre_image_decoder(z)

    def fit(
        self, X: Union[Float[Array, "N D"], LoaderType], learning_rate: float = 1e-3, epochs: int = 500, **kwargs
    ) -> "KernelPCA":
        X_full = _get_full_X(X)
        K = jax.vmap(lambda x1: jax.vmap(lambda x2: self.kernel(x1, x2))(X_full))(X_full)

        N = K.shape[0]
        one_n = jnp.ones((N, N)) / N
        K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n

        eigvals, eigvecs = jnp.linalg.eigh(K_centered)
        idx = jnp.argsort(eigvals)[::-1]
        top_eigvals = eigvals[idx][: self.latent_dim]
        top_eigvecs = eigvecs[:, idx][: self.latent_dim]

        model = eqx.tree_at(
            lambda m: (m.X_fit, m.eigenvalues, m.eigenvectors), self, (X_full, top_eigvals, top_eigvecs)
        )

        # Pre-image training uses iterative neural fit
        def decoder_loss(decoder, batch):
            Z_target, X_target = batch
            X_pred = jax.vmap(decoder)(Z_target)
            return jnp.mean((X_target - X_pred) ** 2)

        Z_full = jax.vmap(model.encode)(X_full)
        new_decoder = fit_neural_model(model.pre_image_decoder, (Z_full, X_full), decoder_loss, learning_rate, epochs)

        return eqx.tree_at(lambda m: m.pre_image_decoder, model, new_decoder)


# ==========================================
# 2. Deep Generative Models
# ==========================================


@register_manifold("VAE")
class VariationalAutoencoder(Manifold):
    """Standard VAE. Enforces a Gaussian prior on the latent space for MCMC."""

    encoder: eqx.nn.MLP = _
    decoder: eqx.nn.MLP = _

    def __init__(
        self, key, physical_dim: int, latent_dim: int, width: int = 128, depth: int = 3, activation=jax.nn.gelu
    ):
        k1, k2 = jax.random.split(key)
        self.latent_dim = latent_dim
        self.encoder = eqx.nn.MLP(physical_dim, latent_dim * 2, width, depth, activation, key=k1)
        self.decoder = eqx.nn.MLP(latent_dim, physical_dim, width, depth, activation, key=k2)

    def encode(self, x: Float[Array, " D"], key=None) -> Float[Array, " L"]:
        stats = self.encoder(x)
        mu, logvar = jnp.split(stats, 2, axis=-1)
        if key is None:
            return mu
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(key, mu.shape)
        return mu + eps * std

    def decode(self, z: Float[Array, " L"]) -> Float[Array, " D"]:
        return self.decoder(z)

    def fit(
        self, X: Union[Float[Array, "N D"], LoaderType], learning_rate: float = 1e-3, epochs: int = 500, **kwargs
    ) -> "VariationalAutoencoder":
        def vae_loss(model, batch):
            X_batch = _unpack_batch(batch)

            stats = jax.vmap(model.encoder)(X_batch)
            mu, logvar = jnp.split(stats, 2, axis=-1)

            kl_loss = -0.5 * jnp.mean(jnp.sum(1 + logvar - mu**2 - jnp.exp(logvar), axis=-1))

            std = jnp.exp(0.5 * logvar)
            # Pseudo-random key hashing ensures stochasticity during dataloader streaming
            batch_key = jax.random.fold_in(jax.random.PRNGKey(0), jnp.sum(X_batch).astype(jnp.int32))
            eps = jax.random.normal(batch_key, mu.shape)
            z = mu + eps * std

            X_recon = jax.vmap(model.decoder)(z)
            recon_loss = jnp.mean((X_batch - X_recon) ** 2)

            return recon_loss + kl_loss

        return fit_neural_model(self, X, vae_loss, learning_rate, epochs)


@register_manifold("T_AE")
class TransformerAutoencoder(Manifold):
    embedding: eqx.nn.Linear = _
    positional_encoding: jax.Array = _
    blocks: list = _
    latent_proj: eqx.nn.Linear = _
    decoder_proj: eqx.nn.Linear = _
    seq_len: int = _
    feature_dim: int = _

    def __init__(self, key, seq_len: int, feature_dim: int, latent_dim: int, num_heads: int = 4, hidden_size: int = 64):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.embedding = eqx.nn.Linear(feature_dim, hidden_size, key=k1)
        self.positional_encoding = jax.random.normal(k2, (seq_len, hidden_size)) * 0.02

        block_keys = jax.random.split(k3, 3)
        self.blocks = [TransformerBlock(hidden_size, num_heads, bk) for bk in block_keys]

        self.latent_proj = eqx.nn.Linear(seq_len * hidden_size, latent_dim, key=k4)
        self.decoder_proj = eqx.nn.Linear(latent_dim, seq_len * feature_dim, key=k1)

    def encode(self, x: Float[Array, " D"]) -> Float[Array, " L"]:
        # Reshape the flat 1D array back into sequence blocks
        x_seq = x.reshape((self.seq_len, self.feature_dim))
        embedded = jax.vmap(self.embedding)(x_seq) + self.positional_encoding
        for block in self.blocks:
            embedded = block(embedded)
        return self.latent_proj(embedded.flatten())

    def decode(self, z: Float[Array, " L"]) -> Float[Array, " D"]:
        # decoder_proj already flattens to match the 1D contract
        return self.decoder_proj(z)

    def fit(
        self, X: Union[Float[Array, "N D"], LoaderType], learning_rate: float = 1e-3, epochs: int = 500, **kwargs
    ) -> "TransformerAutoencoder":
        def ae_loss(model, batch):
            X_batch = _unpack_batch(batch)
            Z = jax.vmap(model.encode)(X_batch)
            X_recon = jax.vmap(model.decode)(Z)
            return jnp.mean((X_batch - X_recon) ** 2)

        return fit_neural_model(self, X, ae_loss, learning_rate, epochs)


@register_manifold("SSM_AE")
class StateSpaceAutoencoder(Manifold):
    """Continuous-time sequence modeling (Mamba/S4 style)."""

    A: jax.Array = _
    B: jax.Array = _
    C: jax.Array = _
    D: jax.Array = _
    decoder_proj: eqx.nn.Linear = _
    seq_len: int = _
    feature_dim: int = _

    def __init__(self, key, seq_len: int, feature_dim: int, latent_dim: int, state_dim: int = 64):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.A = jax.random.normal(k1, (state_dim, state_dim))
        self.B = jax.random.normal(k2, (state_dim, feature_dim))
        self.C = jax.random.normal(k3, (latent_dim, state_dim))
        self.D = jax.random.normal(k4, (latent_dim, feature_dim))
        self.decoder_proj = eqx.nn.Linear(latent_dim, seq_len * feature_dim, key=k5)

    def encode(self, x: Float[Array, " D"]) -> Float[Array, " L"]:
        x_seq = x.reshape((self.seq_len, self.feature_dim))

        def scan_fn(state, x_t):
            new_state = self.A @ state + self.B @ x_t
            return new_state, new_state

        initial_state = jnp.zeros(self.A.shape[0])
        _, hidden_states = jax.lax.scan(scan_fn, initial_state, x_seq)
        return self.C @ hidden_states[-1] + self.D @ x_seq[-1]

    def decode(self, z: Float[Array, " L"]) -> Float[Array, " D"]:
        return self.decoder_proj(z)

    def fit(
        self, X: Union[Float[Array, "N D"], LoaderType], learning_rate: float = 1e-3, epochs: int = 500, **kwargs
    ) -> "StateSpaceAutoencoder":
        def ae_loss(model, batch):
            X_batch = _unpack_batch(batch)
            Z = jax.vmap(model.encode)(X_batch)
            X_recon = jax.vmap(model.decode)(Z)
            return jnp.mean((X_batch - X_recon) ** 2)

        return fit_neural_model(self, X, ae_loss, learning_rate, epochs)


class AffineCoupling(Module):
    mask: jax.Array = _
    scale_translate_net: eqx.nn.MLP = _

    def __init__(self, key, dim: int, mask: jax.Array):
        self.mask = mask
        self.scale_translate_net = eqx.nn.MLP(dim, dim * 2, 64, 2, key=key)

    def forward(self, x):
        x_masked = x * self.mask
        st = self.scale_translate_net(x_masked)
        s, t = jnp.split(st, 2, axis=-1)
        s = jax.nn.tanh(s) * (1 - self.mask)
        t = t * (1 - self.mask)
        return x * jnp.exp(s) + t, s

    def inverse(self, y):
        y_masked = y * self.mask
        st = self.scale_translate_net(y_masked)
        s, t = jnp.split(st, 2, axis=-1)
        s = jax.nn.tanh(s) * (1 - self.mask)
        t = t * (1 - self.mask)
        return (y - t) * jnp.exp(-s)


@register_manifold("NormalizingFlow")
class NormalizingFlow(Manifold):
    """Perfectly invertible probability mapping using RealNVP."""

    layers: List[AffineCoupling] = _

    def __init__(self, key, dim: int, num_layers: int = 4):
        keys = jax.random.split(key, num_layers)
        self.latent_dim = dim
        self.layers = []
        for i, k in enumerate(keys):
            mask = jnp.arange(dim) < (dim // 2) if i % 2 == 0 else jnp.arange(dim) >= (dim // 2)
            self.layers.append(AffineCoupling(key=k, dim=dim, mask=jnp.array(mask, dtype=jnp.float32)))

    def encode(self, x: Float[Array, " D"]) -> Float[Array, " L"]:
        for layer in self.layers:
            x, _ = layer.forward(x)
        return x

    def decode(self, z: Float[Array, " L"]) -> Float[Array, " D"]:
        for layer in reversed(self.layers):
            z = layer.inverse(z)
        return z

    def fit(
        self, X: Union[Float[Array, "N D"], LoaderType], learning_rate: float = 1e-3, epochs: int = 500, **kwargs
    ) -> "NormalizingFlow":
        def ae_loss(model, batch):
            X_batch = _unpack_batch(batch)
            Z = jax.vmap(model.encode)(X_batch)
            X_recon = jax.vmap(model.decode)(Z)
            return jnp.mean((X_batch - X_recon) ** 2)

        return fit_neural_model(self, X, ae_loss, learning_rate, epochs)


@register_manifold("PointCloud")
class PointCloudManifold(Manifold):
    """Handles unordered 3D point clouds via permutation-invariant max pooling."""

    shared_mlp1: eqx.nn.MLP = _
    shared_mlp2: eqx.nn.MLP = _
    decoder: eqx.nn.MLP = _
    num_points: int = _

    def __init__(self, key, num_points: int, latent_dim: int):
        k1, k2, k3 = jax.random.split(key, 3)
        self.num_points = num_points
        self.latent_dim = latent_dim
        self.shared_mlp1 = eqx.nn.MLP(3, 64, 64, 2, key=k1)
        self.shared_mlp2 = eqx.nn.MLP(64, latent_dim, 128, 2, key=k2)
        self.decoder = eqx.nn.MLP(latent_dim, num_points * 3, 256, 3, key=k3)

    def encode(self, x: Float[Array, " D"]) -> Float[Array, " L"]:
        points = x.reshape((self.num_points, 3))
        features = jax.vmap(self.shared_mlp1)(points)
        features = jax.vmap(self.shared_mlp2)(features)
        global_feature = jnp.max(features, axis=0)
        return global_feature

    def decode(self, z: Float[Array, " L"]) -> Float[Array, " D"]:
        # decoder already projects to flat num_points * 3 array
        return self.decoder(z)

    def fit(
        self, X: Union[Float[Array, "N D"], LoaderType], learning_rate: float = 1e-3, epochs: int = 500, **kwargs
    ) -> "PointCloudManifold":
        def ae_loss(model, batch):
            X_batch = _unpack_batch(batch)
            Z = jax.vmap(model.encode)(X_batch)
            X_recon = jax.vmap(model.decode)(Z)
            return jnp.mean((X_batch - X_recon) ** 2)

        return fit_neural_model(self, X, ae_loss, learning_rate, epochs)
