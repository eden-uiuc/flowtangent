import abc
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import optimistix as optx
from typing import List, Optional, Any, TypeVar

from ..utils import Module
from ..utils.typing import _
from .kernels import Kernel
from .nn import TransformerBlock, fit_neural_model

M = TypeVar("M", bound=Module)

# --- Module-Level Registries ---
_MANIFOLD_REGISTRY = {}

def register_manifold(name: str):
    def decorator(cls):
        _MANIFOLD_REGISTRY[name.upper()] = cls
        return cls
    return decorator


# --- Base Class ---
class Manifold(Module):
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
    def encode(self, x: jax.Array) -> jax.Array: 
        pass

    @abc.abstractmethod
    def decode(self, z: jax.Array) -> jax.Array: 
        pass
    
    @abc.abstractmethod
    def fit(self, X: jax.Array, **kwargs) -> "Manifold": 
        pass


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
    latent_dim: int = _

    def __init__(self, physical_dim: int, latent_dim: int, key=None):
        self.latent_dim = latent_dim
        # Placeholders prior to fitting
        self.W_enc = jnp.zeros((latent_dim, physical_dim))
        self.b_enc = jnp.zeros((latent_dim,))
        self.W_dec = jnp.zeros((physical_dim, latent_dim))
        self.b_dec = jnp.zeros((physical_dim,))

    def encode(self, x): return self.W_enc @ x + self.b_enc
    def decode(self, z): return self.W_dec @ z + self.b_dec

    def fit(self, X: jax.Array, **kwargs) -> "LinearPCA":
        # Exact PCA via Eigendecomposition
        mu = jnp.mean(X, axis=0)
        X_centered = X - mu
        cov = (X_centered.T @ X_centered) / (X.shape[0] - 1)
        
        eigenvalues, eigenvectors = jnp.linalg.eigh(cov)
        
        # Sort descending
        idx = jnp.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        
        W_enc = eigenvectors[:, :self.latent_dim].T
        b_enc = -W_enc @ mu
        W_dec = eigenvectors[:, :self.latent_dim]
        b_dec = mu
        
        return eqx.tree_at(
            lambda m: (m.W_enc, m.b_enc, m.W_dec, m.b_dec), 
            self, 
            (W_enc, b_enc, W_dec, b_dec)
        )

@register_manifold("KPCA")
class KernelPCA(Manifold):
    """Non-Linear PCA utilizing the generalized Kernel class."""
    X_fit: jax.Array = _
    eigenvectors: jax.Array = _
    eigenvalues: jax.Array = _
    kernel: Kernel = _
    pre_image_decoder: eqx.nn.MLP = _
    latent_dim: int = _

    def __init__(self, key, input_dim: int, latent_dim: int, kernel: Kernel):
        self.latent_dim = latent_dim
        self.kernel = kernel
        self.X_fit = jnp.zeros((1, input_dim))
        self.eigenvectors = jnp.zeros((1, latent_dim))
        self.eigenvalues = jnp.ones(latent_dim)
        self.pre_image_decoder = eqx.nn.MLP(latent_dim, input_dim, 64, 2, key=key)

    def encode(self, x):
        K_x = jax.vmap(lambda x_train: self.kernel(x, x_train))(self.X_fit)
        return (K_x @ self.eigenvectors) / jnp.sqrt(self.eigenvalues)

    def decode(self, z):
        return self.pre_image_decoder(z)

    def fit(self, X: jax.Array, learning_rate: float = 1e-3, epochs: int = 500, **kwargs) -> "KernelPCA":
        K = jax.vmap(lambda x1: jax.vmap(lambda x2: self.kernel(x1, x2))(X))(X)
        
        N = K.shape[0]
        one_n = jnp.ones((N, N)) / N
        K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n
        
        eigvals, eigvecs = jnp.linalg.eigh(K_centered)
        idx = jnp.argsort(eigvals)[::-1]
        top_eigvals = eigvals[idx][:self.latent_dim]
        top_eigvecs = eigvecs[:, idx][:self.latent_dim]
        
        # 1. Update statistical KPCA parameters
        model = eqx.tree_at(
            lambda m: (m.X_fit, m.eigenvalues, m.eigenvectors), 
            self, 
            (X, top_eigvals, top_eigvecs)
        )
        
        # 2. Train the pre-image decoder using Optimistix
        def decoder_loss(decoder, args):
            Z_target, X_target = args
            X_pred = jax.vmap(decoder)(Z_target)
            return jnp.mean((X_target - X_pred)**2)

        Z = jax.vmap(model.encode)(X)
        new_decoder = fit_neural_model(model.pre_image_decoder, (Z, X), decoder_loss, learning_rate, epochs)
        
        return eqx.tree_at(lambda m: m.pre_image_decoder, model, new_decoder)


# ==========================================
# 2. Deep Generative Models
# ==========================================

@register_manifold("VAE")
class VariationalAutoencoder(Manifold):
    """Standard VAE. Enforces a Gaussian prior on the latent space for MCMC."""
    encoder: eqx.nn.MLP = _
    decoder: eqx.nn.MLP = _
    latent_dim: int = _

    def __init__(self, key, physical_dim: int, latent_dim: int, width: int = 128, depth: int = 3, activation=jax.nn.gelu):
        k1, k2 = jax.random.split(key)
        self.latent_dim = latent_dim
        self.encoder = eqx.nn.MLP(physical_dim, latent_dim * 2, width, depth, activation, key=k1)
        self.decoder = eqx.nn.MLP(latent_dim, physical_dim, width, depth, activation, key=k2)

    def encode(self, x, key=None):
        stats = self.encoder(x)
        mu, logvar = jnp.split(stats, 2, axis=-1)
        if key is None:
            return mu 
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(key, mu.shape)
        return mu + eps * std

    def decode(self, z): 
        return self.decoder(z)

    def fit(self, X: jax.Array, learning_rate: float = 1e-3, epochs: int = 500, **kwargs) -> "VariationalAutoencoder":
        def vae_loss(model, args):
            X_batch, key = args
            stats = jax.vmap(model.encoder)(X_batch)
            mu, logvar = jnp.split(stats, 2, axis=-1)
            
            kl_loss = -0.5 * jnp.mean(jnp.sum(1 + logvar - mu**2 - jnp.exp(logvar), axis=-1))
            
            std = jnp.exp(0.5 * logvar)
            eps = jax.random.normal(key, mu.shape)
            z = mu + eps * std
            
            X_recon = jax.vmap(model.decoder)(z)
            recon_loss = jnp.mean((X_batch - X_recon)**2)
            
            return recon_loss + kl_loss

        key = jax.random.PRNGKey(0) # Static key for optimistix loop
        return fit_neural_model(self, (X, key), vae_loss, learning_rate, epochs)


@register_manifold("TRANSFORMER_AE")
class TransformerAutoencoder(Manifold):
    embedding: eqx.nn.Linear = _
    positional_encoding: jax.Array = _
    blocks: list = _  # Replace transformer with a list of blocks
    latent_proj: eqx.nn.Linear = _
    decoder_proj: eqx.nn.Linear = _
    seq_len: int = _

    def __init__(self, key, seq_len: int, feature_dim: int, latent_dim: int, num_heads: int = 4, hidden_size: int = 64):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.seq_len = seq_len
        self.embedding = eqx.nn.Linear(feature_dim, hidden_size, key=k1)
        self.positional_encoding = jax.random.normal(k2, (seq_len, hidden_size)) * 0.02
        
        # Instantiate 3 custom transformer blocks
        block_keys = jax.random.split(k3, 3)
        self.blocks = [TransformerBlock(hidden_size, num_heads, bk) for bk in block_keys]
        
        self.latent_proj = eqx.nn.Linear(seq_len * hidden_size, latent_dim, key=k4)
        self.decoder_proj = eqx.nn.Linear(latent_dim, seq_len * feature_dim, key=k1)

    def encode(self, x):
        x = jax.vmap(self.embedding)(x) + self.positional_encoding
        for block in self.blocks:
            x = block(x)
        return self.latent_proj(x.flatten())


@register_manifold("SSM_AE")
class StateSpaceAutoencoder(Manifold):
    """Continuous-time sequence modeling (Mamba/S4 style)."""
    A: jax.Array = _
    B: jax.Array = _
    C: jax.Array = _
    D: jax.Array = _
    decoder_proj: eqx.nn.Linear = _
    seq_len: int = _
    
    def __init__(self, key, seq_len: int, physical_dim: int, latent_dim: int, state_dim: int = 64):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.seq_len = seq_len
        self.A = jax.random.normal(k1, (state_dim, state_dim))
        self.B = jax.random.normal(k2, (state_dim, physical_dim))
        self.C = jax.random.normal(k3, (latent_dim, state_dim))
        self.D = jax.random.normal(k4, (latent_dim, physical_dim))
        self.decoder_proj = eqx.nn.Linear(latent_dim, seq_len * physical_dim, key=k5)

    def encode(self, x_seq):
        def scan_fn(state, x_t):
            new_state = self.A @ state + self.B @ x_t
            return new_state, new_state
            
        initial_state = jnp.zeros(self.A.shape[0])
        _, hidden_states = jax.lax.scan(scan_fn, initial_state, x_seq)
        return self.C @ hidden_states[-1] + self.D @ x_seq[-1]

    def decode(self, z):
        return self.decoder_proj(z).reshape(self.seq_len, -1)

    def fit(self, X: jax.Array, learning_rate: float = 1e-3, epochs: int = 500, **kwargs) -> "StateSpaceAutoencoder":
        def ae_loss(model, X_batch):
            Z = jax.vmap(model.encode)(X_batch)
            X_recon = jax.vmap(model.decode)(Z)
            return jnp.mean((X_batch - X_recon)**2)

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


@register_manifold("NORMALIZING_FLOW")
class NormalizingFlow(Manifold):
    """Perfectly invertible probability mapping using RealNVP."""
    layers: List[AffineCoupling] = _

    def __init__(self, key, dim: int, num_layers: int = 4):
        keys = jax.random.split(key, num_layers)
        self.layers = []
        for i, k in enumerate(keys):
            mask = jnp.arange(dim) < (dim // 2) if i % 2 == 0 else jnp.arange(dim) >= (dim // 2)
            self.layers.append(AffineCoupling(key=k, dim=dim, mask=jnp.array(mask, dtype=jnp.float32)))

    def encode(self, x):
        for layer in self.layers:
            x, _ = layer.forward(x)
        return x

    def decode(self, z):
        for layer in reversed(self.layers):
            z = layer.inverse(z)
        return z

    def fit(self, X: jax.Array, learning_rate: float = 1e-3, epochs: int = 500, **kwargs) -> "NormalizingFlow":
        def ae_loss(model, X_batch):
            # Optimizing flows often involves likelihood, but we use MSE reconstruction 
            # as a general proxy to fit the Manifold standard AE interface.
            Z = jax.vmap(model.encode)(X_batch)
            X_recon = jax.vmap(model.decode)(Z)
            return jnp.mean((X_batch - X_recon)**2)

        return fit_neural_model(self, X, ae_loss, learning_rate, epochs)


@register_manifold("POINTNET")
class PointCloudManifold(Manifold):
    """Handles unordered 3D point clouds via permutation-invariant max pooling."""
    shared_mlp1: eqx.nn.MLP = _
    shared_mlp2: eqx.nn.MLP = _
    decoder: eqx.nn.MLP = _
    num_points: int = _

    def __init__(self, key, num_points: int, latent_dim: int):
        k1, k2, k3 = jax.random.split(key, 3)
        self.num_points = num_points
        self.shared_mlp1 = eqx.nn.MLP(3, 64, 64, 2, key=k1) 
        self.shared_mlp2 = eqx.nn.MLP(64, latent_dim, 128, 2, key=k2)
        self.decoder = eqx.nn.MLP(latent_dim, num_points * 3, 256, 3, key=k3)

    def encode(self, points):
        features = jax.vmap(self.shared_mlp1)(points)
        features = jax.vmap(self.shared_mlp2)(features)
        global_feature = jnp.max(features, axis=0) 
        return global_feature

    def decode(self, z):
        flat_points = self.decoder(z)
        return flat_points.reshape(self.num_points, 3)

    def fit(self, X: jax.Array, learning_rate: float = 1e-3, epochs: int = 500, **kwargs) -> "PointCloudManifold":
        def ae_loss(model, X_batch):
            Z = jax.vmap(model.encode)(X_batch)
            X_recon = jax.vmap(model.decode)(Z)
            return jnp.mean((X_batch - X_recon)**2)

        return fit_neural_model(self, X, ae_loss, learning_rate, epochs)