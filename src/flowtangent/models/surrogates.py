import abc
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, TypeVar, List

from ..utils import Module
from ..utils.typing import _
from .kernels import Kernel
from .nn import TransformerBlock, fit_neural_model

# --- Module-Level Registries ---
_SURROGATE_REGISTRY = {}

def register_surrogate(name: str):
    def decorator(cls):
        _SURROGATE_REGISTRY[name.upper()] = cls
        return cls
    return decorator

# --- Shared Optimistix Training Helper ---
M = TypeVar("M", bound=Module)


# --- Base State and Classes ---
class PredictionState(Module):
    """The universal, strongly-typed return payload for all surrogates."""
    means: jax.Array = _
    variances: Optional[jax.Array] = None
    covariance: Optional[jax.Array] = None
    samples: Optional[jax.Array] = None       
    derivatives: Optional[jax.Array] = None   

    @property
    def is_probabilistic(self) -> bool:
        return self.variances is not None or self.samples is not None


class Surrogate(Module):
    def __new__(cls, method: str = "", *args, **kwargs):
        if cls is not Surrogate:
            return super().__new__(cls)
            
        method = method.upper()
        if method not in _SURROGATE_REGISTRY:
            raise ValueError(f"Unknown Surrogate: {method}. Available: {list(_SURROGATE_REGISTRY.keys())}")
            
        return _SURROGATE_REGISTRY[method](*args, **kwargs)

    @abc.abstractmethod
    def predict(self, latent_z: jax.Array, context: jax.Array, compute_derivatives: bool = False) -> PredictionState: 
        pass

    @abc.abstractmethod
    def fit(self, latent_z: jax.Array, context: jax.Array, targets: jax.Array, **kwargs) -> "Surrogate":
        pass


# ==========================================
# 1. Classical & Statistical Surrogates
# ==========================================

@register_surrogate("PCE")
class PolynomialChaosSurrogate(Surrogate):
    """Projects inputs into orthogonal polynomial bases."""
    weights: eqx.nn.Linear = _
    degree: int = _

    def __init__(self, key, input_dim: int, output_dim: int, degree: int = 3):
        self.degree = degree
        poly_features = input_dim * degree 
        self.weights = eqx.nn.Linear(poly_features, output_dim, use_bias=True, key=key)

    def _polynomial_basis(self, x):
        bases = [x ** d for d in range(1, self.degree + 1)]
        return jnp.concatenate(bases, axis=-1)
        
    def _internal_predict(self, x):
        poly_x = self._polynomial_basis(x)
        return self.weights(poly_x)

    def predict(self, latent_z, context, compute_derivatives=False):
        x = jnp.concatenate([latent_z, context], axis=-1)
        means = self._internal_predict(x)
        
        preds = PredictionState(means=means)
        if compute_derivatives:
            preds = eqx.tree_at(lambda p: p.derivatives, preds, jax.jacrev(self._internal_predict)(x))
        return preds

    def fit(self, latent_z: jax.Array, context: jax.Array, targets: jax.Array, learning_rate: float = 1e-2, epochs: int = 500, **kwargs) -> "PolynomialChaosSurrogate":
        def loss_fn(model, args):
            z, c, y = args
            x = jnp.concatenate([z, c], axis=-1)
            preds = jax.vmap(model._internal_predict)(x)
            return jnp.mean((y - preds)**2)

        return fit_neural_model(self, (latent_z, context, targets), loss_fn, learning_rate, epochs)


@register_surrogate("IDW")
class InverseDistanceWeighting(Surrogate):
    """Fast, deterministic interpolation without training."""
    X_fit: jax.Array = _
    Y_fit: jax.Array = _
    power: float = _

    def __init__(self, input_dim: int, output_dim: int, power: float = 2.0):
        self.X_fit = jnp.zeros((1, input_dim))
        self.Y_fit = jnp.zeros((1, output_dim))
        self.power = power

    def _internal_predict(self, x_in):
        distances = jnp.linalg.norm(self.X_fit - x_in, axis=-1)
        weights = 1.0 / (distances ** self.power + 1e-8)
        weights /= jnp.sum(weights)
        return jnp.sum(weights[:, None] * self.Y_fit, axis=0)

    def predict(self, latent_z, context, compute_derivatives=False):
        x = jnp.concatenate([latent_z, context], axis=-1)
        means = self._internal_predict(x)
        
        preds = PredictionState(means=means)
        if compute_derivatives:
            preds = eqx.tree_at(lambda p: p.derivatives, preds, jax.jacrev(self._internal_predict)(x))
        return preds

    def fit(self, latent_z: jax.Array, context: jax.Array, targets: jax.Array, **kwargs) -> "InverseDistanceWeighting":
        # IDW doesn't "train", it just memorizes the dataset.
        X = jnp.concatenate([latent_z, context], axis=-1)
        return eqx.tree_at(lambda m: (m.X_fit, m.Y_fit), self, (X, targets))


@register_surrogate("RBF")
class RadialBasisFunction(Surrogate):
    """Interpolates using radial basis functions. Requires exact linear solve."""
    centers: jax.Array = _
    weights: jax.Array = _
    lengthscale: float = _

    def __init__(self, input_dim: int, output_dim: int, lengthscale: float = 1.0):
        self.centers = jnp.zeros((1, input_dim))
        self.weights = jnp.zeros((1, output_dim))
        self.lengthscale = lengthscale

    def _internal_predict(self, x_in):
        r = jnp.linalg.norm(self.centers - x_in, axis=-1)
        phi = jnp.exp(-0.5 * (r / self.lengthscale)**2)
        return self.weights.T @ phi

    def predict(self, latent_z, context, compute_derivatives=False):
        x = jnp.concatenate([latent_z, context], axis=-1)
        means = self._internal_predict(x)
        
        preds = PredictionState(means=means)
        if compute_derivatives:
            preds = eqx.tree_at(lambda p: p.derivatives, preds, jax.jacrev(self._internal_predict)(x))
        return preds

    def fit(self, latent_z: jax.Array, context: jax.Array, targets: jax.Array, **kwargs) -> "RadialBasisFunction":
        X = jnp.concatenate([latent_z, context], axis=-1)
        
        # Calculate Phi matrix: phi_ij = exp(-0.5 * ||X_i - X_j||^2 / lengthscale^2)
        def phi_fn(x1, x2):
            return jnp.exp(-0.5 * jnp.sum((x1 - x2)**2) / self.lengthscale**2)
            
        Phi = jax.vmap(lambda x1: jax.vmap(lambda x2: phi_fn(x1, x2))(X))(X)
        
        # Add ridge term for numerical stability
        Phi_ridge = Phi + 1e-6 * jnp.eye(X.shape[0])
        
        # Solve exactly for weights
        W = jnp.linalg.solve(Phi_ridge, targets)
        
        return eqx.tree_at(lambda m: (m.centers, m.weights), self, (X, W))


@register_surrogate("SVGP")
class GaussianProcessSurrogate(Surrogate):
    """Sparse Variational Gaussian Process using the FlowTangent Kernel class."""
    inducing_points: jax.Array = _
    kernel: Kernel = _
    variational_mean: jax.Array = _
    noise_variance: jax.Array = _

    def __init__(self, key, input_dim: int, output_dim: int, kernel: Kernel, num_inducing: int = 128):
        k1, k2 = jax.random.split(key)
        self.kernel = kernel
        self.inducing_points = jax.random.uniform(k1, (num_inducing, input_dim))
        self.variational_mean = jax.random.normal(k2, (num_inducing, output_dim)) * 1e-2
        self.noise_variance = jnp.zeros(output_dim) # log-space

    def predict(self, latent_z, context, compute_derivatives=False):
        x = jnp.concatenate([latent_z, context], axis=-1)
        
        # Simplified prediction path (In a full implementation, you also need K_zz inverse)
        K_xz = jax.vmap(lambda z: self.kernel(x, z))(self.inducing_points)
        means = K_xz @ self.variational_mean
        
        # Rough diagonal variance estimation
        var = self.kernel(x, x) - jnp.sum(K_xz**2, axis=-1, keepdims=True) + jnp.exp(self.noise_variance)
        
        preds = PredictionState(means=means, variances=var)
        if compute_derivatives:
            mean_fn = lambda x_in: (jax.vmap(lambda z: self.kernel(x_in, z))(self.inducing_points)) @ self.variational_mean
            preds = eqx.tree_at(lambda p: p.derivatives, preds, jax.jacrev(mean_fn)(x))
        return preds

    def fit(self, latent_z: jax.Array, context: jax.Array, targets: jax.Array, learning_rate: float = 1e-2, epochs: int = 500, **kwargs) -> "GaussianProcessSurrogate":
        # Note: A true GP uses ELBO. This is a proxy MSE optimization for demonstration.
        def loss_fn(model, args):
            z, c, y = args
            preds = jax.vmap(model.predict)(z, c).means
            return jnp.mean((y - preds)**2)
            
        return fit_neural_model(self, (latent_z, context, targets), loss_fn, learning_rate, epochs)


# ==========================================
# 2. Deep Learning Surrogates
# ==========================================

@register_surrogate("MLP")
class MLPSurrogate(Surrogate):
    """The deep learning workhorse. Fast and highly scalable."""
    network: eqx.nn.MLP = _
    
    def __init__(self, key, input_dim: int, output_dim: int, width: int = 256, depth: int = 4):
        self.network = eqx.nn.MLP(input_dim, output_dim, width, depth, activation=jax.nn.silu, key=key)

    def _internal_predict(self, x):
        return self.network(x)

    def predict(self, latent_z, context, compute_derivatives=False):
        x = jnp.concatenate([latent_z, context], axis=-1)
        means = self._internal_predict(x)
        
        preds = PredictionState(means=means)
        if compute_derivatives:
            preds = eqx.tree_at(lambda p: p.derivatives, preds, jax.jacrev(self._internal_predict)(x))
        return preds

    def fit(self, latent_z: jax.Array, context: jax.Array, targets: jax.Array, learning_rate: float = 1e-3, epochs: int = 1000, **kwargs) -> "MLPSurrogate":
        def loss_fn(model, args):
            z, c, y = args
            preds = jax.vmap(model.predict)(z, c).means
            return jnp.mean((y - preds)**2)
            
        return fit_neural_model(self, (latent_z, context, targets), loss_fn, learning_rate, epochs)


@register_surrogate("TRANSFORMER")
class TransformerSurrogate(Surrogate):
    """Treats latent and context as separate tokens for cross-attention."""
    latent_embed: eqx.nn.Linear = _
    context_embed: eqx.nn.Linear = _
    blocks: List[TransformerBlock] = _
    head: eqx.nn.Linear = _

    def __init__(self, key, latent_dim: int, context_dim: int, output_dim: int, hidden_size: int = 128):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.latent_embed = eqx.nn.Linear(latent_dim, hidden_size, key=k1)
        self.context_embed = eqx.nn.Linear(context_dim, hidden_size, key=k2)
        
        block_keys = jax.random.split(k3, 4)
        self.blocks = [TransformerBlock(hidden_size, 4, bk) for bk in block_keys]
        self.head = eqx.nn.Linear(hidden_size, output_dim, key=k4)

    def _internal_predict(self, latent_z, context):
        z_tok = self.latent_embed(latent_z)
        c_tok = self.context_embed(context)
        seq = jnp.stack([z_tok, c_tok], axis=0)
        
        for block in self.blocks:
            seq = block(seq)
            
        pooled = jnp.mean(seq, axis=0)
        return self.head(pooled)

    def predict(self, latent_z, context, compute_derivatives=False):
        means = self._internal_predict(latent_z, context)
        preds = PredictionState(means=means)
        
        if compute_derivatives:
            jac_z = jax.jacrev(lambda z: self._internal_predict(z, context))(latent_z)
            jac_c = jax.jacrev(lambda c: self._internal_predict(latent_z, c))(context)
            preds = eqx.tree_at(lambda p: p.derivatives, preds, jnp.concatenate([jac_z, jac_c], axis=-1))
            
        return preds

    def fit(self, latent_z: jax.Array, context: jax.Array, targets: jax.Array, learning_rate: float = 1e-3, epochs: int = 500, **kwargs) -> "TransformerSurrogate":
        def loss_fn(model, args):
            z, c, y = args
            preds = jax.vmap(model.predict)(z, c).means
            return jnp.mean((y - preds)**2)
            
        return fit_neural_model(self, (latent_z, context, targets), loss_fn, learning_rate, epochs)


@register_surrogate("DEEP_ENSEMBLE")
class DeepEnsembleSurrogate(Surrogate):
    """Uses multiple MLPs initialized with different seeds to provide epistemic uncertainty."""
    models: eqx.nn.MLP = _  

    def __init__(self, key, num_models: int, input_dim: int, output_dim: int):
        keys = jax.random.split(key, num_models)
        self.models = eqx.filter_vmap(lambda k: eqx.nn.MLP(input_dim, output_dim, 128, 3, key=k))(keys)

    def _internal_predict(self, x):
        return eqx.filter_vmap(lambda m: m(x))(self.models)

    def predict(self, latent_z, context, compute_derivatives=False):
        x = jnp.concatenate([latent_z, context], axis=-1)
        samples = self._internal_predict(x)
        
        means = jnp.mean(samples, axis=0)
        variances = jnp.var(samples, axis=0)
        preds = PredictionState(means=means, variances=variances, samples=samples)
        
        if compute_derivatives:
            mean_fn = lambda x_in: jnp.mean(self._internal_predict(x_in), axis=0)
            preds = eqx.tree_at(lambda p: p.derivatives, preds, jax.jacrev(mean_fn)(x))
            
        return preds

    def fit(self, latent_z: jax.Array, context: jax.Array, targets: jax.Array, learning_rate: float = 1e-3, epochs: int = 500, **kwargs) -> "DeepEnsembleSurrogate":
        def loss_fn(model, args):
            z, c, y = args
            x = jnp.concatenate([z, c], axis=-1)
            # Evaluate all models across the whole batch
            all_preds = jax.vmap(model._internal_predict)(x) # shape: [batch, num_models, outputs]
            return jnp.mean((jnp.expand_dims(y, 1) - all_preds)**2)
            
        return fit_neural_model(self, (latent_z, context, targets), loss_fn, learning_rate, epochs)


@register_surrogate("MOE")
class MixtureOfExperts(Surrogate):
    """Learns to route inputs to specialized local MLPs."""
    gating_network: eqx.nn.Linear = _
    experts: eqx.nn.MLP = _ 

    def __init__(self, key, num_experts: int, input_dim: int, output_dim: int):
        k1, k2 = jax.random.split(key)
        self.gating_network = eqx.nn.Linear(input_dim, num_experts, key=k1)
        keys = jax.random.split(k2, num_experts)
        self.experts = eqx.filter_vmap(lambda k: eqx.nn.MLP(input_dim, output_dim, 64, 2, key=k))(keys)

    def _internal_predict(self, x_in):
        gate_logits = self.gating_network(x_in)
        routing_weights = jax.nn.softmax(gate_logits)
        
        expert_outputs = eqx.filter_vmap(lambda m: m(x_in))(self.experts)
        return jnp.sum(routing_weights[:, None] * expert_outputs, axis=0)

    def predict(self, latent_z, context, compute_derivatives=False):
        x = jnp.concatenate([latent_z, context], axis=-1)
        means = self._internal_predict(x)
        
        preds = PredictionState(means=means)
        if compute_derivatives:
            preds = eqx.tree_at(lambda p: p.derivatives, preds, jax.jacrev(self._internal_predict)(x))
        return preds

    def fit(self, latent_z: jax.Array, context: jax.Array, targets: jax.Array, learning_rate: float = 1e-3, epochs: int = 500, **kwargs) -> "MixtureOfExperts":
        def loss_fn(model, args):
            z, c, y = args
            preds = jax.vmap(model.predict)(z, c).means
            return jnp.mean((y - preds)**2)
            
        return fit_neural_model(self, (latent_z, context, targets), loss_fn, learning_rate, epochs)


class SpectralConv1d(Module):
    """Core operator layer: Convolution in the frequency domain."""
    weights_real: jax.Array = _
    weights_imag: jax.Array = _
    modes: int = _

    def __init__(self, key, in_channels, out_channels, modes):
        k1, k2 = jax.random.split(key)
        self.modes = modes
        scale = (1.0 / (in_channels * out_channels))
        self.weights_real = jax.random.normal(k1, (in_channels, out_channels, modes)) * scale
        self.weights_imag = jax.random.normal(k2, (in_channels, out_channels, modes)) * scale

    def __call__(self, x):
        x_ft = jnp.fft.rfft(x)
        complex_weights = self.weights_real + 1j * self.weights_imag
        out_ft = jnp.zeros((complex_weights.shape[1], x_ft.shape[-1]), dtype=jnp.complex64)
        out_ft_modes = jnp.einsum("im,iom->om", x_ft[:, :self.modes], complex_weights)
        out_ft = out_ft.at[:, :self.modes].set(out_ft_modes)
        return jnp.fft.irfft(out_ft, n=x.shape[-1])


@register_surrogate("FNO")
class FourierNeuralOperator(Surrogate):
    """Maps continuous functions to continuous functions."""
    lifting: eqx.nn.Linear = _
    spectral_conv: SpectralConv1d = _
    projection: eqx.nn.Linear = _

    def __init__(self, key, input_dim: int, output_dim: int, modes: int = 16, width: int = 64):
        k1, k2, k3 = jax.random.split(key, 3)
        self.lifting = eqx.nn.Linear(input_dim, width, key=k1)
        self.spectral_conv = SpectralConv1d(k2, width, width, modes)
        self.projection = eqx.nn.Linear(width, output_dim, key=k3)

    def _internal_predict(self, x_in):
        lifted = self.lifting(x_in).reshape(-1, 1) 
        conv_out = jax.nn.gelu(self.spectral_conv(lifted))
        return self.projection(conv_out.flatten())

    def predict(self, latent_z, context, compute_derivatives=False):
        x = jnp.concatenate([latent_z, context], axis=-1)
        means = self._internal_predict(x)
        
        preds = PredictionState(means=means)
        if compute_derivatives:
            preds = eqx.tree_at(lambda p: p.derivatives, preds, jax.jacrev(self._internal_predict)(x))
        return preds

    def fit(self, latent_z: jax.Array, context: jax.Array, targets: jax.Array, learning_rate: float = 1e-3, epochs: int = 500, **kwargs) -> "FourierNeuralOperator":
        def loss_fn(model, args):
            z, c, y = args
            preds = jax.vmap(model.predict)(z, c).means
            return jnp.mean((y - preds)**2)
            
        return fit_neural_model(self, (latent_z, context, targets), loss_fn, learning_rate, epochs)