import abc
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, TypeAlias, List, Union, Tuple
from jaxtyping import Float, Array

from ..utils import Module
from ..utils.typing import _
from ..utils.data import DataLoader, LatentDataLoader
from .kernels import Kernel
from .nn import TransformerBlock, fit_neural_model

_loader: TypeAlias =  DataLoader | LatentDataLoader

# --- Module-Level Registries ---
_SURROGATE_REGISTRY = {}

def register_surrogate(name: str):
    def decorator(cls):
        _SURROGATE_REGISTRY[name.upper()] = cls
        return cls
    return decorator

# --- DataLoader Compatibility Helper ---
def _get_full_XY(data: Union[Tuple[jax.Array, jax.Array], _loader]) -> Tuple[jax.Array, jax.Array]:
    """Drains a DataLoader into full-batch arrays for classical matrix inversion."""
    if hasattr(data, "dataset") and hasattr(data, "__iter__"):
        X_list, Y_list = [], []
        for batch in data:
            X_list.append(batch[0])
            Y_list.append(batch[1])
        return jnp.concatenate(X_list, axis=0), jnp.concatenate(Y_list, axis=0)
    return data[0], data[1]


# --- Base State and Classes ---
class SurrogateEvaluation(Module):
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
    def predict(self, x: jax.Array, compute_derivatives: bool = False) -> SurrogateEvaluation: 
        pass

    @abc.abstractmethod
    def fit(self, data: Union[Tuple[jax.Array, jax.Array], _loader], **kwargs) -> "Surrogate":
        pass


# ==========================================
# 1. Classical & Statistical Surrogates
# ==========================================

@register_surrogate("PCE")
class PolynomialChaosSurrogate(Surrogate):
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

    def predict(self, x, compute_derivatives=False):
        means = self._internal_predict(x)
        preds = SurrogateEvaluation(means=means)
        if compute_derivatives:
            preds = eqx.tree_at(lambda p: p.derivatives, preds, jax.jacrev(self._internal_predict)(x))
        return preds

    def fit(self, data: Union[Tuple[jax.Array, jax.Array], _loader], learning_rate: float = 1e-2, epochs: int = 500, **kwargs) -> "PolynomialChaosSurrogate":
        def loss_fn(model, batch):
            x_batch, y_batch = batch
            
            preds = jax.vmap(model._internal_predict)(x_batch)
            return jnp.mean((y_batch - preds)**2)

        return fit_neural_model(self, data, loss_fn, learning_rate, epochs)


@register_surrogate("IDW")
class InverseDistanceWeighting(Surrogate):
    X_fit: jax.Array = _
    Y_fit: jax.Array = _
    power: float = _

    def __init__(self, input_dim: int, output_dim: int, power: float = 2.0):
        self.X_fit = jnp.zeros((1, input_dim))
        self.Y_fit = jnp.zeros((1, output_dim))
        self.power = power

    def _internal_predict(self, x):
        distances = jnp.linalg.norm(self.X_fit - x, axis=-1)
        weights = 1.0 / (distances ** self.power + 1e-8)
        weights /= jnp.sum(weights)
        return jnp.sum(weights[:, None] * self.Y_fit, axis=0)

    def predict(self, x, compute_derivatives=False):
        means = self._internal_predict(x)
        preds = SurrogateEvaluation(means=means)
        if compute_derivatives:
            preds = eqx.tree_at(lambda p: p.derivatives, preds, jax.jacrev(self._internal_predict)(x))
        return preds

    def fit(self, data: Union[Tuple[jax.Array, jax.Array], _loader], **kwargs) -> "InverseDistanceWeighting":
        X_full, Y_full = _get_full_XY(data)
            
        return eqx.tree_at(lambda m: (m.X_fit, m.Y_fit), self, (X_full, Y_full))


@register_surrogate("RBF")
class RadialBasisFunction(Surrogate):
    centers: jax.Array = _
    weights: jax.Array = _
    lengthscale: float = _

    def __init__(self, input_dim: int, output_dim: int, lengthscale: float = 1.0):
        self.centers = jnp.zeros((1, input_dim))
        self.weights = jnp.zeros((1, output_dim))
        self.lengthscale = lengthscale

    def _internal_predict(self, x):
        r = jnp.linalg.norm(self.centers - x, axis=-1)
        phi = jnp.exp(-0.5 * (r / self.lengthscale)**2)
        return self.weights.T @ phi

    def predict(self, x, compute_derivatives=False):
        means = self._internal_predict(x)
        preds = SurrogateEvaluation(means=means)
        if compute_derivatives:
            preds = eqx.tree_at(lambda p: p.derivatives, preds, jax.jacrev(self._internal_predict)(x))
        return preds

    def fit(self, data: Union[Tuple[jax.Array, jax.Array], _loader], **kwargs) -> "RadialBasisFunction":
        X_full, Y_full = _get_full_XY(data)
            
        def phi_fn(x1, x2):
            return jnp.exp(-0.5 * jnp.sum((x1 - x2)**2) / self.lengthscale**2)
            
        Phi = jax.vmap(lambda x1: jax.vmap(lambda x2: phi_fn(x1, x2))(X_full))(X_full)
        Phi_ridge = Phi + 1e-6 * jnp.eye(X_full.shape[0])
        W = jnp.linalg.solve(Phi_ridge, Y_full)
        
        return eqx.tree_at(lambda m: (m.centers, m.weights), self, (X_full, W))


@register_surrogate("SVGP")
class GaussianProcessSurrogate(Surrogate):
    inducing_points: jax.Array = _
    kernel: Kernel = _
    variational_mean: jax.Array = _
    noise_variance: jax.Array = _

    def __init__(self, key, input_dim: int, output_dim: int, kernel: Kernel, num_inducing: int = 128):
        k1, k2 = jax.random.split(key)
        self.kernel = kernel
        self.inducing_points = jax.random.uniform(k1, (num_inducing, input_dim))
        self.variational_mean = jax.random.normal(k2, (num_inducing, output_dim)) * 1e-2
        self.noise_variance = jnp.zeros(output_dim)

    def predict(self, x, compute_derivatives=False):
        K_xz = jax.vmap(lambda z: self.kernel(x, z))(self.inducing_points)
        means = K_xz @ self.variational_mean
        var = self.kernel(x, x) - jnp.sum(K_xz**2, axis=-1, keepdims=True) + jnp.exp(self.noise_variance)
        
        preds = SurrogateEvaluation(means=means, variances=var)
        if compute_derivatives:
            mean_fn = lambda x_in: (jax.vmap(lambda z: self.kernel(x_in, z))(self.inducing_points)) @ self.variational_mean
            preds = eqx.tree_at(lambda p: p.derivatives, preds, jax.jacrev(mean_fn)(x))
        return preds

    def fit(self, data: Union[Tuple[jax.Array, jax.Array], _loader], learning_rate: float = 1e-2, epochs: int = 500, **kwargs) -> "GaussianProcessSurrogate":
        def loss_fn(model, batch):
            x_batch, y_batch = batch
            
            preds = jax.vmap(model.predict)(x_batch).means
            return jnp.mean((y_batch - preds)**2)
            
        return fit_neural_model(self, data, loss_fn, learning_rate, epochs)


# ==========================================
# 2. Deep Learning Surrogates
# ==========================================

@register_surrogate("MLP")
class MLPSurrogate(Surrogate):
    network: eqx.nn.MLP = _
    
    def __init__(self, key, input_dim: int, output_dim: int, width: int = 256, depth: int = 4):
        self.network = eqx.nn.MLP(input_dim, output_dim, width, depth, activation=jax.nn.silu, key=key)

    def _internal_predict(self, x):
        return self.network(x)

    def predict(self, x, compute_derivatives=False):
        means = self._internal_predict(x)
        preds = SurrogateEvaluation(means=means)
        if compute_derivatives:
            preds = eqx.tree_at(lambda p: p.derivatives, preds, jax.jacrev(self._internal_predict)(x))
        return preds

    def fit(self, data: Union[Tuple[jax.Array, jax.Array], _loader], learning_rate: float = 1e-3, epochs: int = 1000, **kwargs) -> "MLPSurrogate":
        def loss_fn(model, batch):
            x_batch, y_batch = batch
            preds = jax.vmap(model._internal_predict)(x_batch)
            return jnp.mean((y_batch - preds)**2)
            
        return fit_neural_model(self, data, loss_fn, learning_rate, epochs)


@register_surrogate("TRANSFORMER")
class TransformerSurrogate(Surrogate):
    latent_dim: int = _
    latent_embed: eqx.nn.Linear = _
    context_embed: eqx.nn.Linear = _
    blocks: List[TransformerBlock] = _
    head: eqx.nn.Linear = _

    def __init__(self, key, latent_dim: int, context_dim: int, output_dim: int, hidden_size: int = 128):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.latent_dim = latent_dim
        self.latent_embed = eqx.nn.Linear(latent_dim, hidden_size, key=k1)
        self.context_embed = eqx.nn.Linear(context_dim, hidden_size, key=k2)
        
        block_keys = jax.random.split(k3, 4)
        self.blocks = [TransformerBlock(hidden_size, 4, bk) for bk in block_keys]
        self.head = eqx.nn.Linear(hidden_size, output_dim, key=k4)

    def _internal_predict(self, x):
        z, c = jnp.split(x, [self.latent_dim], axis=-1)
        z_tok = self.latent_embed(z)
        c_tok = self.context_embed(c)
        seq = jnp.stack([z_tok, c_tok], axis=0)
        
        for block in self.blocks:
            seq = block(seq)
            
        pooled = jnp.mean(seq, axis=0)
        return self.head(pooled)

    def predict(self, x, compute_derivatives=False):
        means = self._internal_predict(x)
        preds = SurrogateEvaluation(means=means)
        if compute_derivatives:
            preds = eqx.tree_at(lambda p: p.derivatives, preds, jax.jacrev(self._internal_predict)(x))
        return preds

    def fit(self, data: Union[Tuple[jax.Array, jax.Array], _loader], learning_rate: float = 1e-3, epochs: int = 500, **kwargs) -> "TransformerSurrogate":
        def loss_fn(model, batch):
            x_batch, y_batch = batch
            
            preds = jax.vmap(model._internal_predict)(x_batch)
            return jnp.mean((y_batch - preds)**2)
            
        return fit_neural_model(self, data, loss_fn, learning_rate, epochs)


@register_surrogate("DEEP_ENSEMBLE")
class DeepEnsembleSurrogate(Surrogate):
    models: eqx.nn.MLP = _  

    def __init__(self, key, num_models: int, input_dim: int, output_dim: int):
        keys = jax.random.split(key, num_models)
        self.models = eqx.filter_vmap(lambda k: eqx.nn.MLP(input_dim, output_dim, 128, 3, key=k))(keys)

    def _internal_predict(self, x):
        return eqx.filter_vmap(lambda m: m(x))(self.models)

    def predict(self, x, compute_derivatives=False):
        samples = self._internal_predict(x)
        means = jnp.mean(samples, axis=0)
        variances = jnp.var(samples, axis=0)
        
        preds = SurrogateEvaluation(means=means, variances=variances, samples=samples)
        if compute_derivatives:
            mean_fn = lambda x_in: jnp.mean(self._internal_predict(x_in), axis=0)
            preds = eqx.tree_at(lambda p: p.derivatives, preds, jax.jacrev(mean_fn)(x))
        return preds

    def fit(self, data: Union[Tuple[jax.Array, jax.Array], _loader], learning_rate: float = 1e-3, epochs: int = 500, **kwargs) -> "DeepEnsembleSurrogate":
        def loss_fn(model, batch):
            x_batch, y_batch = batch
            
            all_preds = jax.vmap(model._internal_predict)(x_batch)
            return jnp.mean((jnp.expand_dims(y_batch, 1) - all_preds)**2)
            
        return fit_neural_model(self, data, loss_fn, learning_rate, epochs)


@register_surrogate("MOE")
class MixtureOfExperts(Surrogate):
    gating_network: eqx.nn.Linear = _
    experts: eqx.nn.MLP = _ 

    def __init__(self, key, num_experts: int, input_dim: int, output_dim: int):
        k1, k2 = jax.random.split(key)
        self.gating_network = eqx.nn.Linear(input_dim, num_experts, key=k1)
        keys = jax.random.split(k2, num_experts)
        self.experts = eqx.filter_vmap(lambda k: eqx.nn.MLP(input_dim, output_dim, 64, 2, key=k))(keys)

    def _internal_predict(self, x):
        gate_logits = self.gating_network(x)
        routing_weights = jax.nn.softmax(gate_logits)
        expert_outputs = eqx.filter_vmap(lambda m: m(x))(self.experts)
        return jnp.sum(routing_weights[:, None] * expert_outputs, axis=0)

    def predict(self, x, compute_derivatives=False):
        means = self._internal_predict(x)
        preds = SurrogateEvaluation(means=means)
        if compute_derivatives:
            preds = eqx.tree_at(lambda p: p.derivatives, preds, jax.jacrev(self._internal_predict)(x))
        return preds

    def fit(self, data: Union[Tuple[jax.Array, jax.Array], _loader], learning_rate: float = 1e-3, epochs: int = 500, **kwargs) -> "MixtureOfExperts":
        def loss_fn(model, batch):
            x_batch, y_batch = batch
            
            preds = jax.vmap(model._internal_predict)(x_batch)
            return jnp.mean((y_batch - preds)**2)
            
        return fit_neural_model(self, data, loss_fn, learning_rate, epochs)


class SpectralConv1d(Module):
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
    lifting: eqx.nn.Linear = _
    spectral_conv: SpectralConv1d = _
    projection: eqx.nn.Linear = _

    def __init__(self, key, input_dim: int, output_dim: int, modes: int = 16, width: int = 64):
        k1, k2, k3 = jax.random.split(key, 3)
        self.lifting = eqx.nn.Linear(input_dim, width, key=k1)
        self.spectral_conv = SpectralConv1d(k2, width, width, modes)
        self.projection = eqx.nn.Linear(width, output_dim, key=k3)

    def _internal_predict(self, x):
        lifted = self.lifting(x).reshape(-1, 1) 
        conv_out = jax.nn.gelu(self.spectral_conv(lifted))
        return self.projection(conv_out.flatten())

    def predict(self, x, compute_derivatives=False):
        means = self._internal_predict(x)
        preds = SurrogateEvaluation(means=means)
        if compute_derivatives:
            preds = eqx.tree_at(lambda p: p.derivatives, preds, jax.jacrev(self._internal_predict)(x))
        return preds

    def fit(self, data: Union[Tuple[jax.Array, jax.Array], _loader], learning_rate: float = 1e-3, epochs: int = 500, **kwargs) -> "FourierNeuralOperator":
        def loss_fn(model, batch):
            x_batch, y_batch = batch
            
            preds = jax.vmap(model._internal_predict)(x_batch)
            return jnp.mean((y_batch - preds)**2)
            
        return fit_neural_model(self, data, loss_fn, learning_rate, epochs)