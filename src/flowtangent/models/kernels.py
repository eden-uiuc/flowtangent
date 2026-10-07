import abc
import jax
import jax.numpy as jnp
from typing import Dict, Any, Union
from ..utils import Module
from ..utils.typing import _

class Kernel(Module):
    """Base class for all covariance kernels."""
    
    @abc.abstractmethod
    def __call__(self, x1: jax.Array, x2: jax.Array) -> jax.Array:
        pass

    @property
    @abc.abstractmethod
    def theta(self) -> Dict[str, jax.Array]:
        """Returns the learnable hyperparameters of the kernel."""
        pass

    # --- Operator Overloading for Composition ---
    def __add__(self, other: "Kernel") -> "Kernel":
        return SumKernel(k1=self, k2=other)

    def __mul__(self, other: "Kernel") -> "Kernel":
        return ProductKernel(k1=self, k2=other)
        
    def __pow__(self, exponent: float) -> "Kernel":
        return ExponentiationKernel(kernel=self, exponent=exponent)

# --- 1. Composition Kernels ---

class SumKernel(Kernel):
    k1: Kernel = _
    k2: Kernel = _
    
    def __init__(self, k1: Kernel, k2: Kernel):
        self.k1 = k1
        self.k2 = k2
        
    def __call__(self, x1, x2): 
        return self.k1(x1, x2) + self.k2(x1, x2)
        
    @property
    def theta(self): 
        return {"k1": self.k1.theta, "k2": self.k2.theta}

class ProductKernel(Kernel):
    k1: Kernel = _
    k2: Kernel = _
    
    def __init__(self, k1: Kernel, k2: Kernel):
        self.k1 = k1
        self.k2 = k2
        
    def __call__(self, x1, x2): 
        return self.k1(x1, x2) * self.k2(x1, x2)
        
    @property
    def theta(self): 
        return {"k1": self.k1.theta, "k2": self.k2.theta}

class ExponentiationKernel(Kernel):
    kernel: Kernel = _
    exponent: float = _
    
    def __init__(self, kernel: Kernel, exponent: float):
        self.kernel = kernel
        self.exponent = exponent
        
    def __call__(self, x1, x2):
        return self.kernel(x1, x2) ** self.exponent
        
    @property
    def theta(self):
        return self.kernel.theta

# --- 2. Base Scikit-Learn Kernels ---

class ConstantKernel(Kernel):
    """Returns a constant value. Often used with ProductKernel to scale another kernel."""
    constant_value: jax.Array = _

    def __init__(self, constant_value: float = 1.0):
        self.constant_value = jnp.array([jnp.log(constant_value)])

    def __call__(self, x1, x2):
        return jnp.exp(self.constant_value)[0]

    @property
    def theta(self):
        return {"constant_value": jnp.exp(self.constant_value)}


class WhiteKernel(Kernel):
    """
    White noise kernel. Returns noise_level if x1 == x2, else 0.
    In practice, usually added directly to the diagonal of the covariance matrix.
    """
    noise_level: jax.Array = _

    def __init__(self, noise_level: float = 1e-5):
        self.noise_level = jnp.array([jnp.log(noise_level)])

    def __call__(self, x1, x2):
        # Uses a tight tolerance for float equality in JAX
        is_equal = jnp.allclose(x1, x2, atol=1e-6)
        return jnp.where(is_equal, jnp.exp(self.noise_level)[0], 0.0)

    @property
    def theta(self):
        return {"noise_level": jnp.exp(self.noise_level)}


class DotProductKernel(Kernel):
    """Dot-product (linear) kernel: k(x, y) = sigma_0^2 + x * y"""
    sigma_0: jax.Array = _

    def __init__(self, sigma_0: float = 1.0):
        self.sigma_0 = jnp.array([jnp.log(sigma_0)])

    def __call__(self, x1, x2):
        return jnp.exp(self.sigma_0)**2 + jnp.sum(x1 * x2, axis=-1)

    @property
    def theta(self):
        return {"sigma_0": jnp.exp(self.sigma_0)}


class PairwiseKernel(Kernel):
    """
    Wrapper for non-stationary pairwise metrics (linear, polynomial, cosine).
    Note: Some metrics may not yield positive-semidefinite matrices.
    """
    metric: str = _
    gamma: jax.Array = _

    def __init__(self, metric: str = "linear", gamma: float = 1.0):
        self.metric = metric.lower()
        self.gamma = jnp.array([jnp.log(gamma)])

    def __call__(self, x1, x2):
        g = jnp.exp(self.gamma)
        if self.metric == "linear":
            return jnp.sum(x1 * x2, axis=-1)
        elif self.metric == "polynomial":
            return (g * jnp.sum(x1 * x2, axis=-1) + 1.0) ** 3.0 # Hardcoded degree 3 for proxy
        elif self.metric == "cosine":
            n1 = jnp.linalg.norm(x1, axis=-1)
            n2 = jnp.linalg.norm(x2, axis=-1)
            return jnp.sum(x1 * x2, axis=-1) / (n1 * n2 + 1e-8)
        else:
            raise ValueError(f"Unsupported Pairwise metric: {self.metric}")

    @property
    def theta(self):
        return {"gamma": jnp.exp(self.gamma)}

# --- 3. Stationary / Distance-based Kernels ---

class RBFKernel(Kernel):
    lengthscales: jax.Array = _
    variance: jax.Array = _

    def __init__(self, input_dim: int):
        self.lengthscales = jnp.zeros(input_dim) 
        self.variance = jnp.zeros(1)

    def __call__(self, x1, x2):
        ls = jnp.exp(self.lengthscales)
        sq_dist = jnp.sum(((x1 / ls) - (x2 / ls))**2, axis=-1)
        return jnp.exp(self.variance) * jnp.exp(-0.5 * sq_dist)

    @property
    def theta(self):
        return {"lengthscales": jnp.exp(self.lengthscales), "variance": jnp.exp(self.variance)}


class Matern52Kernel(Kernel):
    lengthscales: jax.Array = _
    variance: jax.Array = _

    def __init__(self, input_dim: int):
        self.lengthscales = jnp.zeros(input_dim) 
        self.variance = jnp.zeros(1)

    def __call__(self, x1, x2):
        r = jnp.sqrt(jnp.sum(((x1 - x2) / jnp.exp(self.lengthscales))**2, axis=-1) + 1e-8)
        sqrt5_r = jnp.sqrt(5.0) * r
        return jnp.exp(self.variance) * (1.0 + sqrt5_r + (5.0/3.0)*r**2) * jnp.exp(-sqrt5_r)

    @property
    def theta(self):
        return {"lengthscales": jnp.exp(self.lengthscales), "variance": jnp.exp(self.variance)}


class RationalQuadraticKernel(Kernel):
    """
    Equivalent to adding together many RBF kernels with different lengthscales.
    Excellent for data featuring variations across multiple scales.
    """
    lengthscale: jax.Array = _
    alpha: jax.Array = _

    def __init__(self, lengthscale: float = 1.0, alpha: float = 1.0):
        self.lengthscale = jnp.array([jnp.log(lengthscale)])
        self.alpha = jnp.array([jnp.log(alpha)])

    def __call__(self, x1, x2):
        ls = jnp.exp(self.lengthscale)
        alpha = jnp.exp(self.alpha)
        sq_dist = jnp.sum((x1 - x2)**2, axis=-1)
        return (1.0 + sq_dist / (2.0 * alpha * ls**2)) ** (-alpha)

    @property
    def theta(self):
        return {"lengthscale": jnp.exp(self.lengthscale), "alpha": jnp.exp(self.alpha)}


class ExpSineSquaredKernel(Kernel):
    """
    Also known as the Periodic Kernel. 
    Models functions that repeat themselves exactly.
    """
    lengthscale: jax.Array = _
    periodicity: jax.Array = _

    def __init__(self, lengthscale: float = 1.0, periodicity: float = 1.0):
        self.lengthscale = jnp.array([jnp.log(lengthscale)])
        self.periodicity = jnp.array([jnp.log(periodicity)])

    def __call__(self, x1, x2):
        ls = jnp.exp(self.lengthscale)
        p = jnp.exp(self.periodicity)
        # Uses Euclidean distance for the sine argument
        dist = jnp.linalg.norm(x1 - x2, axis=-1)
        return jnp.exp(-2.0 * (jnp.sin(jnp.pi * dist / p) / ls)**2)

    @property
    def theta(self):
        return {"lengthscale": jnp.exp(self.lengthscale), "periodicity": jnp.exp(self.periodicity)}