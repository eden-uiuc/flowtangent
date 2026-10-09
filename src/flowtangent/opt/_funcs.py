import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
from typing import Callable, Tuple

# --- Registries ---
_PENALTY_REGISTRY = {}
_ACQUISITION_REGISTRY = {}
_PROBABILITY_ACQ_REGISTRY = {}
_TRUST_REGISTRY = {}

def register_penalty(name: str):
    def decorator(fn):
        _PENALTY_REGISTRY[name.lower()] = fn
        return fn
    return decorator

def register_acquisition(name: str):
    def decorator(fn):
        _ACQUISITION_REGISTRY[name.lower()] = fn
        return fn
    return decorator

def register_trust(name: str):
    def decorator(fn):
        _TRUST_REGISTRY[name.lower()] = fn
        return fn
    return decorator

def register_probability_acq(name: str):
    def decorator(fn):
        _PROBABILITY_ACQ_REGISTRY[name.lower()] = fn
        return fn
    return decorator

# ==========================================
# Module-Level Import Routing (PEP 562)
# ==========================================
def __getattr__(name: str):
    name_lower = name.lower()
    if name_lower in _PENALTY_REGISTRY: return _PENALTY_REGISTRY[name_lower]
    if name_lower in _ACQUISITION_REGISTRY: return _ACQUISITION_REGISTRY[name_lower]
    if name_lower in _TRUST_REGISTRY: return _TRUST_REGISTRY[name_lower]
    if name_lower in _PROBABILITY_ACQ_REGISTRY: return _PROBABILITY_ACQ_REGISTRY[name_lower]
    raise AttributeError(f"module '{__name__}' has no registered function '{name}'")

def __dir__():
    return (
        list(_PENALTY_REGISTRY.keys()) + 
        list(_ACQUISITION_REGISTRY.keys()) + 
        list(_TRUST_REGISTRY.keys()) +
        list(_PROBABILITY_ACQ_REGISTRY.keys())
    )


# ==========================================
# 1. Penalty Functions 
# ==========================================

@register_penalty("quadratic_penalty")
def quadratic_penalty(x: jax.Array, bounds: Tuple[jax.Array, jax.Array], **kwargs) -> jax.Array:
    viol_lower = jnp.maximum(0.0, bounds[0] - x)
    viol_upper = jnp.maximum(0.0, x - bounds[1])
    return jnp.sum(viol_lower**2 + viol_upper**2)

@register_penalty("linear_penalty")
def linear_penalty(x: jax.Array, bounds: Tuple[jax.Array, jax.Array], **kwargs) -> jax.Array:
    viol_lower = jnp.maximum(0.0, bounds[0] - x)
    viol_upper = jnp.maximum(0.0, x - bounds[1])
    return jnp.sum(viol_lower + viol_upper)

@register_penalty("log_barrier")
def log_barrier(x: jax.Array, bounds: Tuple[jax.Array, jax.Array], **kwargs) -> jax.Array:
    d_lower = x - bounds[0]
    d_upper = bounds[1] - x
    return -jnp.sum(jnp.log(jnp.maximum(d_lower, 1e-8))) - jnp.sum(jnp.log(jnp.maximum(d_upper, 1e-8)))

@register_penalty("inverse_barrier")
def inverse_barrier(x: jax.Array, bounds: Tuple[jax.Array, jax.Array], **kwargs) -> jax.Array:
    d_lower = jnp.maximum(x - bounds[0], 1e-8)
    d_upper = jnp.maximum(bounds[1] - x, 1e-8)
    return jnp.sum(1.0 / d_lower + 1.0 / d_upper)

@register_penalty("exponential_penalty")
def exponential_penalty(x: jax.Array, bounds: Tuple[jax.Array, jax.Array], k: float = 10.0, **kwargs) -> jax.Array:
    viol_lower = jnp.maximum(0.0, bounds[0] - x)
    viol_upper = jnp.maximum(0.0, x - bounds[1])
    return jnp.sum((jnp.exp(k * viol_lower) - 1.0) + (jnp.exp(k * viol_upper) - 1.0))

@register_penalty("infinite_penalty")
def infinite_penalty(x: jax.Array, bounds: Tuple[jax.Array, jax.Array], **kwargs) -> jax.Array:
    in_bounds = jnp.all((x >= bounds[0]) & (x <= bounds[1]))
    return jnp.where(in_bounds, 0.0, jnp.inf)


# ==========================================
# 2. Acquisition Functions 
# ==========================================

@register_acquisition("ucb")
def upper_confidence_bound(mu: jax.Array, sigma: jax.Array, best_val: float, maximize: bool = True, kappa: float = 2.0, **kwargs) -> jax.Array:
    sign = 1.0 if maximize else -1.0
    return mu * sign + kappa * sigma

@register_acquisition("ei")
def expected_improvement(mu: jax.Array, sigma: jax.Array, best_val: float, maximize: bool = True, xi: float = 0.01, **kwargs) -> jax.Array:
    sign = 1.0 if maximize else -1.0
    improvement = (mu - best_val) * sign - xi
    sigma_safe = jnp.maximum(sigma, 1e-9)
    Z = improvement / sigma_safe
    ei = improvement * norm.cdf(Z) + sigma_safe * norm.pdf(Z)
    return jnp.where(sigma > 1e-9, ei, 0.0)

@register_acquisition("pi")
def probability_of_improvement(mu: jax.Array, sigma: jax.Array, best_val: float, maximize: bool = True, xi: float = 0.01, **kwargs) -> jax.Array:
    sign = 1.0 if maximize else -1.0
    improvement = (mu - best_val) * sign - xi
    sigma_safe = jnp.maximum(sigma, 1e-9)
    Z = improvement / sigma_safe
    return jnp.where(sigma > 1e-9, norm.cdf(Z), 0.0)

@register_acquisition("greedy")
def greedy_exploitation(mu: jax.Array, sigma: jax.Array, best_val: float, maximize: bool = True, **kwargs) -> jax.Array:
    sign = 1.0 if maximize else -1.0
    return mu * sign


# ==========================================
# 3. Trust Functions 
# ==========================================

@register_trust("exponential_trust")
def exponential_trust(variances: jax.Array, k: float = 1.0, **kwargs) -> jax.Array:
    return jnp.exp(-k * jnp.mean(variances))

@register_trust("inverse_trust")
def inverse_trust(variances: jax.Array, k: float = 1.0, **kwargs) -> jax.Array:
    return 1.0 / (1.0 + k * jnp.mean(variances))

@register_trust("threshold_trust")
def threshold_trust(variances: jax.Array, threshold: float = 0.1, **kwargs) -> jax.Array:
    return jnp.where(jnp.mean(variances) < threshold, 1.0, 0.0)

@register_trust("entropy_trust")
def entropy_trust(variances: jax.Array, max_variance_expected: float = 1.0, **kwargs) -> jax.Array:
    mean_var = jnp.maximum(jnp.mean(variances), 1e-12)
    entropy = 0.5 * jnp.log(2 * jnp.pi * jnp.e * mean_var)
    max_entropy = 0.5 * jnp.log(2 * jnp.pi * jnp.e * max_variance_expected)
    return jnp.clip(1.0 - (entropy / max_entropy), 0.0, 1.0)

# ==========================================
# 4. Probability Acquisition Functions (for Boundaries)
# ==========================================

@register_probability_acq("binary_entropy")
def binary_entropy(p: jax.Array, **kwargs) -> jax.Array:
    """
    Seeks maximum confusion. Peaks at p = 0.5, drops to 0 at p = 0.0 or 1.0.
    """
    # Safe clipping prevents NaNs from log(0)
    p_safe = jnp.clip(p, 1e-12, 1.0 - 1e-12)
    return -p_safe * jnp.log(p_safe) - (1.0 - p_safe) * jnp.log(1.0 - p_safe)

@register_probability_acq("margin_seeking")
def margin_seeking(p: jax.Array, target_p: float = 0.95, **kwargs) -> jax.Array:
    """
    Seeks a specific probability threshold (e.g., finding the exact 95% confidence boundary).
    """
    # Negative squared error (maximizing this drives p towards target_p)
    return -(p - target_p)**2