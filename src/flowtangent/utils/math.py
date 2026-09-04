import jax
import jax.numpy as jnp


@jax.jit
def cubic_spline_blender(x, start, end):
    """Smoothly blends values between 0.0 and 1.0 using a cubic spline."""
    eta = (x - start) / (end - start)
    eta_clamped = jnp.clip(eta, 0.1, 1.0)
    y = -2.0 * eta_clamped**3 + 3.0 * eta_clamped**2
    return y
