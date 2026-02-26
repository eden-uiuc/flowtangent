from jaxopt import ScipyRootFinding
import jax.numpy as jnp
from jax import grad

def fn(x, b):

    return x ** 2 - b


def intercept(b, x0=jnp.array([9.])):

    root = ScipyRootFinding(
                method='hybr',
                optimality_fun=fn,
                tol = 1e-4,
                jit=False,  #TODO: Test JIT compilation
            )

    results, _ = root.run(x0, b)

    return results[0]

res = intercept(3.)
b_grad = grad(intercept)
f_b = b_grad(3.)
print("Done!")