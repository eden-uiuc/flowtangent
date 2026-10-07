from typing import Any
import equinox as eqx
import jax
from ..utils import Module
from ..utils.typing import _, _Mod
import optimistix as optx
import optax

def fit_neural_model(
    model: _Mod, 
    args: Any, 
    loss_fn, 
    learning_rate: float = 1e-3, 
    epochs: int = 500,
    rtol: float = 1e-4,
    atol: float = 1e-4
) -> _Mod:
    """Generic helper to run Optimistix minimization using Optax solvers."""
    solver = optx.OptaxMinimiser(optax.adam(learning_rate), rtol=rtol, atol=atol)
    
    sol = optx.minimise(
        loss_fn,
        solver,
        y0=model,
        args=args,
        max_steps=epochs,
        throw=False
    )
    return sol.value

class TransformerBlock(Module):
    """A standard pre-norm transformer block."""
    mha: eqx.nn.MultiheadAttention = _
    mlp: eqx.nn.MLP = _
    ln1: eqx.nn.LayerNorm = _
    ln2: eqx.nn.LayerNorm = _

    def __init__(self, hidden_size: int, num_heads: int, key):
        k1, k2 = jax.random.split(key)
        self.mha = eqx.nn.MultiheadAttention(num_heads, hidden_size, key=k1)
        self.mlp = eqx.nn.MLP(hidden_size, hidden_size, hidden_size * 4, 1, key=k2)
        self.ln1 = eqx.nn.LayerNorm(hidden_size)
        self.ln2 = eqx.nn.LayerNorm(hidden_size)

    def __call__(self, x):
        # x shape: [seq_len, hidden_size]
        x_norm1 = jax.vmap(self.ln1)(x)
        x = x + self.mha(x_norm1, x_norm1, x_norm1)
        
        x_norm2 = jax.vmap(self.ln2)(x)
        x = x + jax.vmap(self.mlp)(x_norm2)
        return x

__all__ = [
    "fit_neural_model",
    "TransformerBlock"
]