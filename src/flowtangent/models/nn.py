from typing import Any, Callable

import equinox as eqx
import jax
import optax

from ..utils import Module
from ..utils.typing import _


def fit_neural_model(
    model: Any,
    data: Any,
    loss_fn: Callable,
    learning_rate: float = 1e-3,
    epochs: int = 50
):
    """
    Universal Optax training loop. 
    Accepts either a tuple of full-batch arrays (X, Y) or a PyTorch DataLoader.
    """
    # 1. Setup Optax optimizer (filters out static metadata automatically)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # 2. Compile the single-step update
    @eqx.filter_jit
    def make_step(current_model, state, batch):
        # filter_value_and_grad cleanly handles taking derivatives of PyTrees
        loss, grads = eqx.filter_value_and_grad(loss_fn)(current_model, batch)
        updates, state = optimizer.update(grads, state, current_model)
        new_model = eqx.apply_updates(current_model, updates)
        return new_model, state, loss

    # 3. Duck-type check: Is it a DataLoader or a Tuple of Arrays?
    is_dataloader = hasattr(data, "dataset") and hasattr(data, "__iter__")

    # 4. Execute the Training Loop
    for epoch in range(epochs):
        if is_dataloader:
            # Mini-batch out-of-core training
            for batch in data:
                model, opt_state, loss = make_step(model, opt_state, batch)
        else:
            # Full-batch training (data is just the tuple args)
            model, opt_state, loss = make_step(model, opt_state, data)

    return model

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
