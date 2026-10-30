from typing import Any

import equinox as eqx
import jax
import optax
from beartype.typing import Any, Callable, Literal
from tqdm.auto import tqdm
import plotly.graph_objects as go

from ..utils import Module
from ..utils.typing import _


def train_model(
    model: Any, 
    data: Any, 
    loss_fn: Callable, 
    learning_rate: float = 1e-3, 
    epochs: int = 50,
    plot_every: int = 1,
    logging_mode: Literal["auto", "notebook", "terminal"] = "auto",
):
    # 1. Environment Auto-Detection
    if logging_mode == "auto":
        try:
            from IPython import get_ipython
            # ZMQInteractiveShell means Jupyter Notebook/Lab or Colab
            shell = get_ipython().__class__.__name__
            logging_mode = "notebook" if shell == 'ZMQInteractiveShell' else "terminal"
        except (NameError, ImportError, AttributeError):
            logging_mode = "terminal"

    # 2. Setup Optax & JIT (Identical to your current setup)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def make_step(current_model, state, batch):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(current_model, batch)
        updates, state = optimizer.update(grads, state, current_model)
        return eqx.apply_updates(current_model, updates), state, loss

    is_dataloader = hasattr(data, "dataset") and hasattr(data, "__iter__")
    
    # 3. Initialize Tracking
    loss_history = []
    pbar = tqdm(range(epochs), desc=f"Training Neural Model ({logging_mode} mode)")
    
    # Pre-allocate the live widget if in a notebook
    if logging_mode == "notebook":
        from IPython.display import display
        fig = go.FigureWidget()
        fig.add_scatter(y=[], mode='lines', name='Loss', line=dict(color='#2563eb'))
        fig.update_layout(title="Training Convergence", yaxis_type="log", template="plotly_white")
        display(fig)

    # 4. Training Loop
    for epoch in pbar:
        epoch_loss = 0.0
        batches = 0
        
        if is_dataloader:
            
            batch_pbar = tqdm(data, desc=f"Epoch {epoch+1} Batches", leave=False)
            
            for i, batch in enumerate(batch_pbar):
                batch_pbar.set_description(f"Epoch {epoch+1} Batches")
                
                if epoch == 0 and i == 0:
                    batch_pbar.set_description("Compiling XLA (Wait for it...)")

                model, opt_state, loss = make_step(model, opt_state, batch)
                epoch_loss += loss.item()
                batches += 1
                        
                batch_pbar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})
            
            epoch_loss /= batches
        
        else:
            model, opt_state, loss = make_step(model, opt_state, data)
            epoch_loss = loss.item()
            
        loss_history.append(epoch_loss)
        pbar.set_postfix({"Loss": f"{epoch_loss:.4f}"})
        
        # Live update (Notebook only)
        if logging_mode == "notebook" and (epoch + 1) % plot_every == 0:
            fig.data[0].y = loss_history  #type: ignore

    # 5. Terminal Fallback
    if logging_mode == "terminal":
        final_fig = go.Figure(go.Scatter(y=loss_history, mode='lines', line=dict(color='#2563eb')))
        final_fig.update_layout(title="SVGP Training Convergence", yaxis_type="log", template="plotly_white")
        final_fig.write_html("svgp_convergence.html")
        tqdm.write("Training complete. Convergence plot saved to 'svgp_convergence.html'")

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


__all__ = ["train_model", "TransformerBlock"]
