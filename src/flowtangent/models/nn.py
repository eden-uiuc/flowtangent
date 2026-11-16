from typing import Any, Optional, TypeAlias
from collections import defaultdict

import equinox as eqx
import jax
import optax
from beartype.typing import Any, Callable, Literal
from tqdm.auto import tqdm
import plotly.graph_objects as go
import numpy as np

from ..utils import Module
from ..utils.typing import _
from ..utils.data import DataLoader, LatentDataLoader, SlicedDataLoader

_loader: TypeAlias = DataLoader | LatentDataLoader | SlicedDataLoader


def train_model(
    model: Any, 
    train_data: _loader,
    loss_fn: Callable,
    *,
    learning_rate: float = 1e-3, 
    epochs: int = 50,
    plot_every: int = 1,
    logging_mode: Literal["off", "auto", "notebook", "terminal"] = "auto",
    stop_fn:Optional[Callable]=None,
    test_data: Optional[_loader],
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

    # 2. Setup Optax & JIT
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def make_step(current_model, opt_state, batch):
        (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(current_model, batch)
        updates, updated_state = optimizer.update(grads, opt_state, current_model)
        updated_model =  eqx.apply_updates(current_model, updates)
        return updated_model, updated_state, loss, metrics

    is_dataloader = hasattr(train_data, "dataset") and hasattr(train_data, "__iter__")
    
    # 3. Initialize Tracking
    if logging_mode == "off":
        pbar = range(epochs)
    else:
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
        
        # Initialize tracking
        train_trackers = defaultdict(list)

        if logging_mode=="off":  # Just iterate over the data
            batch_pbar = train_data
        else:  # Set up TQDM progress par
            batch_pbar = tqdm(train_data, leave=False)
        
        for i, batch in enumerate(batch_pbar):
            if not logging_mode=="off":
                batch_pbar.set_description(f"Epoch {epoch+1} Batches")  #type: ignore
                if epoch == 0 and i == 0:
                    batch_pbar.set_description("Compiling XLA ...")  #type: ignore

            model, opt_state, train_loss, train_metrics = make_step(model, opt_state, batch)

            if not logging_mode=="off":
                batch_pbar.set_postfix({"Batch Loss": f"{train_loss.item():.4f}"})  #type: ignore

            # Update tracking
            train_trackers["loss"].append(train_loss)
            for m, v in train_metrics.items():
                train_trackers[m].append(v)

        epoch_metrics = {"epoch": epoch}
        for m, v in train_trackers.items():
            epoch_metrics[f"train_{m}"] = float(np.mean(v))  #type: ignore

        if test_data is not None:
            test_trackers = defaultdict(list)

            if logging_mode=="off":
                test_pbar = test_data
            else:
                test_pbar = tqdm(test_data, leave=False)

            for j, batch in enumerate(test_pbar):
                if not logging_mode=="off":
                    test_pbar.set_description(f"Epoch {epoch+1} Test Batches")  #type: ignore

                (test_loss, test_metrics), _ = eqx.filter_jit(
                    eqx.filter_value_and_grad(loss_fn, has_aux=True)
                )(model, batch)
                
                test_trackers["loss"].append(test_loss)
                for m, v in test_metrics.items():
                    test_trackers[m].append(v)

            for m, v in test_trackers.items():
                epoch_metrics[m] = float(np.mean(v))  #type: ignore
            
        if not logging_mode=="off":
            pbar.set_postfix({"Loss": f"{epoch_metrics["train_loss"]:.4f}"})  #type: ignore
        
        # Live update (Notebook only)
        if logging_mode == "notebook" and (epoch + 1) % plot_every == 0:
            with fig.batch_update():
                fig.data[0].y = tuple(epoch_metrics["loss"])

        train_metrics["epoch"] = epoch
        if stop_fn is not None:
            if stop_fn(train_metrics, model):
                break

    # 5. Terminal Fallback
    if logging_mode == "terminal":
        final_fig = go.Figure(go.Scatter(y=epoch_metrics["loss"], mode='lines', line=dict(color='#2563eb')))
        final_fig.update_layout(title="SVGP Training Convergence", yaxis_type="log", template="plotly_white")
        final_fig.write_html("svgp_convergence.html")
        tqdm.write("Training complete. Convergence plot saved to 'svgp_convergence.html'")

    return model, opt_state, train_loss, epoch_metrics


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
