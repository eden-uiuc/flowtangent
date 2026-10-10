from typing import Optional

import numpy as np
import plotly.graph_objects as go

from . import styles


def plot_airfoil(
    airfoil,
    title: Optional[str] = None,
    show_markers: bool = False,
    show_camber: bool = False,
    theme: go.layout.Template | str = styles.nord_dark,
    save_path: Optional[str] = None,
    show: bool = False,
) -> go.Figure:
    """
    Plots the Airfoil class geometry using Plotly.
    Enforces a 1:1 aspect ratio so the thickness and camber are visually accurate.
    """
    plot_title = title if title else f"Airfoil Geometry: {airfoil.name}"

    # Plotly expects standard numpy arrays, safely cast from JAX arrays
    x_up =  np.asarray(airfoil.x_upper)
    y_up =  np.asarray(airfoil.y_upper)
    x_low = np.asarray(airfoil.x_lower)
    y_low = np.asarray(airfoil.y_lower)

    mode = "lines+markers" if show_markers else "lines"

    fig = go.Figure()

    # Upper Surface
    fig.add_trace(go.Scatter(x=x_up, y=y_up, mode=mode, name="Upper Surface", line=dict(width=2), marker=dict(size=4)))

    # Lower Surface
    fig.add_trace(
        go.Scatter(x=x_low, y=y_low, mode=mode, name="Lower Surface", line=dict(width=2), marker=dict(size=4))
    )

    # Camber Line
    if show_camber:
        camber_y = np.asarray(airfoil.camber)
        fig.add_trace(
            go.Scatter(
                x=x_low,
                y=camber_y,
                mode=mode,
                name="Camber Line",
                line=dict(width=2, dash="dash"),
                marker=dict(size=4, symbol="cross"),
            )
        )

    # Layout: The 1:1 aspect ratio is mandatory for airfoil visualization
    fig.update_layout(
        title=plot_title,
        xaxis_title="x/c",
        yaxis_title="y/c",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        hovermode="x unified",
        template=theme,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        margin=dict(l=40, r=40, t=60, b=40),
    )

    if save_path:
        if save_path.endswith(".html"):
            fig.write_html(save_path)
        else:
            fig.write_image(save_path)

    if show:
        fig.show()

    return fig
