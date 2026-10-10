import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# ==========================================
# 1. NORD THEMES (Strictly Blues & Greys)
# ==========================================

nord_dark = go.layout.Template(
    layout=go.Layout(
        plot_bgcolor="#2e3440",
        paper_bgcolor="#2e3440",
        font=dict(color="#eceff4", family="Inter, system-ui, sans-serif"),
        # The four "Nord Frost" blues, plus a light Snow Storm grey for contrast
        colorway=["#88c0d0", "#81a1c1", "#5e81ac", "#8fbcbb", "#e5e9f0"],
        colorscale=dict(
            # A cool, icy sequential scale for when continuous data is used here
            sequential=px.colors.sequential.Blues,
        ),
        xaxis=dict(gridcolor="#4c566a", zerolinecolor="#4c566a", linecolor="#4c566a"),
        yaxis=dict(gridcolor="#4c566a", zerolinecolor="#4c566a", linecolor="#4c566a"),
    )
)

nord_light = go.layout.Template(
    layout=go.Layout(
        plot_bgcolor="#eceff4",
        paper_bgcolor="#eceff4",
        font=dict(color="#2e3440", family="Inter, system-ui, sans-serif"),
        # The same Frost blues, but ending with a dark Polar Night slate
        colorway=["#5e81ac", "#81a1c1", "#88c0d0", "#8fbcbb", "#3b4252"],
        colorscale=dict(
            sequential=px.colors.sequential.Blues,
        ),
        xaxis=dict(gridcolor="#d8dee9", zerolinecolor="#d8dee9", linecolor="#d8dee9"),
        yaxis=dict(gridcolor="#d8dee9", zerolinecolor="#d8dee9", linecolor="#d8dee9"),
    )
)

# ==========================================
# 2. PLASMA THEMES (High Contrast)
# ==========================================

# Samples from the Plasma scale to use for distinct categorical lines
_plasma_colors = ["#f0f921", "#f89441", "#cc4678", "#7e03a8", "#0d0887"]

plasma_dark = go.layout.Template(
    layout=go.Layout(
        plot_bgcolor="#0a0a0a", # Near-black to make the bright yellows/oranges pop
        paper_bgcolor="#0a0a0a",
        font=dict(color="#eeeeee", family="Inter, system-ui, sans-serif"),
        colorway=_plasma_colors,
        colorscale=dict(
            sequential=px.colors.sequential.Plasma,
            diverging=px.colors.diverging.PuOr,
        ),
        xaxis=dict(gridcolor="#222222", zerolinecolor="#333333", linecolor="#333333"),
        yaxis=dict(gridcolor="#222222", zerolinecolor="#333333", linecolor="#333333"),
        coloraxis=dict(colorscale="Plasma"),
    )
)

plasma_light = go.layout.Template(
    layout=go.Layout(
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(color="#111111", family="Inter, system-ui, sans-serif"),
        # Reverse the colors so the dark purples are used first on the white background
        colorway=_plasma_colors[::-1],
        colorscale=dict(
            sequential=px.colors.sequential.Plasma,
            diverging=px.colors.diverging.PuOr,
        ),
        xaxis=dict(gridcolor="#eeeeee", zerolinecolor="#dddddd", linecolor="#dddddd"),
        yaxis=dict(gridcolor="#eeeeee", zerolinecolor="#dddddd", linecolor="#dddddd"),
        coloraxis=dict(colorscale="Plasma"),
    )
)

# ==========================================
# 3. PUBLICATION THEMES (Black & White)
# ==========================================

bw_pub = go.layout.Template(
    layout=go.Layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="black", family="Serif"),
        colorway=["black", "#444444", "#888888", "#bbbbbb"],
        colorscale=dict(
            sequential=px.colors.sequential.Greys,
        ),
        xaxis=dict(
            gridcolor="#eeeeee",
            zerolinecolor="black",
            linecolor="black",
            mirror=True,
            ticks="outside",
            showline=True
        ),
        yaxis=dict(
            gridcolor="#eeeeee",
            zerolinecolor="black",
            linecolor="black",
            mirror=True,
            ticks="outside",
            showline=True
        ),
    )
)

__all__ = [
    "nord_dark",
    "nord_light",
    "plasma_dark",
    "plasma_light",
    "bw_pub",
]

PLOTLY_STYLES = {
    "nord_dark": nord_dark,
    "nord_light": nord_light,
    "plasma_dark": plasma_dark,
    "plasma_light": plasma_light,
    "bw_pub": bw_pub,
}

# 1. Register them globally for string lookup and composition (e.g., "nord_dark+presentation")
for name, template in PLOTLY_STYLES.items():
    pio.templates[name] = template

# 2. Set the global default for the library
# Note: Plotly accepts either the string name or the object itself here.
pio.templates.default = "nord_dark"
