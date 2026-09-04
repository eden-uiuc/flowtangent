import numpy as np


def format_array(v, precision=3, width=10):
    """Formats numeric values and arrays for terminal printing."""
    v_np = np.asarray(v)
    if v_np.size == 1:
        return f"{v_np.item():>{width}.{precision}e}"
    return np.array2string(v_np, precision=precision, separator=', ')

MERMAID_STYLES = {
    "default": "",
    "formal": """%%{init: {'theme': 'base', 'themeVariables': {
        'primaryColor': '#ffffff',
        'primaryBorderColor': '#000000',
        'primaryTextColor': '#000000',
        'lineColor': '#000000',
        'fontFamily': 'Times New Roman, serif'
    }}}%%""",
    "modern": """%%{init: {'theme': 'base', 'themeVariables': {
        'primaryColor': '#f8fafc',
        'primaryBorderColor': '#3b82f6',
        'primaryTextColor': '#0f172a',
        'lineColor': '#94a3b8',
        'fontFamily': 'Inter, system-ui, sans-serif'
    }}}%%""",
    "dark": """%%{init: {'theme': 'dark', 'themeVariables': {
        'primaryColor': '#1e1e1e',
        'primaryBorderColor': '#10b981',
        'primaryTextColor': '#e5e7eb',
        'lineColor': '#10b981',
        'fontFamily': 'Fira Code, monospace'
    }}}%%""",
}
