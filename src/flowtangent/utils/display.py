# src/flowtangent/utils/display.py
from typing import Any
import numpy as np


def format_array(v: Any, precision: int = 3, width: int = 10) -> str:
    """Formats numeric values and arrays for terminal printing.

    Converts the input to a NumPy array. Scalars are formatted using 
    scientific notation with explicit width and precision alignment. 
    Multi-dimensional arrays are formatted using `np.array2string`.

    Args:
        v (Any): The numeric value or array-like object to format.
        precision (int, optional): The number of decimal places to display. Defaults to 3.
        width (int, optional): The minimum character width for scalar formatting. Defaults to 10.

    Returns:
        str: The formatted string representation of the array or scalar.
    """
    v_np = np.asarray(v)
    if v_np.size == 1:
        return f"{v_np.item():>{width}.{precision}e}"
    return np.array2string(v_np, precision=precision, separator=", ")


MERMAID_STYLES: dict[str, str] = {
    "default": "",
    "formal": """%%{init: {"theme": "base", "themeVariables": {
        "primaryColor": "#ffffff",
        "primaryBorderColor": "#000000",
        "primaryTextColor": "#000000",
        "lineColor": "#000000",
        "fontFamily": "Times New Roman, serif"
    }}}%%""",
    "modern": """%%{init: {"theme": "base", "themeVariables": {
        "primaryColor": "#f8fafc",
        "primaryBorderColor": "#3b82f6",
        "primaryTextColor": "#0f172a",
        "lineColor": "#94a3b8",
        "fontFamily": "Inter, system-ui, sans-serif"
    }}}%%""",
    "dark": """%%{init: {"theme": "dark", "themeVariables": {
        "primaryColor": "#1e1e1e",
        "primaryBorderColor": "#10b981",
        "primaryTextColor": "#e5e7eb",
        "lineColor": "#10b981",
        "fontFamily": "Fira Code, monospace"
    }}}%%""",
}
"""dict[str, str]: Pre-configured Mermaid JS theme initialization strings for rendering process graphs."""