# src/flowtangent/utils/display.py
from typing import Any

import numpy as np


def format_array(v: Any, precision: int = 3, width: int = 10) -> str:
    """Formats numeric values and arrays for terminal printing.

    Produces a display string for solver reports. Any input containing exactly
    one element uses right-aligned scientific notation; all other arrays use
    NumPy's array string formatting, including its truncation settings.

    ### Mathematical Formulation
    A one-element value is displayed as $m$ followed by a base-10 exponent $e$,
    where $m 10^e$ approximates $v$ at the requested precision for finite nonzero values.

    ### Usage Example
    ```python
    import jax.numpy as jnp
    from flowtangent.utils import format_array

    assert format_array(jnp.array([1.0])) == " 1.000e+00"
    ```

    Args:
        v (Any): A numeric scalar or ND array-like object convertible to NumPy.
        precision (int, optional): Digits after the decimal point for one-element
            inputs; otherwise passed to `np.array2string`. Defaults to 3.
        width (int, optional): Minimum character width for one-element inputs only.
            Longer representations are not truncated. Defaults to 10.

    Returns:
        str: The formatted string representation of the array or scalar.

    Raises:
        ValueError: If a one-element value or its format specification does not
            support scientific notation, or NumPy conversion/formatting fails.
        TypeError: If a value does not support the requested numeric formatting.
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
