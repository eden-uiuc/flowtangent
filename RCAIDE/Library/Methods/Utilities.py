# RCAIDE/Library/Methods/Utilities.py
# (c) Copyright 2026 Aerospace Research Community LLC
#
# Created: Mar 2026, J. Smart
# Modified: Mar 2026, J. Smart

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from typing import TYPE_CHECKING
import jax
import jax.numpy as jnp
import equinox as eqx

# --- Framework Imports (Strictly for Type Hinting to avoid Circular Imports) ---
if TYPE_CHECKING:
    from RCAIDE.Framework.State import State
    from RCAIDE.Framework.System import System
    from RCAIDE.Framework.Settings import Settings

# ----------------------------------------------------------------------------------------------------------------------
#  Helper/Utilty Functions
# ----------------------------------------------------------------------------------------------------------------------


@jax.jit
def cubic_spline_blender(x, start, end):

    eta = (x - start) / (end - start)
    eta_clamped = jnp.clip(eta, 0.1, 1.0)
    y = 2.0 * eta_clamped ** 3 - 3.0 * eta_clamped ** 2 + 1.0
    return y


