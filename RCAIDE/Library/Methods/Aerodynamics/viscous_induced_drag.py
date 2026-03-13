# RCAIDE/Library/Methods/Aerodynamics/induced_drag.py
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
#  Viscous Induced Drag
# ----------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------
# 1. PURE LIBRARY FUNCTION (Math Only)
# ---------------------------------------------------------
@jax.jit
def func_viscous_induced_drag(
    CL: float | jnp.ndarray,
    parasite_drag: float | jnp.ndarray,
    viscous_lift_factor: float | jnp.ndarray = 0.38,
):
    """Evaluates viscous induced drag based on parasite drag and drag factor"""
    
    return viscous_lift_factor * parasite_drag * (CL**2)

# ---------------------------------------------------------
# 2. STATEFUL FRAMEWORK ROUTER
# ---------------------------------------------------------
def compute_viscous_induced_drag(state: "State", system: "System", settings: "Settings"):
    """ Computes system and wing viscous induced drag """
    
    # Total System
    CL_all      = state.aerodynamics.coefficients.lift.total
    CDp_all     = state.aerodynamics.coefficients.drag.parasite.total
    K           = settings.analysis.aerodynamics.correction.viscous_lift_drag

    CDiv_all    = func_viscous_induced_drag(CL_all, CDp_all, K)
    total_induced_drag = state.aerodynamics.coefficients.drag.induced.inviscid.total + CDiv_all
    
    CL_w        = state.aerodynamics.coefficients.lift.inviscid.wings
    CDp_w       = state.aerodynamics.coefficients.drag.parasite.wings

    CDiv_w      = func_viscous_induced_drag(CL_w, CDp_w, K)
    wing_induced_drag = state.aerodynamics.coefficients.drag.induced.inviscid.wings + CDiv_w

    updated_induced_drag = eqx.tree_at(lambda i:
        (
            i.total, i.viscous.total, i.viscous.wings
        ), state.aerodynamics.coefficients.drag.induced,
        (
            total_induced_drag, CDiv_all, wing_induced_drag
        )
    )

    updated_state = eqx.tree_at(lambda s: s.aerodynamics.coefficients.drag.induced, state, updated_induced_drag)
    
    return updated_state, system, settings
