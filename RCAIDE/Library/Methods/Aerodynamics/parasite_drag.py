# RCAIDE/Library/Methods/Aerodynamics/parasite_drag.py
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

from .flat_plate_friction import func_compressible_mixed_flat_plate
from RCAIDE.Library.Methods.Utilities import cubic_spline_blender

# ----------------------------------------------------------------------------------------------------------------------
#  Parasite Drag Methods
# ----------------------------------------------------------------------------------------------------------------------

# TODO: Parasite Drag Methods
# ---------------------------------------------------------
# Individual Component Functions
# ---------------------------------------------------------
@jax.jit
def func_wing_parasite_drag(
    Re,
    M,
    T,
    x_tu,
    x_tl,
    w_mac,
    w_sweep,
    w_tc,
    S_ref,
    S_wet,
    C
):
    """Computes the parasite drag due to wings"""  
   
    # Reynolds number
    Re_w = Re * w_mac  
    
    cf_w_u, k_comp_u, k_reyn_u = func_compressible_mixed_flat_plate(Re_w, M, T, x_tu)
    cf_w_l, k_comp_l, k_reyn_l = func_compressible_mixed_flat_plate(Re_w, M, T, x_tl) 
    
    # Sweep correciton
    cos_sweep = jnp.cos(w_sweep)
    cos2      = cos_sweep * cos_sweep
    M2        = M * M
    beta2     = jnp.maximum(1.0 - M2 * cos2, 1e-8)
    
    ind = M <= 1.
    
    k_w_subsonic = (
        1.0 
        + (2.0 * C * (w_tc * cos2)) / jnp.sqrt(beta2) 
        + (C * C * cos2 * w_tc * w_tc * (1.0 + 5.0 * cos2)) / (2.0 * beta2)
    )

    k_w_raw = jnp.where(M <= 1.0, k_w_subsonic, 1.0)

    h00_val = cubic_spline_blender(M, 0.95, 1.0)
    k_w = k_w_raw * h00_val + 1.0 * (1.0 - h00_val)

    # find the final result
    wing_parasite_drag = k_w * cf_w_u * S_wet / S_ref /2. + k_w * cf_w_l * S_wet / S_ref /2.


    return wing_parasite_drag , k_w, cf_w_u, cf_w_l, k_comp_u, k_comp_l, k_reyn_u, k_reyn_l

@jax.jit
def func_fuselage_parasite_drag():
    """ Pure JAX mathematical evaluation. """
    
    return fuselage_parasite_drag

@jax.jit
def func_nacelle_parasite_drag():
    """ Pure JAX mathematical evaluation. """
    
    return nacelle_parasite_drag

@jax.jit
def func_pylon_parasite_drag():
    """ Pure JAX mathematical evaluation. """
    
    return pylon_parasite_drag

# ---------------------------------------------------------
# 2. STATEFUL FRAMEWORK ROUTER
# ---------------------------------------------------------
def compute_something_stateful(state: "State", system: "System", settings: "Settings"):
    """ Unpacks PyTrees, calls pure math, repacks PyTrees. """
    
    # 1. Unpack
    # val = state.aerodynamics.something
    
    # 2. Call pure function
    # result = func_compute_something(val)
    
    # 3. Pack and return
    # current_state = eqx.tree_at(lambda s: s.aerodynamics.result, state, result)
    
    return state, system, settings
