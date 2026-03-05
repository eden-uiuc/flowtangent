# RCAIDE/Framework/Methods/Aerodynamics/VLM/boundary_conditions.py
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
    from RCAIDE.Framework.Analyses.Aerodynamics.Vortex_Lattice import VLMSettings
    from RCAIDE.Framework.Methods.Aerodynamics.Vortex_Lattice.vortex_distribution import VortexDistribution

# ----------------------------------------------------------------------------------------------------------------------
#  VLM Boundary Conditions (Vortex Strength Right Hand Side Matrix)
# ----------------------------------------------------------------------------------------------------------------------

def compute_vlm_rhs(state: "State", system: "System", settings: "Settings"):
    """
    Computes the Neumann boundary condition (RHS) for the VLM.
    RHS = (V_freestream + V_rotation + V_wake) @ n
    """
    
    vlm_settings: "VLMSettings" = settings.analysis.aerodynamics # type: ignore
    VD: VortexDistribution = system.analysis_data["vortex_distribution"] # type: ignore
    
    # 1. Extract Conditions (assuming scalar/1D values for the current timestep)
    alpha = state.aerodynamics.angles.alpha
    beta  = state.aerodynamics.angles.beta
    v_inf = state.freestream.speed
    
    p = state.stability.static.roll_rate
    q = state.stability.static.pitch_rate
    r = state.stability.static.yaw_rate
    
    omega = jnp.array([p, q, r])
    
    # 2. Build Freestream Velocity Vector
    # V_fs = [V*cos(a)*cos(b), V*cos(a)*sin(b), V*sin(a)]
    v_fs = jnp.array([
        v_inf * jnp.cos(alpha) * jnp.cos(beta),
        v_inf * jnp.cos(alpha) * jnp.sin(beta),
        v_inf * jnp.sin(alpha)
    ])
    
    # 3. Compute Rotational Velocity at every control point: V_rot = -(Omega x r)
    # r is the vector from the moment center (CG) to the collocation point
    moment_center = system.reference_geometry.center_of_gravity
    r_giro = VD.collocation_points - moment_center
    
    # jnp.cross automatically broadcasts the (3,) omega against the (N, 3) r_giro array!
    v_rot = -jnp.cross(omega, r_giro)
    
    # 4. Sum the total relative velocity (N, 3)
    v_total = v_fs + v_rot
    if vlm_settings.model_propeller_wake:
        # TODO: Convert BEMT and add wake calculation to VLM Process
        raise ValueError("Propeller wake modelling is unsupported pending BEMT inclusion in RCAIDE.")
        v_total = v_total + system.analysis_data["induced_wake"] #type: ignore
        
    # 5. Take the Dot Product with the Panel Normals!
    # Normalize by V_inf to match the standard VLM coefficient formulation
    v_unit = v_total / v_inf
    
    # Dot product: sum(V * N, axis=1)
    rhs_array = jnp.sum(v_unit * VD.normal_vectors, axis=1)

    updated_analysis_data = system.analysis_data | {"boundary_conditions": rhs_array, "relative_velocity": v_total}

    updated_system = eqx.tree_at(lambda s: s.analysis_data, system, updated_analysis_data)
    
    return state, updated_system, settings
