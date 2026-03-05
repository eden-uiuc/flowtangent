# RCAIDE/Framework/Methods/Aerodynamics/VLM/initialization
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
    from RCAIDE.Framework.System import System, Aircraft, AircraftReferenceGeometry
    from RCAIDE.Framework.Settings import Settings


# ----------------------------------------------------------------------------------------------------------------------
#  VLM Initialization
# ----------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------
# Geometry Initialization
# ---------------------------------------------------------
def initialize_VLM_geometry(state: "State", system: "Aircraft", settings: "Settings"):
    """
    Parses the vehicle geometry to find the primary reference parameters 
    and packs them into JAX arrays for the VLM solver.
    """
    
    # Standard Python Control Flow (Safe outside of @jax.jit)
    wings = system.wings
    
    if hasattr(wings, 'main_wing'):
        main_wing = wings.main_wing
        c_bar = main_wing.chords.mean_aerodynamic
        x_mac = main_wing.aerodynamic_center[0] + main_wing.origin[0][0]
        z_mac = main_wing.aerodynamic_center[2] + main_wing.origin[0][2]
        b_ref = main_wing.spans.projected
    else:
        c_bar = 0.0
        x_mac = 0.0
        z_mac = 0.0 # Make sure we initialize z_mac too!
        b_ref = 0.0
        
        for wing in wings:
            if not wing.vertical:
                if c_bar <= wing.chords.mean_aerodynamic:
                    c_bar = wing.chords.mean_aerodynamic
                    x_mac = wing.aerodynamic_center[0] + wing.origin[0][0]
                    z_mac = wing.aerodynamic_center[2] + wing.origin[0][2]
                    b_ref = wing.spans.projected

    # 2. Resolve the Center of Gravity / Moment Reference Center
    # Assuming the legacy shape was a 2D array like [[x, y, z]]
    cg_array = system.mass_properties.center_of_gravity
    x_cg = cg_array[0][0]
    z_cg = cg_array[0][2]
    
    if x_cg == 0.0:
        x_m = x_mac
        z_m = z_mac
    else:
        x_m = x_cg
        z_m = z_cg

    # 3. Pack into strict JAX arrays
    # We use jnp.atleast_1d and explicit array shapes to match your jnp.empty structures
    new_ref_geom = system.reference_geometry.__class__(
        mean_aerodynamic_chord = jnp.atleast_1d(c_bar), #type: ignore
        projected_span         = jnp.atleast_1d(b_ref), #type: ignore
        aerodynamic_center     = jnp.array([[x_mac, 0.0, z_mac]]), #type: ignore
        center_of_gravity      = jnp.array([[x_m,   0.0, z_m]]) #type: ignore
    )

    # Add analysis data keys
    initial_analysis_data = {
            "vlm_wings": tuple(),
            "vlm_topology": None,
            "vortex_distribution": None,
            "induced_wake": None,
            "boundary_conditions": None,
            "relative_velocity": None,
            "AICs": None,
            "singularities": None,
            "vortex_strengths": None,
            "pressure_coefficients": None,
            "VORLAX_EW_Matrix": None,
        }
        
    updated_system = eqx.tree_at(
        lambda s: (s.reference_geometry, s.analysis_data), 
        system, 
        (new_ref_geom, initial_analysis_data)
    )
    return state, updated_system, settings
