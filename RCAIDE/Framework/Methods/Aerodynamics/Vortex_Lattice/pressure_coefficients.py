# RCAIDE/Framework/Methods/Aerodynamics/VLM/pressure_coefficients.py
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
#  Compute VLM Pressure Coefficients
# ----------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
@jax.jit
def compute_pressure_coefficients(VD, v_total, GAMMA, v_inf):
    """ 
    Computes the differential pressure coefficient (Delta C_P) for all panels.
    Input shapes: v_total (n_time, N, 3), GAMMA (n_time, N)
    """
    # 1. Local Velocity Components (normalized by V_inf)
    Vx_local = v_total[:, :, 0] / v_inf
    Vy_local = v_total[:, :, 1] / v_inf
    
    # 2. Local Panel Geometry (Sweep Tangents and Dihedral)
    chord = VD.chord_lengths
    n_cw = VD.panels_per_strip
    dx_frac = 1.0 / n_cw
    
    # Front edge sweep (a1 to b1)
    dy_A = VD.panel_corner_b1[:, 1] - VD.panel_corner_a1[:, 1]
    dz_A = VD.panel_corner_b1[:, 2] - VD.panel_corner_a1[:, 2]
    dx_A = VD.panel_corner_b1[:, 0] - VD.panel_corner_a1[:, 0]
    dy_z_A = jnp.maximum(jnp.sqrt(dy_A**2 + dz_A**2), 1e-12) # Prevent DivByZero
    
    TANA = dx_A / dy_z_A
    cos_DL = dy_A / dy_z_A # Cosine of local dihedral
    
    # Back edge sweep (a2 to b2)
    dy_B = VD.panel_corner_b2[:, 1] - VD.panel_corner_a2[:, 1]
    dz_B = VD.panel_corner_b2[:, 2] - VD.panel_corner_a2[:, 2]
    dx_B = VD.panel_corner_b2[:, 0] - VD.panel_corner_a2[:, 0]
    dy_z_B = jnp.maximum(jnp.sqrt(dy_B**2 + dz_B**2), 1e-12)
    
    TANB = dx_B / dy_z_B
    
    # 3. Segmented Cumulative Sum (GANT)
    gamma_over_c = GAMMA / chord[None, :]
    
    def scan_fn(a, b):
        # a and b are tuples: (value, is_leading_edge_flag)
        v1, le1 = a
        v2, le2 = b
        # If element 'b' is a leading edge, it resets the sum to just v2!
        return jnp.where(le2, v2, v1 + v2), le1 | le2
    
    # Broadcast the 1D LE flag to match the (n_time, N) matrix
    is_le = jnp.broadcast_to(VD.is_leading_edge[None, :], gamma_over_c.shape)
    
    # Run the ultra-fast parallel scan
    gant_full, _ = jax.lax.associative_scan(scan_fn, (gamma_over_c, is_le), axis=1)
    
    # Shift the array by 1 to exclude the current panel (matching VORLAX GANT)
    GANT = jnp.where(is_le, 0.0, jnp.roll(gant_full, shift=1, axis=1))
    
    # 4. Sweep / Sideslip Correction (DCPSID)
    GLAT = GANT * (TANA - TANB)[None, :] - gamma_over_c * TANB[None, :]
    FORLAT = 2.0 * Vy_local 
    
    # DX fraction is 1 / RNMAX
    DCPSID = FORLAT * cos_DL[None, :] * GLAT / dx_frac[None, :]
    
    # 5. Axial Load Component (GNET)
    # GNET = GAMMA * FACTOR * RNMAX / CHORD
    GNET = GAMMA * Vx_local * n_cw[None, :] / chord[None, :]
    
    # 6. Final Delta CP
    DCP = 2.0 * GNET + DCPSID
    
    return DCP

# ---------------------------------------------------------
#  STATEFUL VERSION
# ---------------------------------------------------------
def compute_panel_pressures(state: "State", system: "System", settings: "Settings"):
    """ Calculates the differential pressure coefficient (Delta C_P) for all VLM panels. """
    
    analysis = system.analysis_data
    VD = analysis["vortex_distribution"]
    
    v_total = analysis["relative_velocity"]
    GAMMA = analysis["vortex_strengths"]
    v_inf = state.freestream.speed
    
    # Needs to be reshaped to broadcast across (n_time, 1)
    v_inf_array = v_inf[:, None] 
    
    DCP = compute_pressure_coefficients(VD, v_total, GAMMA, v_inf_array)
    
    updated_analysis_data = analysis | {
        "pressure_coefficients": DCP
    }
    
    updated_system = eqx.tree_at(lambda s: s.analysis_data, system, updated_analysis_data)
    
    return state, updated_system, settings
