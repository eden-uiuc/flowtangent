# RCAIDE/Framework/Methods/Aerodynamics/VLM/aerodynamic_coefficients.py
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
#  Lift and Drag Calculation
# ----------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------
# Trefftz Plane Induced Drag
# ---------------------------------------------------------
@jax.jit
def _compute_trefftz_drag(y_ctrl, z_ctrl, x_ctrl, gamma_segments, alpha, rho, v_inf):
    """
    Computes Trefftz plane induced drag using JAX broadcasting.
    Inputs are expected to be shape (n_time, n_spanwise_strips).
    """
    cos_a = jnp.cos(alpha)[:, None]
    sin_a = jnp.sin(alpha)[:, None]
    
    tp_y_ctrl = y_ctrl
    tp_z_ctrl = z_ctrl * cos_a - x_ctrl * sin_a
    
    tp_y_center = (tp_y_ctrl[:, :-1] + tp_y_ctrl[:, 1:]) / 2.0
    tp_z_center = (tp_z_ctrl[:, :-1] + tp_z_ctrl[:, 1:]) / 2.0
    
    dy = tp_y_ctrl[:, 1:] - tp_y_ctrl[:, :-1]
    direction = jnp.sign(dy)
    shed_vortex = direction * (gamma_segments[:, 1:] - gamma_segments[:, :-1])
    
    dy_mat = tp_y_ctrl[:, :, None] - tp_y_center[:, None, :]
    dz_mat = tp_z_ctrl[:, :, None] - tp_z_center[:, None, :]
    
    r = jnp.sqrt(dy_mat**2 + dz_mat**2)
    
    slope = jnp.gradient(tp_z_ctrl, axis=1) / jnp.gradient(tp_y_ctrl, axis=1)
    n_hat_y = jnp.cos(jnp.arctan2(-1.0, slope))
    n_hat_z = jnp.sin(jnp.arctan2(-1.0, slope))
    
    v_mag = shed_vortex[:, None, :] / (2.0 * jnp.pi * jnp.maximum(r, 1e-8))
    v_ind_y = v_mag * (-dz_mat / r)
    v_ind_z = v_mag * (dy_mat / r)
    
    v_induced = jnp.sum(v_ind_y * n_hat_y[:, :, None] + v_ind_z * n_hat_z[:, :, None], axis=2)
    
    ds = jnp.sqrt(jnp.diff(y_ctrl, axis=1)**2 + jnp.diff(z_ctrl, axis=1)**2)
    integrand = v_induced[:, :-1] * gamma_segments[:, :-1] + v_induced[:, 1:] * gamma_segments[:, 1:]
    D_induced = -0.25 * rho * jnp.sum(integrand * ds, axis=1)
    
    return D_induced, v_induced

# ---------------------------------------------------------
# Full Coefficient Calculation
# ---------------------------------------------------------

@jax.jit
def _compute_aerodynamic_coefficients(VD, DCP, GAMMA, EW, v_total, state, system, settings):
    """
    Computes CL, CD, CM, CY, Cl, Cn and induced drag.
    Maintains 1:1 parity with VORLAX LE Suction and Strip integrations.
    """
    # 1. Extract States & Reference Values
    alpha = state.freestream.angles.alpha[:, None]
    beta = state.freestream.angles.beta[:, None]
    v_inf = state.freestream.speed[:, None]
    mach = state.freestream.mach_number[:, None]
    rho = state.freestream.density[:, None]
    
    S_ref = system.reference_geometry.reference_area
    c_ref = system.reference_geometry.reference_chord
    b_ref = system.reference_geometry.reference_span
    cg = system.reference_geometry.center_of_gravity
    x_m, y_m, z_m = cg[0], cg[1], cg[2]
    
    # Pre-compute Trig
    SINALF, COSALF, TANALF = jnp.sin(alpha), jnp.cos(alpha), jnp.tan(alpha)
    SINPSI, COPSI = jnp.sin(beta), jnp.cos(beta)
    COSIN = COSALF * SINPSI * 2.0
    
    # 2. Geometry & Strip Mapping
    le_mask = VD.is_leading_edge
    strip_ids = jnp.cumsum(le_mask) - 1 # Assigns 0, 1, 2... to each spanwise strip!
    
    RNMAX = VD.panels_per_strip
    RNMAX_strip = RNMAX[le_mask]
    CHORD_strip = VD.chord_lengths[le_mask]
    
    # Local Panel Sweep and Dihedral
    dy_LE = VD.panel_corner_b1[le_mask, 1] - VD.panel_corner_a1[le_mask, 1]
    dz_LE = VD.panel_corner_b1[le_mask, 2] - VD.panel_corner_a1[le_mask, 2]
    dx_LE = VD.panel_corner_b1[le_mask, 0] - VD.panel_corner_a1[le_mask, 0]
    
    dist_LE = jnp.maximum(jnp.sqrt(dy_LE**2 + dz_LE**2), 1e-12)
    TLE = dx_LE / dist_LE
    COD = dy_LE / dist_LE  # cos(dihedral)
    SID = dz_LE / dist_LE  # sin(dihedral)
    
    # 3. Panel Forces (SINF, CAXL, BMLE)
    PION = 2.0 / RNMAX
    ADC = 0.5 * PION
    XLE = 0.125 * PION
    SINF = ADC[None, :] * DCP
    
    # Local slope (TX = -nx/nz)
    nx, _, nz = VD.normal_vectors[:, 0], VD.normal_vectors[:, 1], VD.normal_vectors[:, 2]
    TX = -nx / jnp.maximum(jnp.abs(nz), 1e-12)
    
    CAXL_panel = -SINF * TX[None, :] / (1.0 + TX[None, :]**2)
    
    # Derive chordwise index (RK = 1, 2, 3...)
    panel_indices = jnp.arange(len(le_mask))
    strip_start_indices = panel_indices[le_mask][strip_ids]
    RK = panel_indices - strip_start_indices + 1.0
    
    XX = (RK - 0.75) * PION / 2.0
    BMLE_panel = (XLE[None, :] - XX[None, :]) * SINF
    
    # Rolling couple from sideslip
    X = VD.collocation_points[:, 0]
    XTE = (VD.panel_corner_a2[:, 0] + VD.panel_corner_b2[:, 0]) / 2.0
    CORMED = XTE - X
    SICPLE_panel = SINF * CORMED[None, :]
    
    # 4. Integrate Panels into Strips (jax.ops.segment_sum)
    # Vmap the segment sum across the n_time batch dimension
    seg_sum = jax.vmap(lambda arr: jax.ops.segment_sum(arr, strip_ids, num_segments=VD.total_strips))
    
    CNC = seg_sum(SINF)
    CAXL = seg_sum(CAXL_panel)
    BMLE = seg_sum(BMLE_panel)
    SICPLE = seg_sum(SICPLE_panel)
    
    SICPLE = SICPLE * (-1.0) * COSIN * COD[None, :] * 0.5 # GAF approx = 0.5
    
    # 5. VORLAX Leading Edge Suction (CLE & CSUC)
    if settings.analysis.aerodynamics.VORLAX_empirical_corrections:
        # Recreate CLE = sum(EW * Gamma) - EFFINC
        EW_LE = EW[:, le_mask, :] # Shape: (n_time, n_strips, N)
        CLE_ind = jnp.sum(EW_LE * GAMMA[:, None, :], axis=2)
        
        EFFINC_LE = jnp.sum(v_total[:, le_mask, :] * VD.normal_vectors[None, le_mask, :], axis=2)
        CLE = CLE_ind - EFFINC_LE
        
        B2_LE = jnp.square(mach) - 1.0
        T2 = jnp.square(TLE)[None, :]
        STB = jnp.where(B2_LE < T2, jnp.sqrt(jnp.maximum(T2 - B2_LE, 0.0)), 0.0)
        
        CLE = jnp.where(STB > 0, CLE / RNMAX_strip[None, :] / STB, CLE)
        CLE = CLE + 0.5 * DCP[:, le_mask] * jnp.sqrt(XLE[le_mask])[None, :]
        
        SPC = 1.0 # Vortex lift flag hook
        CSUC = 0.5 * jnp.pi * jnp.abs(SPC) * jnp.square(CLE) * STB
        
        # Suction Resolution (TFX, TFZ)
        TFX = jnp.ones_like(CSUC)
        TFZ = -VD.tangent_incidence_angle[le_mask][None, :]
        
        CAXL = CAXL - TFX * CSUC
        CNC = CNC + CSUC * jnp.sqrt(1.0 + T2) * TFZ

    # 6. Body Axis Transformation
    ZETA = VD.tangent_incidence_angle[le_mask][None, :]
    FCOS, FSIN = jnp.cos(ZETA), jnp.sin(ZETA)
    
    BFX = -CNC * FSIN + CAXL * FCOS
    BFY = -(CNC * FCOS + CAXL * FSIN) * SID[None, :]
    BFZ =  (CNC * FCOS + CAXL * FSIN) * COD[None, :]
    
    CNC = CNC * CHORD_strip[None, :]
    BMLE = BMLE * CHORD_strip[None, :]
    
    X_s = VD.collocation_points[le_mask, 0][None, :]
    Y_s = VD.collocation_points[le_mask, 1][None, :]
    Z_s = VD.collocation_points[le_mask, 2][None, :]
    
    BMX = BFZ * Y_s - BFY * (Z_s - z_m) + SICPLE
    BMY = BMLE * COD[None, :] + BFX * (Z_s - z_m) - BFZ * (X_s - x_m)
    BMZ = BMLE * SID[None, :] - BFX * Y_s + BFY * (X_s - x_m)
    
    # 7. Strip Aerodynamic Integration
    s_half = jnp.abs(VD.panel_corner_b1[le_mask, 1] - VD.panel_corner_a1[le_mask, 1])
    ES = 2.0 * s_half
    STRIP_AREA = ES * CHORD_strip
    
    LIFT = (BFZ * COSALF - (BFX * COPSI + BFY * SINPSI) * SINALF) * STRIP_AREA[None, :]
    FY = (BFY * COPSI - BFX * SINPSI) * STRIP_AREA[None, :]
    MOMENT = STRIP_AREA[None, :] * (BMY * COPSI - BMX * SINPSI)
    
    RM = STRIP_AREA[None, :] * (BMX * COSALF * COPSI + BMY * COSALF * SINPSI + BMZ * SINALF)
    YM = STRIP_AREA[None, :] * (BMZ * COSALF - (BMX * COPSI + BMY * SINPSI) * SINALF)
    
    # Global Coefficients
    CL = jnp.sum(LIFT, axis=1) / S_ref
    CY = jnp.sum(FY, axis=1) / S_ref
    CM = jnp.sum(MOMENT, axis=1) / (S_ref * c_ref)
    
    CRTOT = jnp.sum(RM, axis=1) / S_ref
    Cl_roll = -CRTOT / b_ref
    
    CNTOT = jnp.sum(YM, axis=1) / S_ref
    Cn_yaw = -CNTOT / b_ref
    
    # 8. Trefftz Plane Induced Drag
    y_ctrl = jnp.reshape(VD.collocation_points[:, 1], (-1, VD.panels_per_strip[0]))[:, 0]
    z_ctrl = jnp.reshape(VD.collocation_points[:, 2], (-1, VD.panels_per_strip[0]))[:, 0]
    x_ctrl = jnp.reshape(VD.collocation_points[:, 0], (-1, VD.panels_per_strip[0]))[:, 0]
    
    gamma_reshaped = jnp.reshape(GAMMA, (GAMMA.shape[0], -1, VD.panels_per_strip[0]))
    gamma_spanwise = jnp.sum(gamma_reshaped, axis=2)
    
    D_induced, _ = _compute_trefftz_drag(
        y_ctrl[None, :], z_ctrl[None, :], x_ctrl[None, :], 
        gamma_spanwise, alpha[:, 0], rho[:, 0], v_inf[:, 0]
    )
    
    CDi = D_induced / (0.5 * rho[:, 0] * jnp.square(v_inf[:, 0]) * S_ref)
    
    # Profile Drag (CX and CZ projection back to Wind Axes)
    CX = (TANALF[:, 0] * CL - CDi) / (COSALF[:, 0] - SINALF[:, 0] * TANALF[:, 0])
    CZ = (CDi + CX * COSALF[:, 0]) / SINALF[:, 0]
    
    CD = CDi + CX * COSALF[:, 0] * COPSI[:, 0] + CY * SINPSI[:, 0] + CZ * SINALF[:, 0] * COPSI[:, 0]
    
    return CL, CD, CDi, CM, CY, Cl_roll, Cn_yaw

def compute_coefficients(state: "State", system: "System", settings: "Settings"):
    """ Final VLM step to extract global coefficients and append to State. """
    
    analysis = system.analysis_data
    VD = analysis["vortex_distribution"]

    # TODO: Add State bookkeeping for secondary coefficients
    # TODO: Add book-keeping for individual wing coefficients
    CL, CD, CDi, CM, CY, Cl_roll, Cn_yaw = _compute_aerodynamic_coefficients(
        VD,
        analysis["pressure_coefficients"],
        analysis["vortex_strengths"],
        analysis["VORLAX_EW_matrix"],
        analysis["relative_velocity"],
        state,
        system,
        settings
    )
    
    # Update the Vehicle/Segment State with the aerodynamic forces
    state = eqx.tree_at(lambda s: s.aerodynamics.coefficients.lift.total, state, CL)
    state = eqx.tree_at(lambda s: s.aerodynamics.coefficients.drag.total, state, CD)
    state = eqx.tree_at(lambda s: s.aerodynamics.coefficients.drag.induced.total, state, CDi)    
    
    return state, system, settings