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
    from RCAIDE.Framework.Analyses.Aerodynamics.Vortex_Lattice import VLMSettings

# ----------------------------------------------------------------------------------------------------------------------
#  Lift and Drag Calculation
# ----------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------
# Trefftz Plane Induced Drag
# ---------------------------------------------------------
@jax.jit
def _compute_trefftz_drag(y_ctrl, z_ctrl, x_ctrl, gamma_segments, alpha, rho):
    """
    Computes Trefftz plane induced drag using JAX broadcasting.
    """
    n_time, n_strips = gamma_segments.shape
    
    # Force coordinates to (n_time, n_strips)
    y_ctrl = jnp.broadcast_to(y_ctrl.reshape(-1, n_strips), (n_time, n_strips))
    z_ctrl = jnp.broadcast_to(z_ctrl.reshape(-1, n_strips), (n_time, n_strips))
    x_ctrl = jnp.broadcast_to(x_ctrl.reshape(-1, n_strips), (n_time, n_strips))
    
    # Force alpha to (n_time, 1)
    alpha = alpha.reshape(n_time, 1)
    cos_a = jnp.cos(alpha)
    sin_a = jnp.sin(alpha)
    
    # Trefftz Plane Transformation
    tp_y_ctrl = y_ctrl
    tp_z_ctrl = z_ctrl * cos_a - x_ctrl * sin_a  # Now guaranteed perfectly (16, 42)!
    
    tp_y_center = (tp_y_ctrl[:, :-1] + tp_y_ctrl[:, 1:]) / 2.0
    tp_z_center = (tp_z_ctrl[:, :-1] + tp_z_ctrl[:, 1:]) / 2.0
    
    dy = tp_y_ctrl[:, 1:] - tp_y_ctrl[:, :-1]
    direction = jnp.sign(dy)
    shed_vortex = direction * (gamma_segments[:, 1:] - gamma_segments[:, :-1])
    
    # Distance Matrices
    dy_mat = tp_y_ctrl[:, :, None] - tp_y_center[:, None, :] 
    dz_mat = tp_z_ctrl[:, :, None] - tp_z_center[:, None, :] 
    
    # Safe Distance
    r = jnp.sqrt(dy_mat**2 + dz_mat**2 + 1e-16)
    r_safe = jnp.maximum(r, 1e-8) # Clamp once and use everywhere
    
    # 2. Gradient-Safe Normal Vectors (Fixes the Vertical Tail divide-by-zero)
    dy_ctrl = jnp.gradient(tp_y_ctrl, axis=1)
    dz_ctrl = jnp.gradient(tp_z_ctrl, axis=1)
    
    ctrl_mag = jnp.sqrt(dy_ctrl**2 + dz_ctrl**2 + 1e-16)
    
    n_hat_y = dz_ctrl / ctrl_mag
    n_hat_z = -dy_ctrl / ctrl_mag
    
    # 3. Induced Velocities (Strictly using r_safe)
    v_mag = shed_vortex[:, None, :] / (2.0 * jnp.pi * r_safe)
    v_ind_y = v_mag * (-dz_mat / r_safe)
    v_ind_z = v_mag * (dy_mat / r_safe)
    
    v_induced = jnp.sum(v_ind_y * n_hat_y[:, :, None] + v_ind_z * n_hat_z[:, :, None], axis=2)
    
    # Integrate Drag
    ds = jnp.sqrt(jnp.diff(y_ctrl, axis=1)**2 + jnp.diff(z_ctrl, axis=1)**2)
    integrand = v_induced[:, :-1] * gamma_segments[:, :-1] + v_induced[:, 1:] * gamma_segments[:, 1:]
    
    rho = rho.reshape(n_time) # Ensure 1D for safety
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
    # Extract States & Reference Values
    alpha   = state.aerodynamics.angles.alpha
    beta    = state.aerodynamics.angles.beta
    v_inf   = state.freestream.speed
    mach    = state.freestream.mach_number
    rho     = state.freestream.density
    
    S_ref = system.areas.reference
    c_ref = system.reference_geometry.mean_aerodynamic_chord
    b_ref = system.reference_geometry.projected_span
    
    cg    = system.reference_geometry.center_of_gravity
    x_m   = cg[:, 0][:, None]
    y_m   = cg[:, 1][:, None]
    z_m   = -cg[:, 2][:, None]
    
    # Pre-compute Trig
    sin_alpha, cos_alpha, tan_alpha = jnp.sin(alpha), jnp.cos(alpha), jnp.tan(alpha)
    sin_beta, cos_beta = jnp.sin(beta), jnp.cos(beta)
    crosswind_factor = cos_alpha * sin_beta * 2.0
    
    # Geometry & Strip Mapping
    # JAX-compatible leading edge mask
    le_mask_float = VD.is_leading_edge.astype(jnp.float32)
    strip_ids = jnp.cumsum(VD.is_leading_edge) - 1 # Assigns 0, 1, 2... to each spanwise strip
    
    panels_per_strip = VD.panels_per_strip
    stripwise_panels = jax.ops.segment_max(VD.panels_per_strip, strip_ids, num_segments=VD.total_strips)
    stripwise_chords = jax.ops.segment_max(VD.chord_lengths, strip_ids, num_segments=VD.total_strips)
    
    # Local Panel Sweep and Dihedral (All panels)
    dx_all = VD.panel_corner_b1[:, 0] - VD.panel_corner_a1[:, 0]
    dy_all = VD.panel_corner_b1[:, 1] - VD.panel_corner_a1[:, 1]
    dz_all = VD.panel_corner_b1[:, 2] - VD.panel_corner_a1[:, 2]

    # Extract the leading edge values using the float mask and segment_sum
    # (The trailing edge panels are zeroed out, then collapsed into the strips)
    dy_LE = jax.ops.segment_sum(dy_all * le_mask_float, strip_ids, num_segments=VD.total_strips)
    dz_LE = jax.ops.segment_sum(dz_all * le_mask_float, strip_ids, num_segments=VD.total_strips)
    dx_LE = jax.ops.segment_sum(dx_all * le_mask_float, strip_ids, num_segments=VD.total_strips)
    
    dihedral_length_LE = jnp.maximum(jnp.sqrt(dy_LE**2 + dz_LE**2), 1e-12)
    true_sweep_LE = dx_LE / dihedral_length_LE # leading edge sweep in the plane of the wing
    cos_dihedral = jnp.abs(dy_LE) / dihedral_length_LE  # cos(dihedral), absolute value for negative sign in left hand sings
    sin_dihedral = jnp.sign(dy_LE) * dz_LE / dihedral_length_LE  # sin(dihedral) - multiply by dy_LE so opposed vectors cancel
    
    # Panel Forces
    panel_dx_nondim = 1.0 / panels_per_strip  # Fraction of overall chord length in each panel
    quarter_chord_offset = 0.25 * panel_dx_nondim  # Location of leading edge vortex for each panel
    panel_normal_coeff = panel_dx_nondim[None, :] * DCP  # Fraction of chordwise delta Cp for each panel
    
    # Local slope (TX = -nx/nz)
    nx, _, nz = VD.normal_vectors[:, 0], VD.normal_vectors[:, 1], VD.normal_vectors[:, 2] # Z points down by convention
    panel_chordwise_slope = -nx / jnp.maximum(jnp.abs(nz), 1e-12)
    
    panel_axial_coeff = -panel_normal_coeff * panel_chordwise_slope[None, :] / (1.0 + panel_chordwise_slope[None, :]**2)
    
    # Derive chordwise index
    panel_indices = jnp.arange(VD.collocation_points.shape[0])
    strip_starts = jax.ops.segment_min(panel_indices, strip_ids, num_segments=VD.total_strips)
    strip_start_indices = strip_starts[strip_ids]
    chordwise_indices = panel_indices - strip_start_indices + 1.0
    
    vortex_x_nondim = (chordwise_indices - 0.75) * panel_dx_nondim
    panel_pitching_moment = (quarter_chord_offset[None, :] - vortex_x_nondim[None, :]) * panel_normal_coeff
    
    # Rolling couple from sideslip
    collocation_x = VD.collocation_points[:, 0]
    trailing_edge_x_avg = (VD.panel_corner_a2[:, 0] + VD.panel_corner_b2[:, 0]) / 2.0
    rear_quarter_x = trailing_edge_x_avg - collocation_x
    panel_sideslip_couple = panel_normal_coeff * rear_quarter_x[None, :]
    
    # Integrate Panels into Strips (jax.ops.segment_sum)
    # Vmap the segment sum across the n_time batch dimension
    seg_sum = jax.vmap(lambda arr: jax.ops.segment_sum(arr, strip_ids, num_segments=VD.total_strips))
    
    # Sum and linearly dimensionalize coefficients
    normal_coeff    = seg_sum(panel_normal_coeff)    * stripwise_chords[None, :]
    axial_coeff     = seg_sum(panel_axial_coeff)     * stripwise_chords[None, :] 
    pitching_moment = seg_sum(panel_pitching_moment) * (stripwise_chords[None, :] ** 2)
    
    sideslip_couple = seg_sum(panel_sideslip_couple) * stripwise_chords[None, :]
    sideslip_couple = sideslip_couple * (-1.0) * crosswind_factor * cos_dihedral[None, :] * 0.5 # GAF approx = 0.5
    
    # VORLAX Leading Edge Suction
    if settings.analysis.aerodynamics.VORLAX_empirical_corrections:
        EW_squeezed = EW.squeeze(1)
        EW_masked   = EW_squeezed * le_mask_float[None, :, None]
        EW_LE       = seg_sum(EW_masked)
        
        v_total_masked  = v_total * le_mask_float[None, :, None]
        v_total_LE      = seg_sum(v_total_masked)

        # Use jax.ops directly for static geometry, and broadcast the mask [:, None]
        normals_masked  = VD.normal_vectors * le_mask_float[:, None]
        normals_LE      = jax.ops.segment_sum(normals_masked, strip_ids, num_segments=VD.total_strips)
        
        induced_velocity_LE     = jnp.sum(EW_LE * GAMMA[:, None, :], axis=-1)
        eff_incidence_LE        = jnp.sum(v_total_LE * normals_LE[None, :, :], axis=-1)
        singularity_strength_LE = induced_velocity_LE - eff_incidence_LE
        
        prandtl_glauert_beta_sq = jnp.square(mach) - 1.0
        sweep_sq = jnp.square(true_sweep_LE)[None, :]
        subsonic_LE_factor = jnp.where(
            prandtl_glauert_beta_sq < sweep_sq,
            jnp.sqrt(jnp.maximum(sweep_sq - prandtl_glauert_beta_sq, 0.0)),
            0.0
        )
        
        # Use jax.ops for the 1D structural XLE array
        strip_quarter_chord_offset = jax.ops.segment_sum(quarter_chord_offset * le_mask_float, strip_ids, num_segments=VD.total_strips)
        strip_DCP = seg_sum(DCP * le_mask_float[None, :])

        # Lan's approximation of LE pressure correction
        singularity_strength_LE = jnp.where(
            subsonic_LE_factor > 0,
            singularity_strength_LE / stripwise_panels[None, :] / subsonic_LE_factor,
            singularity_strength_LE
        )
        
        singularity_strength_LE = singularity_strength_LE + 0.5 * strip_DCP * jnp.sqrt(strip_quarter_chord_offset)[None, :]
        
        suction_flag = 1.0
        suction_coeff_LE = 0.5 * jnp.pi * jnp.abs(suction_flag) * jnp.square(singularity_strength_LE) * subsonic_LE_factor
        
        # Use jax.ops for tangent_incidence_angle
        suction_vector_x = jnp.ones_like(suction_coeff_LE)
        suction_vector_z = -jax.ops.segment_sum(VD.tangent_incidence_angle * le_mask_float, strip_ids, num_segments=VD.total_strips)[None, :]
        
        axial_coeff = axial_coeff - suction_vector_x * suction_coeff_LE
        normal_coeff = normal_coeff + suction_coeff_LE * jnp.sqrt(1.0 + sweep_sq) * suction_vector_z

    # Body Axis Transformation
    incidence_angle = jax.ops.segment_sum(VD.tangent_incidence_angle * le_mask_float, strip_ids, num_segments=VD.total_strips)[None, :]
    cos_incidence, sin_incidence = jnp.cos(incidence_angle), jnp.sin(incidence_angle)
    
    strip_body_force_x =  -normal_coeff * sin_incidence + axial_coeff * cos_incidence
    strip_body_force_y = -(normal_coeff * cos_incidence + axial_coeff * sin_incidence) * sin_dihedral[None, :]
    strip_body_force_z =  (normal_coeff * cos_incidence + axial_coeff * sin_incidence) * cos_dihedral[None, :]
    
    # Get leading-edge collocation points
    colloc_LE = jax.ops.segment_sum(VD.collocation_points * le_mask_float[:, None], strip_ids, num_segments=VD.total_strips)
    
    colloc_LE_x = colloc_LE[:, 0][None, :]
    colloc_LE_y = colloc_LE[:, 1][None, :]
    colloc_LE_z = colloc_LE[:, 2][None, :]
    
    strip_body_moment_x = strip_body_force_z * colloc_LE_y - strip_body_force_y * (colloc_LE_z - z_m) + sideslip_couple
    strip_body_moment_y = pitching_moment * cos_dihedral[None, :] + strip_body_force_x * (colloc_LE_z - z_m) - strip_body_force_z * (colloc_LE_x - x_m)
    strip_body_moment_z = pitching_moment * sin_dihedral[None, :] - strip_body_force_x * colloc_LE_y + strip_body_force_y * (colloc_LE_x - x_m)
    
    # Strip Aerodynamic Integration: collapse, then slice the Y column [:, 1]
    corner_b1_LE = jax.ops.segment_sum(VD.panel_corner_b1 * le_mask_float[:, None], strip_ids, num_segments=VD.total_strips)
    corner_a1_LE = jax.ops.segment_sum(VD.panel_corner_a1 * le_mask_float[:, None], strip_ids, num_segments=VD.total_strips)
    
    panel_span_LE = jnp.abs(corner_b1_LE[:, 1] - corner_a1_LE[:, 1])
    strip_area = panel_span_LE * stripwise_chords
    
    lift = (strip_body_force_z * cos_alpha - (strip_body_force_x * cos_beta + strip_body_force_y * sin_beta) * sin_alpha) * panel_span_LE[None, :]
    force_y = (strip_body_force_y * cos_beta - strip_body_force_x * sin_beta) * strip_area[None, :]

    moment = (strip_body_moment_y * cos_beta - strip_body_moment_x * sin_beta) * panel_span_LE[None, :]
    strip_rolling_moment = (strip_body_moment_x * cos_alpha * cos_beta + strip_body_moment_y * cos_alpha * sin_beta + strip_body_moment_z * sin_alpha) * panel_span_LE[None, :]
    strip_yawing_moment  = (strip_body_moment_z * cos_alpha - (strip_body_moment_x * cos_beta + strip_body_moment_y * sin_beta) * sin_alpha) * panel_span_LE[None, :]    
    
    # Trefftz Plane Induced Drag
    # JAX-compatible trailing edge mask
    te_mask_float = VD.is_trailing_edge.astype(jnp.float32)
    
    # Extract TE coordinates directly
    te_coords = jax.ops.segment_sum(VD.collocation_points * te_mask_float[:, None], strip_ids, num_segments=VD.total_strips)
    x_ctrl = te_coords[:, 0]
    y_ctrl = te_coords[:, 1]
    z_ctrl = te_coords[:, 2]
    
    # Sum gamma over strip
    gamma_spanwise = seg_sum(GAMMA)
    gamma_physical = gamma_spanwise * v_inf
    
    D_induced, _ = _compute_trefftz_drag(
        y_ctrl[None, :], z_ctrl[None, :], x_ctrl[None, :], 
        gamma_physical, alpha[:, 0], rho[:, 0]
    )
    
    CDi = D_induced / (0.5 * rho[:, 0] * jnp.square(v_inf[:, 0]) * S_ref)

    # Global Coefficients
    CL = jnp.sum(lift, axis=1) / S_ref              # Lift coefficient
    CY = jnp.sum(force_y, axis=1) / S_ref           # Side-force coefficient
    CM = jnp.sum(moment, axis=1) / (S_ref * c_ref)  # Pitch moment coefficient
    
    Cl_roll = -jnp.sum(strip_rolling_moment, axis=1) / (S_ref * b_ref) # Rolling moment, negative for sign conventions
    Cn_yaw  = -jnp.sum(strip_yawing_moment, axis=1) / (S_ref * b_ref)  # Yawing moment, negative for sign conventions
    
    # Profile Drag (CX and CZ projection back to inertial axes)
    cx_denom = cos_alpha[:, 0] - sin_alpha[:, 0] * tan_alpha[:, 0]
    safe_cx_denom = jnp.where(jnp.abs(cx_denom) < 1e-8, 1e-8 * jnp.sign(cx_denom + 1e-12), cx_denom)
    CX = (tan_alpha[:, 0] * CL - CDi) / safe_cx_denom
    
    safe_sinalf = jnp.where(jnp.abs(sin_alpha[:, 0]) < 1e-8, 1e-8 * jnp.sign(sin_alpha[:, 0] + 1e-12), sin_alpha[:, 0])
    CZ = (CDi + CX * cos_alpha[:, 0]) / safe_sinalf
    
    return CL, CDi, CX, CY, -CZ, CM, Cl_roll, Cn_yaw

def compute_coefficients(state: "State", system: "System", settings: "Settings"):
    """ Final VLM step to extract global coefficients and append to State. """
    
    analysis = system.analysis_data
    VD = analysis["vortex_distribution"]

    # TODO: Add book-keeping for individual wing coefficients
    CL, CDi, CX, CY, CZ, CM, Cl_roll, Cn_yaw = _compute_aerodynamic_coefficients(
        VD,
        analysis["pressure_coefficients"],
        analysis["vortex_strengths"],
        analysis["VORLAX_EW_matrix"],
        analysis["relative_velocity"],
        state,
        system,
        settings
    )

    # Apply Correction Factors

    vlm_settings: VLMSettings = settings.analysis.aerodynamics #type: ignore

    CL = CL * vlm_settings.correction.fuselage_lift
    
    # Update the Vehicle/Segment State with the aerodynamic forces
    state = eqx.tree_at(lambda s: s.aerodynamics.coefficients.lift.total, state, CL[:, None])
    state = eqx.tree_at(lambda s: s.aerodynamics.coefficients.drag.total, state, CDi[:, None])
    state = eqx.tree_at(lambda s: s.aerodynamics.coefficients.drag.induced.total, state, CDi[:, None])

    state = eqx.tree_at(lambda s: s.aerodynamics.coefficients.moments.pitch, state, CM[:, None])
    state = eqx.tree_at(lambda s: s.aerodynamics.coefficients.moments.roll,  state, Cl_roll[:, None])
    state = eqx.tree_at(lambda s: s.aerodynamics.coefficients.moments.yaw,   state, Cn_yaw[:, None])
    
    return state, system, settings