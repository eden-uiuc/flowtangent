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

from RCAIDE.utils import inputs, outputs
# ----------------------------------------------------------------------------------------------------------------------
#  Lift and Drag Calculation
# ----------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------
# Trefftz Plane Induced Drag
# ---------------------------------------------------------
@jax.jit
def _compute_trefftz_drag(tp_y_ctrl, tp_z_ctrl, fil_y_L, fil_y_R, fil_z_L, fil_z_R, fil_x_LE, gamma_segments, alpha, rho):
    # 1. Wind-Axis Rotation (Crucial: Rotate BOTH points and filaments)
    cos_a = jnp.cos(alpha)
    sin_a = jnp.sin(alpha)
    
    # Trefftz plane is at X=infinity, but we project the TE geometry onto it.
    # To be mathematically consistent with AVL, we rotate the Z-positions 
    # based on their X-distance from the LE.
    tp_z_L = fil_z_L * cos_a - fil_x_LE * sin_a
    tp_z_R = fil_z_R * cos_a - fil_x_LE * sin_a
    
    # 2. Distance Matrices
    dy_L = tp_y_ctrl[:, :, None] - fil_y_L[:, None, :]
    dz_L = tp_z_ctrl[:, :, None] - tp_z_L[:, None, :]
    r2_L = jnp.maximum(dy_L**2 + dz_L**2, 1e-12)
    
    dy_R = tp_y_ctrl[:, :, None] - fil_y_R[:, None, :]
    dz_R = tp_z_ctrl[:, :, None] - tp_z_R[:, None, :]
    r2_R = jnp.maximum(dy_R**2 + dz_R**2, 1e-12)

    # 3. Induced Velocity (Point Vortices)
    v_ind_y = jnp.sum((gamma_segments[:, None, :] / (2.0 * jnp.pi)) * ( (dz_L / r2_L) - (dz_R / r2_R) ), axis=-1)
    v_ind_z = jnp.sum((gamma_segments[:, None, :] / (2.0 * jnp.pi)) * ( (-dy_L / r2_L) - (-dy_R / r2_R) ), axis=-1)

    # 4. Normal Vectors in Wind Axis
    dy_c = jnp.gradient(tp_y_ctrl, axis=-1)
    dz_c = jnp.gradient(tp_z_ctrl, axis=-1)
    ctrl_mag = jnp.sqrt(dy_c**2 + dz_c**2 + 1e-16)
    
    # These normals MUST be consistent with the VD.normal_vectors convention
    n_hat_y = dz_c / ctrl_mag
    n_hat_z = dy_c / ctrl_mag
    
    v_normal = v_ind_y * n_hat_y + v_ind_z * n_hat_z

    # 5. Drag Integration
    panel_dy = fil_y_R - fil_y_L
    panel_dz = tp_z_R - tp_z_L # Use rotated Z for panel width
    panel_width = jnp.sqrt(panel_dy**2 + panel_dz**2)
    
    D_induced = 0.5 * rho * jnp.sum(gamma_segments * v_normal * panel_width, axis=1)
    
    return D_induced, v_normal

# ---------------------------------------------------------
# Full Coefficient Calculation
# ---------------------------------------------------------

@jax.jit
def _compute_aerodynamic_coefficients(VD, DCP, GAMMA, EW, v_total, state, system, settings):
    """
    Computes CL, CD, CM, CY, Cl, Cn and induced drag using the unstructured VD mesh.
    """
    alpha = state.aerodynamics.angles.alpha
    beta  = state.aerodynamics.angles.beta
    v_inf = state.freestream.speed
    mach  = state.freestream.mach_number
    rho   = state.freestream.density
    
    S_ref = system.areas.reference
    c_ref = system.reference_geometry.mean_aerodynamic_chord
    b_ref = system.reference_geometry.projected_span
    cg    = system.reference_geometry.center_of_gravity
    x_m, y_m, z_m = cg[:, 0][:, None], cg[:, 1][:, None], -cg[:, 2][:, None]
    
    sin_alpha, cos_alpha = jnp.sin(alpha), jnp.cos(alpha)
    sin_beta, cos_beta = jnp.sin(beta), jnp.cos(beta)
    crosswind_factor = cos_alpha * sin_beta * 2.0
    
    # ------------------------------------------------------------------
    # Unstructured Mesh Topology Resolution
    # ------------------------------------------------------------------
    le_mask_float = VD.is_leading_edge.astype(jnp.float32)
    te_mask_float = VD.is_trailing_edge.astype(jnp.float32)
    
    strip_ids = jnp.cumsum(VD.is_leading_edge) - 1 
    
    panel_ones = jnp.ones_like(strip_ids, dtype=jnp.float32)
    stripwise_panels = jax.ops.segment_sum(panel_ones, strip_ids, num_segments=VD.total_strips)
    
    panels_per_strip = stripwise_panels[strip_ids]
    panel_dx_nondim = 1.0 / panels_per_strip 

    stripwise_chords = jax.ops.segment_max(VD.chord_lengths, strip_ids, num_segments=VD.total_strips)
    
    # ------------------------------------------------------------------
    # Local Panel Sweep and Dihedral (Using VD.panel_vertices)
    # 0: Front-Left, 1: Back-Left, 2: Back-Right, 3: Front-Right
    # ------------------------------------------------------------------
    # Leading edge vector of each panel: Front-Right minus Front-Left
    dx_all = VD.panel_vertices[:, 3, 0] - VD.panel_vertices[:, 0, 0]
    dy_all = VD.panel_vertices[:, 3, 1] - VD.panel_vertices[:, 0, 1]
    dz_all = VD.panel_vertices[:, 3, 2] - VD.panel_vertices[:, 0, 2]

    dy_LE = jax.ops.segment_sum(dy_all * le_mask_float, strip_ids, num_segments=VD.total_strips)
    dz_LE = jax.ops.segment_sum(dz_all * le_mask_float, strip_ids, num_segments=VD.total_strips)
    dx_LE = jax.ops.segment_sum(dx_all * le_mask_float, strip_ids, num_segments=VD.total_strips)
    
    dihedral_length_LE = jnp.maximum(jnp.sqrt(dy_LE**2 + dz_LE**2), 1e-12)
    true_sweep_LE = dx_LE / dihedral_length_LE 
    cos_dihedral = jnp.abs(dy_LE) / dihedral_length_LE  
    sin_dihedral = jnp.sign(dy_LE) * dz_LE / dihedral_length_LE  
    
    # Panel Forces
    quarter_chord_offset = 0.25 * panel_dx_nondim 
    colloc_offset = 0.75 * panel_dx_nondim 
    panel_normal_coeff = panel_dx_nondim[None, :] * DCP  
    
    nx, _, nz = VD.normal_vectors[:, 0], VD.normal_vectors[:, 1], VD.normal_vectors[:, 2] 
    panel_chordwise_slope = -nx / jnp.maximum(jnp.abs(nz), 1e-12)
    panel_axial_coeff = -panel_normal_coeff * panel_chordwise_slope[None, :] / (1.0 + panel_chordwise_slope[None, :]**2)
    
    panel_indices = jnp.arange(VD.total_panels)
    strip_start_indices = jax.ops.segment_min(panel_indices, strip_ids, num_segments=VD.total_strips)[strip_ids]
    chordwise_indices = panel_indices - strip_start_indices + 1.0
    
    vortex_x_nondim = (chordwise_indices - 0.75) * panel_dx_nondim
    panel_pitching_moment = (colloc_offset[None, :] - vortex_x_nondim[None, :]) * panel_normal_coeff
    
    # ------------------------------------------------------------------
    # Rear Quarter Calculation (Using VD.panel_vertices)
    # ------------------------------------------------------------------
    collocation_x = VD.collocation_points[:, 0]
    # Average of Back-Left (1) and Back-Right (2)
    trailing_edge_x_avg = (VD.panel_vertices[:, 1, 0] + VD.panel_vertices[:, 2, 0]) / 2.0
    rear_quarter_x = trailing_edge_x_avg - collocation_x
    panel_sideslip_couple = panel_normal_coeff * rear_quarter_x[None, :]
    
    # Integrate Panels into Strips
    seg_sum = jax.vmap(lambda arr: jax.ops.segment_sum(arr, strip_ids, num_segments=VD.total_strips))
    
    normal_coeff    = seg_sum(panel_normal_coeff)    * stripwise_chords[None, :]
    axial_coeff     = seg_sum(panel_axial_coeff)     * stripwise_chords[None, :] 
    pitching_moment = seg_sum(panel_pitching_moment) * (stripwise_chords[None, :] ** 2)
    
    sideslip_couple = seg_sum(panel_sideslip_couple) * stripwise_chords[None, :]
    sideslip_couple = sideslip_couple * (-1.0) * crosswind_factor * cos_dihedral[None, :] * 0.5 
    
    # ------------------------------------------------------------------
    # VORLAX LE Suction
    # ------------------------------------------------------------------
    if settings.analysis.aerodynamics.VORLAX_empirical_corrections:
        EW_masked = EW.squeeze(1) * le_mask_float[None, :, None]
        
        # Add the trailing None to broadcast across the 3 velocity components!
        v_total_masked = v_total * le_mask_float[None, :, None]

        EW_LE = seg_sum(EW_masked)
        v_total_LE = seg_sum(v_total_masked)

        # normal_vectors is (N, 3), so we need [:, None] here
        normals_masked = VD.normal_vectors * le_mask_float[:, None]
        normals_LE = jax.ops.segment_sum(normals_masked, strip_ids, num_segments=VD.total_strips)
        
        induced_velocity_LE = jnp.sum(EW_LE * GAMMA, axis=-1)
        
        # eff_incidence_LE dot product 
        eff_incidence_LE = jnp.sum(v_total_LE * normals_LE[None, :, :], axis=-1)
        singularity_strength_LE = induced_velocity_LE - eff_incidence_LE
        
        prandtl_glauert_beta_sq = jnp.square(mach) - 1.0
        sweep_sq = jnp.square(true_sweep_LE)[None, :]
        subsonic_LE_factor = jnp.where(
            prandtl_glauert_beta_sq < sweep_sq,
            jnp.sqrt(jnp.maximum(sweep_sq - prandtl_glauert_beta_sq, 0.0)),
            0.0
        )
        
        strip_quarter_chord_offset = jax.ops.segment_sum(quarter_chord_offset * le_mask_float, strip_ids, num_segments=VD.total_strips)
        strip_DCP = seg_sum(DCP * le_mask_float[None, :])

        singularity_strength_LE = jnp.where(
            subsonic_LE_factor > 0,
            singularity_strength_LE / stripwise_panels[None, :] / subsonic_LE_factor,
            singularity_strength_LE
        )
        
        singularity_strength_LE = singularity_strength_LE + 0.5 * strip_DCP * jnp.sqrt(strip_quarter_chord_offset)[None, :]
        suction_coeff_LE = 0.5 * jnp.pi * jnp.square(singularity_strength_LE) * subsonic_LE_factor
        
        suction_vector_x = jnp.ones_like(suction_coeff_LE)
        suction_vector_z = -jax.ops.segment_sum(VD.tangent_incidence_angle * le_mask_float, strip_ids, num_segments=VD.total_strips)[None, :]
        
        axial_coeff = axial_coeff - suction_vector_x * suction_coeff_LE
        normal_coeff = normal_coeff + suction_coeff_LE * jnp.sqrt(1.0 + sweep_sq) * suction_vector_z

    # ------------------------------------------------------------------
    # Body Axis Transformation & Strips Integration
    # ------------------------------------------------------------------
    incidence_angle = jax.ops.segment_sum(VD.tangent_incidence_angle * le_mask_float, strip_ids, num_segments=VD.total_strips)[None, :]
    cos_incidence, sin_incidence = jnp.cos(incidence_angle), jnp.sin(incidence_angle)
    
    strip_body_force_x =  -normal_coeff * sin_incidence + axial_coeff * cos_incidence
    strip_body_force_y = -(normal_coeff * cos_incidence + axial_coeff * sin_incidence) * sin_dihedral[None, :]
    strip_body_force_z =  (normal_coeff * cos_incidence + axial_coeff * sin_incidence) * cos_dihedral[None, :]
    
    colloc_LE = jax.ops.segment_sum(VD.collocation_points * le_mask_float[:, None], strip_ids, num_segments=VD.total_strips)
    colloc_LE_x, colloc_LE_y, colloc_LE_z = colloc_LE[:, 0][None, :], colloc_LE[:, 1][None, :], colloc_LE[:, 2][None, :]
    
    strip_body_moment_x = strip_body_force_z * colloc_LE_y - strip_body_force_y * (colloc_LE_z - z_m) + sideslip_couple
    strip_body_moment_y = pitching_moment * cos_dihedral[None, :] + strip_body_force_x * (colloc_LE_z - z_m) - strip_body_force_z * (colloc_LE_x - x_m)
    strip_body_moment_z = pitching_moment * sin_dihedral[None, :] - strip_body_force_x * colloc_LE_y + strip_body_force_y * (colloc_LE_x - x_m)
    
    # Strip Aerodynamic Integration: Front-Right (3) and Front-Left (0)
    corner_b1_LE = jax.ops.segment_sum(VD.panel_vertices[:, 3, :] * le_mask_float[:, None], strip_ids, num_segments=VD.total_strips)
    corner_a1_LE = jax.ops.segment_sum(VD.panel_vertices[:, 0, :] * le_mask_float[:, None], strip_ids, num_segments=VD.total_strips)
    
    panel_span_LE = jnp.abs(corner_b1_LE[:, 1] - corner_a1_LE[:, 1])
    strip_area = panel_span_LE * stripwise_chords
    
    lift = (strip_body_force_z * cos_alpha - (strip_body_force_x * cos_beta + strip_body_force_y * sin_beta) * sin_alpha) * panel_span_LE[None, :]
    force_y = (strip_body_force_y * cos_beta - strip_body_force_x * sin_beta) * strip_area[None, :]
    moment = (strip_body_moment_y * cos_beta - strip_body_moment_x * sin_beta) * panel_span_LE[None, :]
    
    strip_rolling_moment = (strip_body_moment_x * cos_alpha * cos_beta + strip_body_moment_y * cos_alpha * sin_beta + strip_body_moment_z * sin_alpha) * panel_span_LE[None, :]
    strip_yawing_moment  = (strip_body_moment_z * cos_alpha - (strip_body_moment_x * cos_beta + strip_body_moment_y * sin_beta) * sin_alpha) * panel_span_LE[None, :]    
    
    # ------------------------------------------------------------------
    # Trefftz Plane Execution
    # ------------------------------------------------------------------
    colloc_TE = jax.ops.segment_sum(VD.collocation_points * te_mask_float[:, None], strip_ids, num_segments=VD.total_strips)
    fil_TE_L = jax.ops.segment_sum(VD.bound_vortex_left * te_mask_float[:, None], strip_ids, num_segments=VD.total_strips)
    fil_TE_R = jax.ops.segment_sum(VD.bound_vortex_right * te_mask_float[:, None], strip_ids, num_segments=VD.total_strips)
    
    panel_le_x = (VD.panel_vertices[:, 0, 0] + VD.panel_vertices[:, 3, 0]) / 2.0
    fil_x_LE = jax.ops.segment_sum(panel_le_x * le_mask_float, strip_ids, num_segments=VD.total_strips)

    # Wind-Axis Z conversion for the control point
    tp_z_ctrl = colloc_TE[:, 2] * cos_alpha - colloc_TE[:, 0] * sin_alpha

    D_induced, _ = _compute_trefftz_drag(
        colloc_TE[:, 1][None, :], tp_z_ctrl, 
        fil_TE_L[:, 1][None, :], fil_TE_R[:, 1][None, :], 
        fil_TE_L[:, 2][None, :], fil_TE_R[:, 2][None, :], 
        fil_x_LE[None, :],
        seg_sum(GAMMA) * v_inf, alpha, rho
    )
    
    CDi = D_induced / (0.5 * rho[:, 0] * jnp.square(v_inf[:, 0]) * S_ref)

    # Global Coefficients
    CL = jnp.sum(lift, axis=1) / S_ref              
    CY = jnp.sum(force_y, axis=1) / S_ref           
    CM = jnp.sum(moment, axis=1) / (S_ref * c_ref)  
    
    Cl_roll = -jnp.sum(strip_rolling_moment, axis=1) / (S_ref * b_ref)  
    Cn_yaw  = -jnp.sum(strip_yawing_moment, axis=1) / (S_ref * b_ref)   
    
    # Profile Drag projection
    cx_denom = cos_alpha[:, 0] - sin_alpha[:, 0] * jnp.tan(alpha[:, 0])
    safe_cx_denom = jnp.where(jnp.abs(cx_denom) < 1e-8, 1e-8 * jnp.sign(cx_denom + 1e-12), cx_denom)
    CX = (jnp.tan(alpha[:, 0]) * CL - CDi) / safe_cx_denom
    
    safe_sinalf = jnp.where(jnp.abs(sin_alpha[:, 0]) < 1e-8, 1e-8 * jnp.sign(sin_alpha[:, 0] + 1e-12), sin_alpha[:, 0])
    CZ = (CDi + CX * cos_alpha[:, 0]) / safe_sinalf
    
    return CL, CDi, CX, CY, -CZ, CM, Cl_roll, Cn_yaw

@inputs(
    "system.analysis_data['vortex_distribution']",
    "system.analysis_data['pressure_coefficients']",
    "system.analysis_data['vortex_strengths']",
    "system.analysis_data['VORLAX_EW_matrix']",
    "system.analysis_data['relative_velocity']",
    "state.aerodynamics.angles.alpha",
    "state.aerodynamics.angles.beta",
    "state.freestream.speed",
    "state.freestream.mach_number",
    "state.freestream.density",
    "system.areas.reference",
    "system.reference_geometry.mean_aerodynamic_chord",
    "system.reference_geometry.projected_span",
    "system.reference_geometry.center_of_gravity",
)
@outputs(
    "state.aerodynamics.coefficients.lift.total",
    "state.aerodynamics.coefficients.drag.total",
    "state.aerodynamics.coefficients.drag.induced.total",
    "state.aerodynamics.coefficients.drag.induced.inviscid.total",
    "state.aerodynamics.coefficients.X",
    "state.aerodynamics.coefficients.Y",
    "state.aerodynamics.coefficients.Z",
    "state.aerodynamics.coefficients.moments.pitch",
    "state.aerodynamics.coefficients.moments.roll",
    "state.aerodynamics.coefficients.moments.yaw"
)
def compute_coefficients(state: "State", system: "System", settings: "Settings"):
    """ Final VLM step to extract global coefficients and append to State. """
    
    analysis = system.analysis_data

    CL, CDi, CX, CY, CZ, CM, Cl_roll, Cn_yaw = _compute_aerodynamic_coefficients(
        analysis["vortex_distribution"],
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

    if settings.analysis.aerodynamics.model_fuselage:
        CL = CL * vlm_settings.correction.fuselage_lift
    
    # Update the Vehicle/Segment State with the aerodynamic coefficients
    state = eqx.tree_at(lambda s: s.aerodynamics.coefficients.lift.total, state, CL[:, None])
    state = eqx.tree_at(lambda s: s.aerodynamics.coefficients.drag.total, state, CDi[:, None])
    state = eqx.tree_at(lambda s: s.aerodynamics.coefficients.drag.induced.total, state, CDi[:, None])
    state = eqx.tree_at(lambda s: s.aerodynamics.coefficients.drag.induced.inviscid.total, state, CDi[:, None])

    state = eqx.tree_at(lambda s: s.aerodynamics.coefficients.X, state, CX[:, None])
    state = eqx.tree_at(lambda s: s.aerodynamics.coefficients.Y, state, CY[:, None])
    state = eqx.tree_at(lambda s: s.aerodynamics.coefficients.Z, state, CZ[:, None])

    state = eqx.tree_at(lambda s: s.aerodynamics.coefficients.moments.pitch, state, CM[:, None])
    state = eqx.tree_at(lambda s: s.aerodynamics.coefficients.moments.roll,  state, Cl_roll[:, None])
    state = eqx.tree_at(lambda s: s.aerodynamics.coefficients.moments.yaw,   state, Cn_yaw[:, None])
    
    return state, system, settings