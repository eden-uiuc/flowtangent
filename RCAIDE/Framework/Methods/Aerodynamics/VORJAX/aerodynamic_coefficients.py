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
    from RCAIDE.Framework.Analyses.Aerodynamics.VORJAX import VLMSettings

from RCAIDE.utils import inputs, outputs
# ----------------------------------------------------------------------------------------------------------------------
#  Lift and Drag Calculation
# ----------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------
# Trefftz Plane Induced Drag
# ---------------------------------------------------------
@jax.jit
def _compute_trefftz_drag(tp_y_ctrl, tp_z_ctrl, tp_y_L, tp_y_R, tp_z_L, tp_z_R, gamma_segments, rho):
    
    # 1. Distance Matrices
    dy_L = tp_y_ctrl[:, :, None] - tp_y_L[:, None, :]
    dz_L = tp_z_ctrl[:, :, None] - tp_z_L[:, None, :]
    r2_L = jnp.maximum(dy_L**2 + dz_L**2, 1e-12)
    
    dy_R = tp_y_ctrl[:, :, None] - tp_y_R[:, None, :]
    dz_R = tp_z_ctrl[:, :, None] - tp_z_R[:, None, :]
    r2_R = jnp.maximum(dy_R**2 + dz_R**2, 1e-12)

    # 2. Induced Velocity
    v_ind_y = jnp.sum((gamma_segments[:, None, :] / (2.0 * jnp.pi)) * ((dz_L / r2_L) - (dz_R / r2_R)), axis=-1)
    v_ind_z = jnp.sum((gamma_segments[:, None, :] / (2.0 * jnp.pi)) * (-(dy_L / r2_L) + (dy_R / r2_R)), axis=-1)

    # 3. Strict 2D Unstructured Normal Vectors
    dy_panel = tp_y_R - tp_y_L
    dz_panel = tp_z_R - tp_z_L
    panel_width = jnp.maximum(jnp.sqrt(dy_panel**2 + dz_panel**2), 1e-16)

    n_hat_y = -dz_panel / panel_width
    n_hat_z =  dy_panel / panel_width
    
    v_normal = v_ind_y * n_hat_y + v_ind_z * n_hat_z

    # 4. Drag Integration
    D_induced = -0.5 * rho * jnp.sum(gamma_segments * v_normal * panel_width, axis=1)
    
    return D_induced, v_normal

# ---------------------------------------------------------
# Full Coefficient Calculation
# ---------------------------------------------------------

@jax.jit
def _compute_aerodynamic_coefficients(VD, dCp, Gamma, EW, v_rel, state, system, settings):
    """
    Computes CL, CD, C_m, CY_body, Cl, Cn and induced drag using the unstructured VD mesh.
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
    strip_ids = VD.strip_ids
    
    panel_ones = jnp.ones_like(strip_ids, dtype=jnp.float32)
    stripwise_panels = jax.ops.segment_sum(panel_ones, strip_ids, num_segments=VD.total_strips)

    stripwise_chords = jax.ops.segment_sum(VD.chord_lengths, strip_ids, num_segments=VD.total_strips)
    panel_dx_nondim = VD.chord_lengths/stripwise_chords[VD.strip_ids]
    
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
    tan_sweep_LE = dx_LE / dihedral_length_LE
    cos_dihedral = jnp.abs(dy_LE) / dihedral_length_LE  
    sin_dihedral = jnp.sign(dy_LE) * dz_LE / dihedral_length_LE  
    
    # Panel Forces
    quarter_chord_offset    = 0.25 * panel_dx_nondim 
    colloc_offset           = 0.75 * panel_dx_nondim 
    panel_force_mag         = panel_dx_nondim[None, :] * dCp

    panel_inc           = VD.incidence_angle
    panel_axial_coeff   = panel_force_mag * jnp.sin(panel_inc)[None, :]
    panel_normal_coeff  = panel_force_mag * jnp.cos(panel_inc)[None, :]

    panel_indices       = jnp.arange(VD.total_panels)
    strip_start_indices = jax.ops.segment_min(panel_indices, strip_ids, num_segments=VD.total_strips)[strip_ids]
    chordwise_indices   = panel_indices - strip_start_indices + 1.0
    
    vortex_x_nondim = (chordwise_indices - 0.75) * panel_dx_nondim
    panel_pitching_moment = (colloc_offset[None, :] - vortex_x_nondim[None, :]) * panel_normal_coeff

    # ------------------------------------------------------------------
    # Rear Quarter Calculation (Using VD.panel_vertices)
    # ------------------------------------------------------------------
    collocation_x = VD.collocation_points[:, 0]
    trailing_edge_x_avg = (VD.panel_vertices[:, 1, 0] + VD.panel_vertices[:, 2, 0]) / 2.0
    rear_quarter_x = trailing_edge_x_avg - collocation_x
    panel_sideslip_couple = panel_normal_coeff * rear_quarter_x[None, :]
    
    # Integrate Panels into Strips, V-Mapped over time dimension
    strip_sum = jax.vmap(lambda arr: jax.ops.segment_sum(arr, strip_ids, num_segments=VD.total_strips))

    strip_body_x_coeff  = strip_sum(panel_axial_coeff) * stripwise_chords[None, :]
    strip_body_z_coeff  = strip_sum(panel_normal_coeff)    *  stripwise_chords[None, :]
    pitching_moment     = strip_sum(panel_pitching_moment) * (stripwise_chords[None, :] ** 2)
    
    sideslip_couple = strip_sum(panel_sideslip_couple) * stripwise_chords[None, :]
    sideslip_couple = sideslip_couple * (-1.0) * crosswind_factor * cos_dihedral[None, :] * 0.5 
    
    # ------------------------------------------------------------------
    # Lan's Correction for Leading Edge Suction
    # ------------------------------------------------------------------
    vlm_settings = settings.analysis.aerodynamics
    if vlm_settings.le_suction_correction:
        # EW_masked = EW * le_mask_float[None, :, None]

        # # Add the trailing None to broadcast across the 3 velocity components
        # v_total_masked = v_total_norm * le_mask_float[None, :, None]

        # EW_LE = seg_sum(EW_masked)
        # v_total_LE = seg_sum(v_total_masked)

        # normals_masked = VD.normal_vectors * le_mask_float[:, None]
        # normals_LE = jax.ops.segment_sum(normals_masked, strip_ids, num_segments=VD.total_strips)

        # # --- CAMBER CORRECTION 1: Fetch LE camber slopes ---
        # camber_slopes_LE = jax.ops.segment_sum(VD.camber_slopes * le_mask_float, strip_ids,
        #                                        num_segments=VD.total_strips)

        # induced_velocity_LE = jnp.sum(EW_LE * GAMMA[:, None, :], axis=-1)

        # # --- CAMBER CORRECTION 2: Inject camber slope into the effective incidence ---
        # base_incidence_LE = jnp.sum(v_total_LE * normals_LE[None, :, :], axis=-1)
        # eff_incidence_LE = base_incidence_LE - (v_total_LE[..., 0] * camber_slopes_LE[None, :])

        # A0 = induced_velocity_LE - eff_incidence_LE

        # B_sq = jnp.square(mach) - 1.0
        # sweep_sq = jnp.square(tan_sweep_LE)[None, :]
        # L_eff = jnp.where(
        #     B_sq < sweep_sq,
        #     jnp.sqrt(jnp.maximum(sweep_sq - B_sq, 1e-16)),
        #     0.0
        # )

        # strip_quarter_chord_offset = jax.ops.segment_sum(
        #     quarter_chord_offset * le_mask_float,
        #     strip_ids,
        #     num_segments=VD.total_strips
        # )
        # LE_dCp = seg_sum(dCp * le_mask_float[None, :])

        # A0 = jnp.where(
        #     L_eff > 0,
        #     A0 / stripwise_panels[None, :] / jnp.maximum(L_eff, 1e-16),
        #     A0
        # )

        # A0 = A0 + 0.5 * LE_dCp * jnp.sqrt(strip_quarter_chord_offset)[None, :]
        B_sq = jnp.square(mach) - 1.0
        t_sq = jnp.square(tan_sweep_LE)[None, :]

        # Double guard statement to avoid singularity at B_sq > t_sq
        L_eff = jnp.where(B_sq < t_sq, jnp.sqrt(jnp.maximum(t_sq - B_sq, 1e-16)), 0.0)

        if vlm_settings.vortices.chordwise_cosine_spacing:
            w0 = jnp.sum(EW * Gamma[:, None, :], axis=-1)

            v_unit = v_rel / state.freestream.speed[:, None, :]
            v_dot_n         = jnp.sum(v_unit * VD.normal_vectors[None, :, :], axis=-1)
            local_incidence = v_dot_n - (v_unit[..., 0] * VD.camber_slopes[None, :])

            LE_normalwash = strip_sum(w0 * le_mask_float[None, :])
            LE_incidence  = strip_sum(local_incidence * le_mask_float[None, :])

            # Leading edge singularity strength
            A0 = (LE_normalwash + LE_incidence) / stripwise_panels[None, :]
            A0 = jnp.where(L_eff > 0, A0 / jnp.maximum(L_eff, 1e-16), A0)

        else:
            # Correct singularity strength using leading edge pressure coefficient
            le_qc   = jax.ops.segment_sum(quarter_chord_offset * le_mask_float, strip_ids, num_segments=VD.total_strips)[None, :]
            le_dCp  = strip_sum(dCp * le_mask_float[None, :])
            A0  = 0.5 * le_dCp * jnp.sqrt(le_qc)
        
        # Suction coefficient
        Cs = 0.5 * jnp.pi * jnp.square(A0) * L_eff

        # --- CAMBER CORRECTION 3: True aerodynamic orientation for the suction vector ---
        incidence_LE = jax.ops.segment_sum(VD.incidence_angle * le_mask_float, strip_ids, num_segments=VD.total_strips)[None, :]

        suction_vector_x = jnp.ones_like(Cs)
        suction_vector_z = -incidence_LE

        # --- Update the strip coefficients from the previous block ---
        strip_body_x_coeff = strip_body_x_coeff - suction_vector_x * Cs
        strip_body_z_coeff = strip_body_z_coeff + Cs * jnp.sqrt(1.0 + t_sq) * suction_vector_z

    # ------------------------------------------------------------------
    # Body Axis Transformation & Strips Integration
    # ------------------------------------------------------------------
    strip_body_force_x =  strip_body_x_coeff
    strip_body_force_y = -strip_body_z_coeff * sin_dihedral[None, :]
    strip_body_force_z =  strip_body_z_coeff * cos_dihedral[None, :]
    
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
    
    strip_lift              = (strip_body_force_z * cos_alpha - (strip_body_force_x * cos_beta + strip_body_force_y * sin_beta) * sin_alpha) * panel_span_LE[None, :]
    strip_pitching_moment   = (strip_body_moment_y * cos_beta - strip_body_moment_x * sin_beta) * panel_span_LE[None, :]

    force_x =  strip_body_force_x * strip_area[None, :]
    force_y = (strip_body_force_y * cos_beta - strip_body_force_x * sin_beta) * strip_area[None, :]
    force_z =  strip_body_force_z * strip_area[None, :]
    
    strip_rolling_moment = (strip_body_moment_x * cos_alpha * cos_beta + strip_body_moment_y * cos_alpha * sin_beta + strip_body_moment_z * sin_alpha) * panel_span_LE[None, :]
    strip_yawing_moment  = (strip_body_moment_z * cos_alpha - (strip_body_moment_x * cos_beta + strip_body_moment_y * sin_beta) * sin_alpha) * panel_span_LE[None, :]    
    
    # ------------------------------------------------------------------
    # Trefftz Plane Execution
    # ------------------------------------------------------------------
    TE_corner_L = jax.ops.segment_sum(VD.panel_vertices[:, 1, :] * te_mask_float[:, None], strip_ids, num_segments=VD.total_strips)
    TE_corner_R = jax.ops.segment_sum(VD.panel_vertices[:, 2, :] * te_mask_float[:, None], strip_ids, num_segments=VD.total_strips)
    TE_mid      = (TE_corner_L + TE_corner_R) / 2.0

    # 2. Project exactly from the TE to infinity
    tp_z_ctrl = TE_mid[:, 2] * cos_alpha - TE_mid[:, 0] * sin_alpha
    tp_z_L    = TE_corner_L[:, 2] * cos_alpha - TE_corner_L[:, 0] * sin_alpha
    tp_z_R    = TE_corner_R[:, 2] * cos_alpha - TE_corner_R[:, 0] * sin_alpha

    D_trefftz, _ = _compute_trefftz_drag(
        TE_mid[:, 1][None, :],
        tp_z_ctrl,
        TE_corner_L[:, 1][None, :],
        TE_corner_R[:, 1][None, :],
        tp_z_L,
        tp_z_R,
        strip_sum(Gamma) * v_inf,
        rho[:, 0]
    )

    # Wind-Frame Coefficients
    # Body Frame is Back-Right-Up, Wind-Frame is Front-Right-Down, so CX and CZ are negative
    CX_wind = -jnp.sum(force_x, axis=1) / S_ref
    CY_wind =  jnp.sum(force_y, axis=1) / S_ref
    CZ_wind = -jnp.sum(force_z, axis=1) / S_ref

    CL_near  = jnp.sum(strip_lift, axis=1) / S_ref
    CD_near = -CX_wind * cos_alpha[:, 0] - CZ_wind * sin_alpha[:, 0]

    C_l = -jnp.sum(strip_rolling_moment, axis=1) / (S_ref * b_ref)
    C_m =  jnp.sum(strip_pitching_moment, axis=1) / (S_ref * c_ref)
    C_n = -jnp.sum(strip_yawing_moment, axis=1) / (S_ref * b_ref)
    
    # CD, CX, CL from far-field integration
    CDi_far     = D_trefftz / (0.5 * rho[:, 0] * jnp.square(v_inf[:, 0]) * S_ref)   # Positive in Body Frame
    CX_body_far = -(CDi_far/cos_alpha[:, 0] - CZ_wind * jnp.tan(alpha[:, 0]))       # CX is in Wind Frame, opposite of Body Frame Orientation
    CL_far      = -CZ_wind * cos_alpha[:, 0] + CX_body_far * sin_alpha[:, 0]
    
    return CL_near, CD_near, CL_far, CDi_far, CX_wind, CX_body_far, CY_wind, CZ_wind, C_l, C_m, C_n

@inputs(
    "system.analysis_data['vortex_distribution']",
    "system.analysis_data[dCp]",
    "system.analysis_data['vortex_strengths']",
    "system.analysis_data['effective_wash']",
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

    CL, CD, CL_far, CDi, CX, CX_far, CY, CZ, C_l, C_m, C_n = _compute_aerodynamic_coefficients(
        analysis["vortex_distribution"],
        analysis["dCp"],
        analysis["vortex_strengths"],
        analysis["effective_wash"],
        analysis["relative_velocity"],
        state,
        system,
        settings
    )

    # Apply Correction Factors
    vlm_settings: VLMSettings = settings.analysis.aerodynamics  # type: ignore

    if vlm_settings.far_field:
        CL  = CL_far
        CD  = CDi
        CX  = CX_far
    
    if vlm_settings.model_fuselage:
        CL = CL * vlm_settings.correction.fuselage_lift

    # Update the Vehicle/Segment State with the aerodynamic coefficients
    state = eqx.tree_at(lambda s: s.aerodynamics.coefficients.lift.total, state, CL[:, None])
    
    state = eqx.tree_at(lambda s: s.aerodynamics.coefficients.drag.total, state, CD[:, None])
    state = eqx.tree_at(lambda s: s.aerodynamics.coefficients.drag.induced.total, state, CDi[:, None])
    state = eqx.tree_at(lambda s: s.aerodynamics.coefficients.drag.induced.inviscid.total, state, CDi[:, None])

    # Wind-Frame Coefficients
    state = eqx.tree_at(lambda s: s.aerodynamics.coefficients.X, state, CX[:, None])
    state = eqx.tree_at(lambda s: s.aerodynamics.coefficients.Y, state, CY[:, None])
    state = eqx.tree_at(lambda s: s.aerodynamics.coefficients.Z, state, CZ[:, None])

    # Moment Coefficients
    state = eqx.tree_at(lambda s: s.aerodynamics.coefficients.moments.pitch, state, C_m[:, None])
    state = eqx.tree_at(lambda s: s.aerodynamics.coefficients.moments.roll,  state, C_l[:, None])
    state = eqx.tree_at(lambda s: s.aerodynamics.coefficients.moments.yaw,   state, C_n[:, None])

    return state, system, settings