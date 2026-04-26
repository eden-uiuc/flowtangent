# RCAIDE/Framework/Methods/Aerodynamics/VLM/induced_velocity.py
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

from RCAIDE.utils import inputs, outputs
# ----------------------------------------------------------------------------------------------------------------------
#  Helper Functions
# ----------------------------------------------------------------------------------------------------------------------


@jax.jit
def subsonic_induction(Z, XSQ1, RO1, XSQ2, RO2, XTY, T, B2, ZSQ, TOLSQ, X1, Y1, X2, Y2, RTV1, RTV2):
    """
    Pure JAX translation of the VORLAX subsonic Biot-Savart induction.

    This kernel computes the induced velocity (U, V, W) at a collocation point
    due to a single swept horseshoe vortex, applying local compressibility scaling.

    Variable Glossary (Miranda-Elliott-Baker Local Swept Coordinate System):
    -------------------------------------------------------------------------
    Z          : Vertical distance from the collocation point to the vortex plane.
    X1, X2     : Streamwise distances from the collocation pt to vortex endpoints 1 and 2.
    Y1, Y2     : Spanwise distances from the collocation pt to vortex endpoints 1 and 2.
    XSQ1, XSQ2 : Squared streamwise distances (X1^2, X2^2).
    RTV1, RTV2 : Squared transverse distances (Y^2 + Z^2) to endpoints.
    B2         : Compressibility factor (M^2 - 1). Negative in subsonic flow.
    RO1, RO2   : Compressibility-scaled transverse distances (B2 * RTV).
    RAD1, RAD2 : "Effective" compressible distances to endpoints sqrt(X^2 - B^2 * RTV^2).
    T          : Tangent of the bound vortex sweep angle.
    XTY        : Cross-term projection mapping the distance along the swept vortex line.
    TOLSQ      : Squared singularity tolerance (prevents div-by-zero near the filament).
    FB1, FB2   : Bound vortex influence terms.
    FT1, FT2   : Trailing vortex influence terms.
    QB         : Combined bound vortex induction parameter.
    """
    CPI = 4.0 * jnp.pi

    # 1. Effective Compressible Distances
    # Using 1e-16 prevents exact 0.0 which would cause NaN gradients in downstream divisions
    RAD1 = jnp.sqrt(jnp.maximum(XSQ1 - RO1, 1e-16))
    RAD2 = jnp.sqrt(jnp.maximum(XSQ2 - RO2, 1e-16))

    # 2. Bound Vortex Denominator
    TBZ = (jnp.square(T) - B2) * ZSQ
    DENOM = jnp.maximum(jnp.square(XTY) + TBZ, TOLSQ)

    # 3. DRY Helper Function with NaN-safe division
    def calc_F(X, Y, RAD, RTV):
        # FB: Influence contribution from the bound (swept) segment
        FB = (T * X - B2 * Y) / RAD

        # FT: Influence contribution from the semi-infinite trailing leg
        # safe_denom prevents divide-by-zero in the unselected jnp.where branch
        safe_denom = jnp.where(RTV < TOLSQ, 1.0, RAD * RTV)
        FT = jnp.where(RTV < TOLSQ, 0.0, (X + RAD) / safe_denom)

        return FB, FT

    # Evaluate for Endpoint 1 (Left/A) and Endpoint 2 (Right/B)
    FB1, FT1 = calc_F(X1, Y1, RAD1, RTV1)
    FB2, FT2 = calc_F(X2, Y2, RAD2, RTV2)

    # 4. Final Velocity Assembly
    QB = (FB1 - FB2) / DENOM
    ZETAPI = Z / CPI

    # U: Streamwise induced velocity (Perturbation velocity)
    U = jnp.where(ZSQ < TOLSQ, 0.0, ZETAPI * QB)

    # V: Spanwise induced velocity (Sidewash)
    V = jnp.where(ZSQ < TOLSQ, 0.0, ZETAPI * (FT1 - FT2 - QB * T))

    # W: Normal induced velocity (Downwash)
    W = -(QB * XTY + FT1 * Y1 - FT2 * Y2) / CPI

    return U, V, W


@jax.jit
def supersonic_in_plane(RAD1, RAD2, Y1, Y2, TOL, XTY, CPI):
    """
    Pure JAX translation of the in-plane supersonic induction.
    Evaluates downwash analytically when the collocation point sits exactly
    in the Z=0 plane of the vortex (where RTV -> 0).
    """
    # AD-Safe Denominators (Prevents NaN gradients in unselected branches)
    safe_Y1 = jnp.where(jnp.abs(Y1) > TOL, Y1, 1.0)
    safe_Y2 = jnp.where(jnp.abs(Y2) > TOL, Y2, 1.0)
    safe_XTY = jnp.where(jnp.abs(XTY) > TOL, XTY, 1.0)

    F1 = jnp.where(jnp.abs(Y1) > TOL, RAD1 / safe_Y1, 0.0)
    F2 = jnp.where(jnp.abs(Y2) > TOL, RAD2 / safe_Y2, 0.0)

    W_in = jnp.where(jnp.abs(XTY) > TOL, (-F1 + F2) / (safe_XTY * CPI), 0.0)
    return W_in


@jax.jit
def supersonic_induction(Z, XSQ1, RO1, XSQ2, RO2, XTY, T, B2, ZSQ, TOLSQ, TOL, TOLSQ2, X1, Y1, X2, Y2, RTV1, RTV2,
                         CHORD, RNMAX, sonic_mask, recv_idx):
    """
    Pure JAX translation of the VORLAX supersonic Biot-Savart induction.

    Variable Glossary (Supersonic Additions):
    -------------------------------------------------------------------------
    CUTOFF     : Defines the boundary of the Mach cone interaction.
    REPS       : Mach cone proximity threshold.
    valid1/2   : Boolean masks. True if the point lies inside the downstream Mach cone.
    WWAVE      : The Principal Part of the singular integral. Represents the 2D wave
                 drag contribution of the panel on itself (self-induction).
    T2A / T2F  : Aft and Forward panel sweep tangents, used to detect sonic edges.
    TRANS      : Edge condition parameter. If TRANS < 0, the edge is "sonic"
                 (sweep angle exactly matches the Mach angle).
    RFLAG      : Subsonic/Supersonic leading edge flag used downstream for LE suction.
    sonic_mask : Identifies panels exhibiting mathematical singularities at Mach=sec(sweep).
    """
    CPI = 2.0 * jnp.pi
    T2 = jnp.square(T)
    ZETAPI = Z / CPI
    CUTOFF = 0.8

    # Mach Cone Distances (Real only inside the cone)
    RAD1 = jnp.where(XSQ1 > RO1, jnp.sqrt(jnp.maximum(XSQ1 - RO1, 1e-16)), 0.0)
    RAD2 = jnp.where(XSQ2 > RO2, jnp.sqrt(jnp.maximum(XSQ2 - RO2, 1e-16)), 0.0)

    # Denominator Setup
    DENOM = jnp.square(XTY) + (T2 - B2) * ZSQ
    SIGN = jnp.where(DENOM < 0, -1.0, 1.0)
    DENOM = jnp.where(jnp.abs(DENOM) < TOLSQ, SIGN * TOLSQ, DENOM)

    def calc_F(X, Y, XSQ, RO, RAD, RTV):
        REPS = CUTOFF * XSQ
        valid = (X >= TOL) & (RAD != 0.0) & (RO <= REPS) & (RTV >= TOLSQ)

        # AD-Safe denominators (only applied when 'valid' is True)
        safe_RAD = jnp.where(valid, RAD, 1.0)
        safe_RAD_RTV = jnp.where(valid, RAD * RTV, 1.0)

        # 1.0 fallback is mathematically required by VORLAX supersonic integration
        FB = jnp.where(valid, (T * X - B2 * Y) / safe_RAD, 1.0)
        FT = jnp.where(valid, X / safe_RAD_RTV, 1.0)

        return FB, FT

    FB1, FT1 = calc_F(X1, Y1, XSQ1, RO1, RAD1, RTV1)
    FB2, FT2 = calc_F(X2, Y2, XSQ2, RO2, RAD2, RTV2)

    # Global Velocity Assembly
    QB = (FB1 - FB2) / DENOM
    U = ZETAPI * QB
    V = ZETAPI * (FT1 - FT2 - QB * T)
    W = -(QB * XTY + FT1 * Y1 - FT2 * Y2) / CPI

    # In-Plane Singularity Override
    in_plane = ZSQ < TOLSQ2
    W_in = supersonic_in_plane(RAD1, RAD2, Y1, Y2, TOL, XTY, CPI)

    U = jnp.where(in_plane, 0.0, U)
    V = jnp.where(in_plane, 0.0, V)
    W = jnp.where(in_plane, W_in, W)

    # WWAVE: Principal Part of the Integral (Self-Influence / Wave Drag)
    N = U.shape[1]
    COX = CHORD # / RNMAX
    T2 = T ** 2

    WWAVE_cond = B2 > T2[None, :]
    WWAVE_output = -0.5 * jnp.sqrt(jnp.where(WWAVE_cond, B2 - T2[None, :], 1.0)) / jnp.maximum(COX, 1e-12)

    # Calculate WWAVE for all senders (Shape: n_time, N)
    WWAVE_val = jnp.where(
        WWAVE_cond,
        WWAVE_output,
        0.0
    )

    # Create a boolean mask for the diagonal element of this specific row
    j_indices = jnp.arange(N)
    is_diag = (j_indices == recv_idx)[None, :]

    # Only add the WWAVE value to the element where sender == receiver
    W = W + jnp.where(is_diag, WWAVE_val, 0.0)

    # Build the 1D slice of the Laplacian stencil for THIS receiver row
    sonic_row = jnp.where(
        j_indices == recv_idx, 2.0,
        jnp.where(j_indices == recv_idx - 1, -1.0,
        jnp.where(j_indices == recv_idx + 1, -1.0, 0.0))
    )[None, :]

    is_recv_sonic = sonic_mask[:, recv_idx][:, None]

    # Overwrite the influence of sonic sending panels with the smoothing stencil
    W = jnp.where(is_recv_sonic, sonic_row, W)

    return U, V, W

@jax.jit
def compute_C_mn(VD, Mach):
    """
    Computes the Aerodynamic Influence Coefficient matrix C_mn.
    Output Shape: (n_time, N, N, 3)
    """

    # Unpack Vortex Distribution Data ----------------------------------------------------------------------------------
    vortex_A = VD.bound_vortex_A.astype(jnp.float32)
    vortex_B = VD.bound_vortex_B.astype(jnp.float32)
    center = VD.bound_vortex_center.astype(jnp.float32)
    colloc = VD.collocation_points.astype(jnp.float32)

    # Local Panel Orientation ------------------------------------------------------------------------------------------
    dy = vortex_B[:, 1] - vortex_A[:, 1]
    dz = vortex_B[:, 2] - vortex_A[:, 2]

    norm_yz = jnp.maximum(jnp.sqrt(dy ** 2 + dz ** 2), 1e-16)
    costheta = dy / norm_yz
    sintheta = dz / norm_yz

    # Local Panel Sweep ------------------------------------------------------------------------------------------------
    dx_vortex = vortex_B[:, 0] - center[:, 0]
    dy_vortex = (vortex_B[:, 1] - center[:, 1]) * costheta + (vortex_B[:, 2] - center[:, 2]) * sintheta

    # s = local half-span, t = tangent of the sweep angle
    s = jnp.abs(dy_vortex)
    t = (dx_vortex / jnp.maximum(dy_vortex, 1e-16))
    t_sq = t ** 2

    # Beta-Squared = Mach^2 - 1.0 --------------------------------------------------------------------------------------
    beta_sq = (Mach.squeeze(1) ** 2 - 1.0).astype(jnp.float32)
    beta_sq_exp = beta_sq[:, None] if beta_sq.ndim == 1 else beta_sq
    is_subsonic = beta_sq_exp < 0.0

    # Sonic Mask Pre-Calc ----------------------------------------------------------------------------------------------
    t_sq_fore = jnp.where(VD.is_leading_edge, 0.0, jnp.roll(t_sq, shift=1))
    t_sq_aft = jnp.where(VD.is_trailing_edge, 0.0, jnp.roll(t_sq, shift=-1))

    sonic_check = (beta_sq_exp - t_sq_fore[None, :]) * (beta_sq_exp - t_sq_aft[None, :])
    sonic_mask = (sonic_check < 0) & VD.is_leading_edge

    # Check for singularity (Mach cone passes through panel)
    singularity_flag = jnp.where(sonic_mask, 0, 1)
    singularity_flag = jnp.where(is_subsonic, 1, singularity_flag)

    safe_beta_sq = jnp.maximum(t_sq_fore[None, :], t_sq_aft[None, :]) + 0.01
    beta_sq_exp = jnp.where(sonic_mask, safe_beta_sq, beta_sq_exp)

    # C_mn Calculation -------------------------------------------------------------------------------------------------

    tol = s / 500.0
    tol_sq = tol ** 2
    tol_sq_scl = 2500.0 * tol_sq

    # Row-wise vector-mapping to minimize peak memory usage
    def compute_row(c_pt, ct_R, st_R, recv_idx):
        dx = c_pt[0] - center[:, 0]
        dy = c_pt[1] - center[:, 1]
        dz = c_pt[2] - center[:, 2]

        y_dist = dy * costheta + dz * sintheta
        z_dist = -dy * sintheta + dz * costheta

        x_dist_left = dx + t * s
        x_dist_right = dx - t * s
        x_dist_center = dx - t * y_dist

        y_dist_left = y_dist + s
        y_dist_right = y_dist - s

        # Arrays are (N,) instead of (N, N)
        XSQ1 = x_dist_left ** 2
        XSQ2 = x_dist_right ** 2
        YSQ1 = y_dist_left ** 2
        YSQ2 = y_dist_right ** 2
        ZSQ = z_dist ** 2

        RTV1 = YSQ1 + ZSQ
        RTV2 = YSQ2 + ZSQ

        # Broadcast the Mach/Time dimension here: (n_time, 1) * (N,) -> (n_time, N)
        R01 = beta_sq_exp * RTV1[None, :]
        R02 = beta_sq_exp * RTV2[None, :]

        # --- Subsonic Kernel ---
        U_sub, V_sub, W_sub = subsonic_induction(
            XSQ1=XSQ1,
            XSQ2=XSQ2,
            XTY=x_dist_center,
            X1=x_dist_left,
            X2=x_dist_right,
            Y1=y_dist_left,
            Y2=y_dist_right,
            Z=z_dist,
            ZSQ=ZSQ,
            RTV1=RTV1,
            RTV2=RTV2,
            RO1=R01,
            RO2=R02,
            T=t,
            B2=beta_sq_exp,
            TOLSQ=tol_sq,
        )

        # --- Supersonic Kernel ---
        U_sup, V_sup, W_sup = supersonic_induction(
            XSQ1=XSQ1,
            XSQ2=XSQ2,
            XTY=x_dist_center,
            X1=x_dist_left,
            X2=x_dist_right,
            Y1=y_dist_left,
            Y2=y_dist_right,
            Z=z_dist,
            ZSQ=ZSQ,
            RTV1=RTV1,
            RTV2=RTV2,
            RO1=R01,
            RO2=R02,
            T=t,
            B2=beta_sq_exp,
            TOL=tol,
            TOLSQ=tol_sq,
            TOLSQ2=tol_sq_scl,
            CHORD=VD.chord_lengths,
            RNMAX=VD.panels_per_strip,
            sonic_mask=sonic_mask,
            recv_idx=recv_idx
        )

        # --- Blending ---
        U_ind = jnp.where(is_subsonic, U_sub, U_sup)
        V_ind = jnp.where(is_subsonic, V_sub, V_sup)
        W_ind = jnp.where(is_subsonic, W_sub, W_sup)

        # --- EW Calculation ---
        # Note: ct_S and st_S are just the global 'costheta' and 'sintheta' arrays
        COS_RS = ct_R * costheta + st_R * sintheta
        SIN_RS = st_R * costheta - ct_R * sintheta

        EW_row = W_ind * COS_RS[None, :] - V_ind * SIN_RS[None, :]

        # --- Rotate to Global Frame ---
        C_mn_row = jnp.stack([
            U_ind,
            V_ind * costheta[None, :] - W_ind * sintheta[None, :],
            V_ind * sintheta[None, :] + W_ind * costheta[None, :]
        ], axis=-1)

        # Return the tuple!
        return C_mn_row, EW_row

    C_mn_mapped, EW_mapped = jax.vmap(compute_row)(colloc, costheta, sintheta, jnp.arange(VD.total_panels))

    C_mn = jnp.swapaxes(C_mn_mapped, 0, 1)
    EW = jnp.swapaxes(EW_mapped, 0, 1)

    return C_mn.astype(jnp.float64), singularity_flag, EW


# ----------------------------------------------------------------------------------------------------------------------
#  Wing Induced Velocity Calculation
# ----------------------------------------------------------------------------------------------------------------------


@inputs(
    "system.analysis_data['vortex_distribution']",
    "state.freestream.mach_number"
)
@outputs(
    "system.analysis_data['AICs']",
    "system.analysis_data['singularities']",
    "system.analysis_data['VORLAX_EW_matrix']"
)
def compute_induced_velocity(state: "State", system: "System", settings: "Settings"):
    
    VD = system.analysis_data["vortex_distribution"]
    Mach = state.freestream.mach_number
    
    C_mn, singularity_flag, EW = compute_C_mn(VD, Mach)
    
    updated_analysis_data = system.analysis_data | {
        "AICs": C_mn,
        "singularities": singularity_flag,
        "VORLAX_EW_matrix": EW
    }

    updated_system = eqx.tree_at(lambda s: s.analysis_data, system, updated_analysis_data)
    
    return state, updated_system, settings
