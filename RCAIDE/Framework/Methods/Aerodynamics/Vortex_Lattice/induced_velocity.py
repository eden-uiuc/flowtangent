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
def subsonic_jax(Z, XSQ1, RO1, XSQ2, RO2, XTY, T, B2, ZSQ, TOLSQ, X1, Y1, X2, Y2, RTV1, RTV2):
    """ Pure JAX translation of the VORLAX subsonic Biot-Savart induction """
    CPI = 4.0 * jnp.pi
    
    RAD1 = jnp.sqrt(jnp.maximum(XSQ1 - RO1, 0.0))
    RAD2 = jnp.sqrt(jnp.maximum(XSQ2 - RO2, 0.0))
    
    TBZ = (T * T - B2) * ZSQ
    DENOM = XTY * XTY + TBZ
    DENOM = jnp.maximum(DENOM, TOLSQ)
    
    FB1 = (T * X1 - B2 * Y1) / RAD1
    FT1 = jnp.where(RTV1 < TOLSQ, 0.0, (X1 + RAD1) / (RAD1 * RTV1))
    
    FB2 = (T * X2 - B2 * Y2) / RAD2
    FT2 = jnp.where(RTV2 < TOLSQ, 0.0, (X2 + RAD2) / (RAD2 * RTV2))
    
    QB = (FB1 - FB2) / DENOM
    ZETAPI = Z / CPI
    
    U = jnp.where(ZSQ < TOLSQ, 0.0, ZETAPI * QB)
    V = jnp.where(ZSQ < TOLSQ, 0.0, ZETAPI * (FT1 - FT2 - QB * T))
    W = -(QB * XTY + FT1 * Y1 - FT2 * Y2) / CPI
    
    return U, V, W

@jax.jit
def supersonic_in_plane_jax(RAD1, RAD2, Y1, Y2, TOL, XTY, CPI):
    """ Pure JAX translation of the in-plane supersonic induction """
    F1 = jnp.where(jnp.abs(Y1) > TOL, RAD1 / Y1, 0.0)
    F2 = jnp.where(jnp.abs(Y2) > TOL, RAD2 / Y2, 0.0)
    W = jnp.where(jnp.abs(XTY) > TOL, (-F1 + F2) / (XTY * CPI), 0.0)
    return W

@jax.jit
def supersonic_jax(Z, XSQ1, RO1, XSQ2, RO2, XTY, T, B2, ZSQ, TOLSQ, TOL, TOLSQ2, X1, Y1, X2, Y2, RTV1, RTV2, CHORD, RNMAX, TE_ind, LE_ind):
    """ Pure JAX translation of the VORLAX supersonic Biot-Savart induction """
    CPI = 2.0 * jnp.pi
    T2 = T * T
    ZETAPI = Z / CPI
    CUTOFF = 0.8
    
    RAD1 = jnp.where(XSQ1 > RO1, jnp.sqrt(jnp.maximum(XSQ1 - RO1, 0.0)), 0.0)
    RAD2 = jnp.where(XSQ2 > RO2, jnp.sqrt(jnp.maximum(XSQ2 - RO2, 0.0)), 0.0)
    
    DENOM = XTY * XTY + (T2 - B2) * ZSQ
    SIGN = jnp.where(DENOM < 0, -1.0, 1.0)
    DENOM = jnp.where(jnp.abs(DENOM) < TOLSQ, SIGN * TOLSQ, DENOM)
    
    REPS1 = CUTOFF * XSQ1
    valid1 = (X1 >= TOL) & (RAD1 != 0.0) & (RO1 <= REPS1) & (RTV1 >= TOLSQ)
    FB1 = jnp.where(valid1, (T * X1 - B2 * Y1) / RAD1, 1.0)
    FT1 = jnp.where(valid1, X1 / (RAD1 * RTV1), 1.0)
    
    REPS2 = CUTOFF * XSQ2
    valid2 = (X2 >= TOL) & (RAD2 != 0.0) & (RO2 <= REPS2) & (RTV2 >= TOLSQ)
    FB2 = jnp.where(valid2, (T * X2 - B2 * Y2) / RAD2, 1.0)
    FT2 = jnp.where(valid2, X2 / (RAD2 * RTV2), 1.0)
    
    QB = (FB1 - FB2) / DENOM
    U = ZETAPI * QB
    V = ZETAPI * (FT1 - FT2 - QB * T)
    W = -(QB * XTY + FT1 * Y1 - FT2 * Y2) / CPI
    
    in_plane = ZSQ < TOLSQ2
    W_in = supersonic_in_plane_jax(RAD1, RAD2, Y1, Y2, TOL, XTY, CPI)
    
    U = jnp.where(in_plane, 0.0, U)
    V = jnp.where(in_plane, 0.0, V)
    W = jnp.where(in_plane, W_in, W)
    
    # Principal Part of the Integral (WWAVE)
    N = U.shape[2] # Number of sending panels
    eye = jnp.eye(N)[None, :, :]
    COX = (CHORD / RNMAX)[None, None, :] * eye
    WWAVE_cond = (B2 * eye > T2 * eye)
    WWAVE = jnp.where(WWAVE_cond, -0.5 * jnp.sqrt(jnp.maximum(B2 * eye - T2 * eye, 0.0)) / jnp.maximum(COX, 1e-12), 0.0)
    W = W + WWAVE
    
    # The Sonic Vortex Fix
    T2S = T2[0, :] # Shape (N,)
    T2F = jnp.where(TE_ind, 0.0, jnp.roll(T2S, shift=-1))
    T2A = jnp.where(LE_ind, 0.0, jnp.roll(T2S, shift=1))
    
    TRANS = (B2[:, 0, :] - T2F[None, :]) * (B2[:, 0, :] - T2A[None, :])
    RFLAG = jnp.where(TRANS < 0, 0, 1) # Shape (n_time, N)
    sonic_mask = (TRANS < 0)
    
    sonic_matrix = jnp.diag(jnp.full(N, 2.0)) + jnp.diag(jnp.full(N-1, -1.0), k=-1) + jnp.diag(jnp.full(N-1, -1.0), k=1)
    sonic_matrix_3d = jnp.broadcast_to(sonic_matrix[None, :, :], W.shape)
    
    # Where sending panel j is sonic, overwrite the entire column j with the sonic smoothing operator
    W = jnp.where(sonic_mask[:, None, :], sonic_matrix_3d, W)
    
    return U, V, W, RFLAG

@jax.jit
def compute_induced_velocity_matrix(VD, mach_array):
    """
    Computes the Aerodynamic Influence Coefficient matrix C_mn.
    Output Shape: (n_time, N, N, 3)
    """
    
    # 1. Coordinate Symmetry Flip
    flip = VD.bound_vortex_left[:, 1] > VD.bound_vortex_right[:, 1]
    
    xa = jnp.where(flip, VD.bound_vortex_right[:, 0], VD.bound_vortex_left[:, 0])
    ya = jnp.where(flip, VD.bound_vortex_right[:, 1], VD.bound_vortex_left[:, 1])
    za = jnp.where(flip, VD.bound_vortex_right[:, 2], VD.bound_vortex_left[:, 2])
    
    xb = jnp.where(flip, VD.bound_vortex_left[:, 0], VD.bound_vortex_right[:, 0])
    yb = jnp.where(flip, VD.bound_vortex_left[:, 1], VD.bound_vortex_right[:, 1])
    zb = jnp.where(flip, VD.bound_vortex_left[:, 2], VD.bound_vortex_right[:, 2])
    
    xc = 0.5 * (xa + xb)
    yc = 0.5 * (ya + yb)
    zc = 0.5 * (za + zb)
    
    xo = VD.collocation_points[:, 0]
    yo = VD.collocation_points[:, 1]
    zo = VD.collocation_points[:, 2]
    
    theta = jnp.arctan2(zb - za, yb - ya)
    costheta = jnp.cos(theta)
    sintheta = jnp.sin(theta)
    
    # 2. Broadcasting Spatial Distances to (N, N)
    xobar = xo[:, None] - xc[None, :]
    y_diff = yo[:, None] - yc[None, :]
    z_diff = zo[:, None] - zc[None, :]
    
    ct = costheta[None, :]
    st = sintheta[None, :]
    
    yobar = y_diff * ct + z_diff * st
    zobar = -y_diff * st + z_diff * ct
    
    x1bar = xb - xc
    y1bar = (yb - yc) * costheta + (zb - zc) * sintheta
    
    s = jnp.abs(y1bar)[None, :]
    t = (x1bar / y1bar)[None, :]
    
    X1 = xobar + t * s 
    Y1 = yobar + s   
    X2 = xobar - t * s 
    Y2 = yobar - s   
    XTY = xobar - t * yobar
    
    # 3. Setup broadcasted tolerances (N, N)
    TOL = s / 500.0
    TOLSQ = TOL * TOL
    TOLSQ2 = 2500.0 * TOLSQ
    ZSQ = zobar * zobar
    YSQ1 = Y1 * Y1
    YSQ2 = Y2 * Y2
    RTV1 = YSQ1 + ZSQ
    RTV2 = YSQ2 + ZSQ
    XSQ1 = X1 * X1
    XSQ2 = X2 * X2
    
    # 4. Temporal Mach Broadcasting to (n_time, N, N)
    mach = mach_array[:, None, None]
    B2 = jnp.square(mach) - 1.0
    RO1 = B2 * RTV1[None, :, :]
    RO2 = B2 * RTV2[None, :, :]
    
    # 5. Evaluate Full Subsonic and Supersonic Regimes
    U_sub, V_sub, W_sub = subsonic_jax(
        zobar[None, :, :], XSQ1[None, :, :], RO1, XSQ2[None, :, :], RO2, 
        XTY[None, :, :], t[None, :, :], B2, ZSQ[None, :, :], TOLSQ[None, :, :], 
        X1[None, :, :], Y1[None, :, :], X2[None, :, :], Y2[None, :, :], 
        RTV1[None, :, :], RTV2[None, :, :]
    )
    
    U_sup, V_sup, W_sup, RFLAG = supersonic_jax(
        zobar[None, :, :], XSQ1[None, :, :], RO1, XSQ2[None, :, :], RO2, 
        XTY[None, :, :], t[None, :, :], B2, ZSQ[None, :, :], TOLSQ[None, :, :], TOL[None, :, :], TOLSQ2[None, :, :], 
        X1[None, :, :], Y1[None, :, :], X2[None, :, :], Y2[None, :, :], 
        RTV1[None, :, :], RTV2[None, :, :], 
        VD.chord_lengths, VD.panels_per_strip, VD.is_trailing_edge, VD.is_leading_edge
    )
    
    # 6. Blend based on Compressibility Condition
    is_subsonic = (B2 < 0)[:, :, :]
    
    U = jnp.where(is_subsonic, U_sub, U_sup)
    V = jnp.where(is_subsonic, V_sub, V_sup)
    W = jnp.where(is_subsonic, W_sub, W_sup)
    
    # Fix RFLAG (Subsonic is always 1)
    RFLAG = jnp.where(is_subsonic[:, :, 0], 1, RFLAG)
    
    # Legacy VORLAX downwash calcuation
    
    # Panel Dihedral Angle (DL) using bound vortex nodes (AH and BH)
    # Original: D = sqrt((YAH-YBH)**2 + (ZAH-ZBH)**2)
    dy_h = VD.bound_vortex_right[:, 1] - VD.bound_vortex_left[:, 1]
    dz_h = VD.bound_vortex_right[:, 2] - VD.bound_vortex_left[:, 2]

    DL = jnp.arctan2(dz_h, dy_h)
    DL = jnp.where(DL > jnp.pi / 2.0, DL - jnp.pi, DL)
    DL = jnp.where(DL < -jnp.pi / 2.0, DL + jnp.pi, DL)

    # Broadcast the relative dihedral: DL.T - DL -> DL_receiver - DL_sender
    DL_diff = DL[:, None] - DL[None, :]
    
    # Broadcast over the time dimension (n_time, N, N)
    COS1 = jnp.cos(DL_diff)[None, :, :]
    SIN1 = jnp.sin(DL_diff)[None, :, :]
    
    # Calculate EW
    EW = W * COS1 - V * SIN1

    # Rotate back into Vehicle Frame for C_mn
    C_mn = jnp.stack([
        U, 
        V * costheta[None, None, :] - W * sintheta[None, None, :], 
        V * sintheta[None, None, :] + W * costheta[None, None, :]
    ], axis=-1)
    
    return C_mn, RFLAG, EW

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
    mach_array = state.freestream.mach_number
    
    C_mn, RFLAG, EW = compute_induced_velocity_matrix(VD, mach_array)
    
    updated_analysis_data = system.analysis_data | {
        "AICs": C_mn,
        "singularities": RFLAG,
        "VORLAX_EW_matrix": EW
    }

    updated_system = eqx.tree_at(lambda s: s.analysis_data, system, updated_analysis_data)
    
    return state, updated_system, settings
