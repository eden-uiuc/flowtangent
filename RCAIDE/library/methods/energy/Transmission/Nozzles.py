# RCAIDE/Library/Methods/Propulsors/Converters/compression_nozzle.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: Mar 2025, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from RCAIDE.library.gases import IdealGas

# package imports
import jax.numpy as jnp

# RCAIDE imports


# ----------------------------------------------------------------------------------------------------------------------
#  Compression Nozzle Functional Methods
# ----------------------------------------------------------------------------------------------------------------------


def func_isentropic_nozzle_performance(
    T_t: jnp.ndarray,
    P_t: jnp.ndarray,
    P0: jnp.ndarray,
    gamma: jnp.ndarray,
    PR: jnp.ndarray | float,
    n_r: jnp.ndarray | float,
):

    # Isentropic Outputs
    P_t_out = jnp.maximum(P_t * PR * n_r, P0)  # Output stagnation pressure, minimum is freestream pressure
    T_t_out = T_t  # Output stagnation temperature, adiabatically conserved

    M_out = jnp.sqrt((((P_t_out / P0) ** ((gamma - 1.0) / gamma)) - 1.0) * 2.0 / (gamma - 1.0))  # Output Mach number
    T_out = T_t_out / (1.0 + (gamma - 1.0) / 2.0 * M_out**2)  # Output temperature

    return P_t_out, T_t_out, T_out, M_out


def func_compression_nozzle_performance(
    gas: IdealGas,
    T_t: jnp.ndarray,
    P_t: jnp.ndarray,
    P0: jnp.ndarray,
    M0: jnp.ndarray,
    PR: jnp.ndarray | float,
    n_r: jnp.ndarray | float,
):
    # Dynamically evaluate gamma for the isentropic and shock relations
    gamma = gas.compute_gamma(T_t)

    (P_t_out, T_t_out, T_out, M_out) = func_isentropic_nozzle_performance(T_t, P_t, P0, gamma, PR, n_r)

    # Normal Shock Outputs (Evaluated dynamically)
    ns_M = jnp.sqrt((1.0 + (gamma - 1.0) / 2.0 * M0**2.0) / (gamma * M0**2 - (gamma - 1.0) / 2.0))
    ns_T = T_t_out / (1.0 + (gamma - 1.0) / 2 * ns_M**2)
    ns_P_t = (
        PR
        * P_t
        * ((((gamma + 1.0) * (M0**2.0)) / ((gamma - 1.0) * M0**2.0 + 2.0)) ** (gamma / (gamma - 1.0)))
        * ((gamma + 1.0) / (2.0 * gamma * M0**2.0 - (gamma - 1.0))) ** (1.0 / (gamma - 1.0))
    )

    # Combine Outputs
    P_t_out = jnp.where(M0 > 1.0, ns_P_t, P_t_out)
    T_out = jnp.where(M0 > 1.0, ns_T, T_out)
    M_out = jnp.where(M0 > 1.0, ns_M, M_out)

    # Calculate Velocity using Absolute Enthalpy (Strictly enforces First Law)
    h_t_out = gas.compute_enthalpy(T_t_out)
    h_out = gas.compute_enthalpy(T_out)
    u_out = jnp.sqrt(2.0 * (h_t_out - h_out))

    return M_out, u_out, P_t_out, T_t_out, T_out, h_t_out, h_out


def func_expansion_nozzle_design(
    gas,  # Local working fluid (e.g., BurnedGas)
    T_t,
    T_t0,
    P_t,
    P_t0,
    P0,
    M0,
    gamma_0,  # Explicit freestream gamma
    PR,
):
    # Dynamic gas properties for the exhaust flow
    gamma = gas.compute_gamma(T_t)
    R = gas.R_specific

    P_t_out, T_t_out, T_out, M_isn = func_isentropic_nozzle_performance(T_t, P_t, P0, gamma, PR, 1.0)

    # Supersonic Expansion / Choking Logic
    sup = M_isn > 1.0
    M_out = jnp.maximum(jnp.minimum(M_isn, 1.0), 0.001)  # Bound Mach number to [0.001, 1]

    # Recalculate static conditions based on the bounded Mach number
    P = P_t_out / (1.0 + (gamma - 1.0) / 2.0 * M_out**2) ** (gamma / (gamma - 1.0))
    P_out = jnp.where(sup, P, P0)

    T_out = T_t_out / (1.0 + (gamma - 1.0) / 2.0 * M_out**2)

    # High-fidelity absolute enthalpy and velocity
    h_t_out = gas.compute_enthalpy(T_t_out)
    h_out = gas.compute_enthalpy(T_out)
    u_out = jnp.sqrt(2.0 * (h_t_out - h_out))

    # Gas density based on the specific gas constant
    r_out = P_out / (R * T_out)

    def fm(M, g):
        m0 = (g + 1.0) / (2.0 * (g - 1.0))
        m1 = ((g + 1.0) / 2.0) ** m0
        m2 = (1.0 + (g - 1.0) / 2.0 * M * M) ** m0
        return m1 * M / m2

    # Area Ratio calculation rigorously separates freestream and exhaust gammas
    AR = fm(M0, gamma_0) / fm(M_out, gamma) * (1.0 / (P_t_out / P_t0)) * (jnp.sqrt(T_t_out / T_t0))

    return AR, M_out, r_out, u_out, P_out, P_t_out, T_out, T_t_out, h_out, h_t_out

def func_expansion_nozzle_design(
        gas: IdealGas,  # Local working fluid (e.g., BurnedGas)
        T_t: jnp.ndarray,
        T_t0: jnp.ndarray,
        P_t: jnp.ndarray,
        P_t0: jnp.ndarray,
        mdot: jnp.ndarray,
        P0: jnp.ndarray,
        M0: jnp.ndarray,
        gamma_0: jnp.ndarray,
        PR: jnp.ndarray | float,
):
    # Dynamic gas properties for the exhaust flow
    gamma = gas.compute_gamma(T_t)
    R = gas.R_specific

    P_t_out, T_t_out, T_out, M_isn = func_isentropic_nozzle_performance(T_t, P_t, P0, gamma, PR, 1.0)

    # Supersonic Expansion / Choking Logic
    critical_PR = (1.0 + (gamma - 1.0) / 2.0) ** (gamma / (gamma - 1.0))
    is_choked = (P_t / P0) >= critical_PR

    M_out = jnp.maximum(M_isn, 0.001)

    # Recalculate static conditions
    P_out = P_t_out / (1.0 + (gamma - 1.0) / 2.0 * M_out**2) ** (gamma / (gamma - 1.0))
    P_out = jnp.where(is_choked, P_out, P0)

    T_out = T_t_out / (1.0 + (gamma - 1.0) / 2.0 * M_out**2)

    # Enthalpy and velocity
    h_t_out = gas.compute_enthalpy(T_t_out)
    h_out = gas.compute_enthalpy(T_out)
    u_out = jnp.sqrt(2.0 * (h_t_out - h_out))

    # Exit area
    rho_out = P_out / (R * T_out)
    A_exit = mdot / (rho_out * u_out)

    # Throat area
    T_star = T_t / (1.0 + (gamma - 1.0) / 2.0)
    P_star = P_t / critical_PR
    rho_star = P_star / (R * T_star)
    u_star = jnp.sqrt(gamma * R * T_star)

    A_throat_choked = mdot / (rho_star * u_star)
    A_throat = jnp.where(is_choked, A_throat_choked, A_exit)

    return A_throat, A_exit, M_out, rho_out, u_out, P_out, P_t_out, T_out, T_t_out, h_out, h_t_out

def func_expansion_nozzle_performance(
        gas: IdealGas,
        T_t: jnp.ndarray,
        P_t: jnp.ndarray,
        P0: jnp.ndarray,
        A_throat: float,
        A_exit: float,
    ):

    gamma = gas.compute_gamma(T_t)
    R = gas.R_specific
    AR = A_exit / A_throat

    # Check for choked flow
    critical_PR = (1.0 + (gamma - 1.0) / 2.0) ** (gamma / (gamma - 1.0))
    actual_PR = P_t / P0
    choked = actual_PR >= critical_PR

    # Find exit Mach number
    M_exit_sup = 2.0
    for _ in range(5):
        term = (2.0 / (gamma + 1.0)) * (1.0 + (gamma - 1.0) / 2.0 * M_exit_sup**2)
        power = (gamma + 1.0) / (2.0 * (gamma - 1.0))
        AR_calc = (1.0 / M_exit_sup) * (term ** power)
        
        # Analytical derivative: d(A/A*) / dM
        dAR_dM = AR_calc * (M_exit_sup**2 - 1.0) / (M_exit_sup * (1.0 + (gamma - 1.0) / 2.0 * M_exit_sup**2))
        
        # Newton step
        M_exit_sup = M_exit_sup - (AR_calc - AR) / dAR_dM
    
    M_exit_sub  = jnp.sqrt((2.0 / (gamma - 1.0)) * ((P_t / P0)**((gamma - 1.0) / gamma) - 1.0))
    M_exit      = jnp.where(choked, M_exit_sup, M_exit_sub)
    M_throat    = jnp.where(choked, 1.0, M_exit)

    # Nozzle Mass Flow
    m_1 = (P_t * A_throat) / jnp.sqrt(R * T_t)
    m_2 = jnp.sqrt(gamma) * M_throat
    m_3 = (1.0 + (gamma - 1.0) / 2.0 * M_throat**2) ** (- (gamma + 1.0) / (2.0 * (gamma - 1.0)))

    mdot_out = m_1 * m_2 * m_3

    P_out_choked = P_t / (1.0 + (gamma - 1.0) / 2.0 * M_exit**2) ** (gamma / (gamma - 1.0))
    P_out = jnp.where(choked, P_out_choked, P0)
    P_t_out = P_out * (1.0 + (gamma - 1.0) / 2.0 * M_exit**2) ** (gamma / (gamma - 1.0))
    
    T_out = T_t / (1.0 + (gamma - 1.0) / 2.0 * M_exit**2)
    T_t_out = T_t
    
    h_t_out = gas.compute_enthalpy(T_t_out)
    h_out = gas.compute_enthalpy(T_out)
    u_out = jnp.sqrt(2.0 * (h_t_out - h_out))

    rho_out = P_out / (R * T_out)

    return mdot_out, M_exit, u_out, rho_out, P_out, P_t_out, T_out, T_t_out, h_out, h_t_out
