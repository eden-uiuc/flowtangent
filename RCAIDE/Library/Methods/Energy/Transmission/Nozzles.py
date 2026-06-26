# RCAIDE/Library/Methods/Propulsors/Converters/compression_nozzle.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: Mar 2025, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from __future__ import annotations
from typing import TYPE_CHECKING

# package imports
import jax.numpy as jnp

# RCAIDE imports
if TYPE_CHECKING:
    import RCAIDE.Framework as rcf

# ----------------------------------------------------------------------------------------------------------------------
#  Compression Nozzle Functional Methods
# ----------------------------------------------------------------------------------------------------------------------


def func_isentropic_nozzle_performance(
    T_t, P_t,
    P0, gamma,
    PR, n_r,
):

    # Isentropic Outputs
    P_t_out = jnp.maximum(P_t * PR * n_r, P0)   # Output stagnation pressure, minimum is freestream pressure
    T_t_out = T_t                               # Output stagnation temperature, adiabatically conserved

    M_out   = jnp.sqrt((((P_t_out / P0) ** ((gamma - 1.) / gamma)) - 1.) * 2. / (gamma - 1.))  # Output Mach number
    T_out   = T_t_out / (1. + (gamma - 1.) / 2. * M_out ** 2)                                  # Output temperature

    return P_t_out, T_t_out, T_out, M_out


def func_compression_nozzle_performance(
    gas,          # Replaces Cp, gamma
    T_t, P_t,
    P0, M0,
    PR, n_r
):
    # Dynamically evaluate gamma for the isentropic and shock relations
    gamma = gas.compute_gamma(T_t)

    (P_t_out, T_t_out,
     T_out, M_out) = func_isentropic_nozzle_performance(
       T_t, P_t,
       P0, gamma,
       PR, n_r)

    # Normal Shock Outputs (Evaluated dynamically)
    ns_M    = jnp.sqrt((1. + (gamma - 1.) / 2. * M0 ** 2.) / (gamma * M0 ** 2 - (gamma - 1.) / 2.))
    ns_T    = T_t_out / (1. + (gamma - 1.) / 2 * ns_M ** 2)
    ns_P_t  = (PR *
               P_t *
               ((((gamma + 1.) * (M0 ** 2.)) / ((gamma - 1.) * M0 ** 2. + 2.)) ** (gamma / (gamma - 1.))) *
               ((gamma + 1.) / (2. * gamma * M0 ** 2.-(gamma - 1.))) ** (1./(gamma - 1.))
               )

    # Combine Outputs
    P_t_out = jnp.where(M0 > 1.0, ns_P_t, P_t_out)
    T_out   = jnp.where(M0 > 1.0, ns_T, T_out)
    M_out   = jnp.where(M0 > 1.0, ns_M, M_out)

    # Calculate Velocity using Absolute Enthalpy (Strictly enforces First Law)
    h_t_out = gas.compute_enthalpy(T_t_out)
    h_out   = gas.compute_enthalpy(T_out)                     
    u_out   = jnp.sqrt(2. * (h_t_out - h_out))  

    return M_out, u_out, P_t_out, T_t_out, T_out, h_t_out, h_out

def func_expansion_nozzle_performance(
    gas,          # Local working fluid (e.g., BurnedGas)
    T_t, T_t0,
    P_t, P_t0,
    P0, M0,
    gamma_0,      # Explicit freestream gamma
    PR,
):
    # Dynamic gas properties for the exhaust flow
    gamma = gas.compute_gamma(T_t)
    R = gas.R_specific

    (P_t_out, T_t_out,
     T_out_isn, M_isn) = func_isentropic_nozzle_performance(
         T_t, P_t,
         P0, gamma,
         PR, 1.0
    )

    # Supersonic Expansion / Choking Logic
    sup     = M_isn > 1.0
    M_out   = jnp.maximum(jnp.minimum(M_isn, 1.0), 0.001)  # Bound Mach number to [0.001, 1]
    
    # Recalculate static conditions based on the bounded Mach number
    P       = P_t_out / (1. + (gamma - 1.) / 2. * M_out ** 2) ** (gamma / (gamma - 1.))
    P_out   = jnp.where(sup, P, P0)

    T_out   = T_t_out / (1. + (gamma - 1.) / 2. * M_out ** 2)

    # High-fidelity absolute enthalpy and velocity
    h_t_out = gas.compute_enthalpy(T_t_out)
    h_out   = gas.compute_enthalpy(T_out)
    u_out   = jnp.sqrt(2. * (h_t_out - h_out))
    
    # Gas density based on the specific gas constant
    r_out   = P_out / (R * T_out)

    def fm(M, g):
        m0 = (g + 1.) / (2. * (g - 1.))
        m1 = ((g + 1.) / 2.) ** m0
        m2 = (1. + (g - 1.) / 2. * M * M) ** m0
        return m1 * M / m2

    # Area Ratio calculation rigorously separates freestream and exhaust gammas
    AR = (fm(M0, gamma_0) / fm(M_out, gamma) * (1. / (P_t_out / P_t0)) * (jnp.sqrt(T_t_out / T_t0)))

    return AR, M_out, r_out, u_out, P_out, P_t_out, T_out, T_t_out, h_out, h_t_out


def _expansion_nozzle_performance(
    state: "rcf.State",
    system: "rcf.Systems",
    settings: "rcf.Settings",
    input_converter_state,
    output_converter_state,
    PR,
    n_p
):

    # Get Inputs
    T_t = input_converter_state.outputs.stagnation_temperature
    P_t = input_converter_state.stagnation_pressure

    fs      = state.freestream
    P0      = fs.pressure
    M0      = fs.mach_number
    Cp      = fs.Cp
    gamma   = fs.gamma
    R       = fs.R
    P_t0    = fs.stagnation_pressure
    T_t0    = fs.stagnation_temperature
    # Call function

    (AR, M,
     r_out, u_out,
     P_out, P_t_out,
     T_out, T_t_out,
     h_out, h_t_out) = func_expansion_nozzle_performance(
        T_t, T_t0,
        P_t, P_t0,
        P0, M0,
        Cp, gamma, R,
        PR, n_p
    )

    # Set Input State
    inputs = input_converter_state

    inputs.stagnation_temperature               = T_t
    inputs.stagnation_pressure                  = P_t

    inputs.freestream_stagnation_temperature    = T_t0
    inputs.freestream_stagnation_pressure       = P_t0
    inputs.freestream_pressure                  = P0
    inputs.freestream_mach_number               = M0
    inputs.freestream_Cp                        = Cp
    inputs.freestream_gamma                     = gamma
    inputs.freestream_R                         = R

    # Set Output State
    outputs = output_converter_state

    outputs.area_ratio                          = AR
    outputs.mach_number                         = M

    outputs.density                             = r_out
    outputs.velocity                            = u_out

    outputs.pressure                            = P_out
    outputs.stagnation_pressure                 = P_t_out

    outputs.temperature                         = T_out
    outputs.stagnation_temperature              = T_t_out

    outputs.enthalpy                            = h_out
    outputs.stagnation_enthalpy                 = h_t_out

    return state, system, settings


def fan_nozzle_performance(
    state: "rcf.State",
    system: "rcf.Aircraft",
    settings: "rcf.Settings"
):
    # Get Inputs

    for l_idx, line in enumerate(system.energy.lines):
        for p_idx, prop in enumerate(line.propulsors):

            nozzle = prop.converters.fan_nozzle
            PR  = nozzle.pressure_ratio
            n_p = nozzle.efficiencies.flow

            state, system, settings = _expansion_nozzle_performance(state, system, settings,
                                                                    prop.converters.fan,
                                                                    prop.converters.fan_nozzle,
                                                                    PR, n_p)

    return state, system, settings


def core_nozzle_performance(
    state: "rcf.State",
    system: "rcf.Aircraft",
    settings: "rcf.Settings"
):
    # Get Inputs

    for l_idx, line in enumerate(system.energy.lines):
        for p_idx, prop in enumerate(line.propulsors):

            nozzle = prop.converters.core_nozzle
            PR = nozzle.pressure_ratio
            n_p = nozzle.efficiencies.flow

            state, system, settings = _expansion_nozzle_performance(state, system, settings,
                                                                    prop.converters.turbines[-1],
                                                                    prop.converters.core_nozzle,
                                                                    PR, n_p)

    return state, system, settings
