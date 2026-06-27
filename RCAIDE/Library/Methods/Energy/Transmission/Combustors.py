# $NAME.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: Mar 2025, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from RCAIDE.Framework import Aircraft, Settings, State

# package imports
import jax.numpy as jnp

# RCAIDE imports
from RCAIDE.Library.gases import CO2, H2O, O2, IdealGas
from RCAIDE.Library.Methods.Energy.Transmission import Rayleigh, fM

# ----------------------------------------------------------------------------------------------------------------------
#  Combustor Performance
# ----------------------------------------------------------------------------------------------------------------------


def func_combustor_performance(
    gas: IdealGas,
    T_t,
    P_t,
    T_t_out,
    h_t_f,
    PR,
    n_b,
):

    h_t = gas.compute_enthalpy(T_t)
    P_t_out = P_t * PR

    # Enthalpy of gases at exit temp
    h_gas_out = gas.compute_enthalpy(T_t_out)
    h_O2 = O2.compute_enthalpy(T_t_out)
    h_CO2 = CO2.compute_enthalpy(T_t_out)
    h_H2O = H2O.compute_enthalpy(T_t_out)

    # Jet-A Reaction Enthalpy
    dh_react = (3.155 * h_CO2) + (1.242 * h_H2O) - (3.396 * h_O2)

    numerator = h_gas_out - h_t
    denominator = (n_b * h_t_f) - dh_react - h_gas_out

    FAR = numerator / denominator

    # 6. Final output mixture enthalpy
    h_t_out = (h_gas_out + FAR * dh_react) / (1.0 + FAR)

    return P_t_out, h_t_out, jnp.atleast_2d(FAR)


def jet_combustor_transmission(state: State, system: Aircraft, settings: Settings):

    for l_idx, line in enumerate(system.energy.lines):
        for p_idx, propulsor in enumerate(line.propulsors):
            comp_outputs = state.energy.lines[l_idx].propulsors[p_idx].compressors[-1].outputs
            T_t_in = comp_outputs.stagnation_temperature
            P_t_in = comp_outputs.stagnation_pressure

            T_t_out = propulsor.converters.turbines[0].design_intake_temperature

            Cp = state.freestream.Cp

            combustor = propulsor.converters.combustor
            PR = combustor.pressure_ratio
            n_b = combustor.efficiency

            h_t_f = propulsor.fuel.specific_energy

            # Call the function
            P_t_out, T_t_out, h_t_out, f = func_combustor_performance(T_t_in, P_t_in, T_t_out, Cp, PR, n_b, h_t_f)

            # Set Input State
            combustor_state = state.energy.lines[l_idx].propulsors[p_idx].combustor

            inputs = combustor_state.inputs
            inputs.freestream_Cp = Cp
            inputs.stagnation_temperature = T_t_in
            inputs.stagnation_pressure = P_t_in

            # Set Output State

            outputs = combustor_state.outputs
            outputs.stagnation_pressure = P_t_out
            outputs.stagnation_temperature = T_t_out
            outputs.stagnation_enthalpy = h_t_out
            outputs.fuel_air_ratio = f

    return state, system, settings


# ----------------------------------------------------------------------------------------------------------------------
# Rayleigh Line Flow
# ----------------------------------------------------------------------------------------------------------------------


def func_rayleigh_line_flow(
    g,
    Cp,
    Tt_in,
    Pt_in,
    M0,
    Tt_4,
    eta_b,
    ht_f,
    AR,
):

    # Isentropic deceleration through divergent nozzle
    M1 = jnp.atleast_2d(fM(AR, M0[:, 0], g[:, 0])).T

    # Max stagnation temperature to thermally choke flow
    Tt_out_Rayleigh = Tt_in * (1.0 + g * M1**2) ** 2.0 / ((2.0 * (1.0 + g) * M1**2) * (1.0 + (g - 1.0) / 2.0 * M1**2))

    # Limit Tt_out
    Tt_out = jnp.ones_like(Tt_out_Rayleigh) * Tt_4
    Tt_out = jnp.minimum(Tt_out, Tt_out_Rayleigh)

    # Rayleigh calculations
    M_out = jnp.zeros_like(Pt_in)
    Pt_R = jnp.zeros_like(Pt_in)
    M_out[:, 0], Pt_R[:, 0] = Rayleigh(g[:, 0], M1[:, 0], Tt_out[:, 0] / Tt_in[:, 0])
    Pt_out = Pt_R * Pt_in

    # Stagnation enthalpies from stagnation temperatures
    ht_in = Tt_in * Cp
    ht_out = Tt_out * Cp

    # Fuel air ratio
    f = (ht_out - ht_in) / (eta_b * ht_f - ht_out)

    return Tt_out, Pt_out, ht_out, M_out, f


def rayleigh_line_flow(state: State, system: Aircraft, settings: Settings):

    for l_idx, line in system.energy.lines:
        for p_idx, propulsor in enumerate(line.propulsors):
            combustor = propulsor.propulsors.combustor
            combustor_state = state.energy.lines[l_idx].propulsors[p_idx].combustor

            Tt_in = combustor_state.inputs.stagnation_temperature
            Pt_in = combustor_state.inputs.stagnation_pressure
            M0 = combustor_state.inputs.mach_number

            eta_b = combustor.efficiency
            AR = combustor.area_ratio

            ht_f = propulsor.fuel.specific_energy
            Tt_4 = propulsor.propulsors.turbines[0].design_intake_temperature

            g = state.freestream.gamma
            Cp = state.freestream.Cp

            Tt_out, Pt_out, ht_out, M_out, f = func_rayleigh_line_flow(g, Cp, Tt_in, Pt_in, M0, Tt_4, eta_b, ht_f, AR)

            combustor_state.outputs.stagnation_temperature = Tt_out
            combustor_state.outputs.stagnation_pressure = Pt_out
            combustor_state.outputs.stagnation_enthalpy = ht_out
            combustor_state.outputs.mach_number = M_out
            combustor_state.outputs.fuel_to_air_ratio = f

    return state, system, settings
