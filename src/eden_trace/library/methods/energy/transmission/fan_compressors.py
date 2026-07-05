# Trace/Library/Methods/Propulsors/Converters/fan_compressor.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: Mar 2025, Trace Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING

# Trace imports
if TYPE_CHECKING:
    import src.eden_trace.framework as rcf

# ----------------------------------------------------------------------------------------------------------------------
#  Compressor
# ----------------------------------------------------------------------------------------------------------------------


def func_fan_compressor_performance(
    gas,
    T_t,
    P_t,
    PR,
    n_isn,
):
    # 1. Find ideal exit temperature with average gamma
    gamma_guess = gas.compute_gamma(T_t)
    T_t_out_ideal = T_t * (PR ** ((gamma_guess - 1.0) / gamma_guess))
    
    for _ in range(3):
        gamma_out = gas.compute_gamma(T_t_out_ideal)
        gamma_avg = 0.5 * (gamma_guess + gamma_out)
        T_t_out_ideal = T_t * (PR ** ((gamma_avg - 1.0) / gamma_avg))

    # 2. Apply isentropic efficiency
    T_t_out = T_t + (T_t_out_ideal - T_t) / n_isn

    # 3. Calculate work using absolute enthalpies
    h_t = gas.compute_enthalpy(T_t)
    h_t_out = gas.compute_enthalpy(T_t_out)
    d_work = h_t_out - h_t
    
    P_t_out = P_t * PR

    return d_work, P_t_out, T_t_out, h_t_out


def fan_performance(
    state: "rcf.state",
    system: "rcf.Aircraft",
    settings: "rcf.settings",
) -> ("rcf.State", "rcf.Aircraft", "rcf.Settings"):

    # Get inputs

    fs = state.freestream
    g = fs.gamma
    Cp = fs.Cp

    for l_idx, line in enumerate(system.energy.lines):
        for p_idx, propulsor in enumerate(line.propulsors):
            inputs = state.energy.lines[l_idx].propulsors[p_idx].converters.inlet_nozzle.outputs
            T_t = inputs.stagnation_temperature
            P_t = inputs.stagnation_pressure

            fan = propulsor.converters.fan
            PR = fan.pressure_ratio
            n_p = fan.efficiencies.flow

            work, P_t_out, T_t_out, h_t_out = func_fan_compressor_performance(g, Cp, T_t, P_t, PR, n_p)

            # Set Input State

            inputs = state.energy.lines[l_idx].propulsors[p_idx].converters.fan.inputs
            inputs.freestream_gamma = g
            inputs.freestream_Cp = Cp
            inputs.stagnation_temperature = T_t
            inputs.stagnation_pressure = P_t

            # Set Output State
            outputs = state.energy.lines[l_idx].propulsors[p_idx].converters.fan.outputs
            outputs.work = work
            outputs.stagnation_pressure = P_t_out
            outputs.stagnation_temperature = T_t_out
            outputs.stagnation_enthalpy = h_t_out

    return state, system, settings


def compressor_performance(
    state: "rcf.state",
    system: "rcf.Aircraft",
    settings: "rcf.settings",
) -> ("rcf.State", "rcf.Aircraft", "rcf.Settings"):

    # Get inputs

    fs = state.freestream
    g = fs.gamma
    Cp = fs.Cp

    for l_idx, line in enumerate(system.energy.lines):
        for p_idx, propulsor in enumerate(line.propulsors):
            inputs = state.energy.lines[l_idx].propulsors[p_idx].converters.inlet_nozzle.outputs
            T_t = inputs.stagnation_temperature
            P_t = inputs.stagnation_pressure

            for c_idx, comp in enumerate(propulsor.converters.compressors):
                PR = comp.pressure_ratio
                n_p = comp.efficiencies.flow

                work, P_t_out, T_t_out, h_t_out = func_fan_compressor_performance(g, Cp, T_t, P_t, PR, n_p)

                # Set Input State for current compressor

                inputs = state.energy.lines[l_idx].propulsors[p_idx].converters.compressors[c_idx].inputs
                inputs.freestream_gamma = g
                inputs.freestream_Cp = Cp
                inputs.stagnation_temperature = T_t
                inputs.stagnation_pressure = P_t

                # Set Output State for current compressor
                outputs = state.energy.lines[l_idx].propulsors[p_idx].converters.compressors[c_idx].outputs
                outputs.work = work
                outputs.stagnation_pressure = P_t_out
                outputs.stagnation_temperature = T_t_out
                outputs.stagnation_enthalpy = h_t_out

                # Set outputs of current compressor as inputs for next compressor
                T_t = T_t_out
                P_t = P_t_out

    return state, system, settings
