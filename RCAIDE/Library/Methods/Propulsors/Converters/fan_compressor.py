# RCAIDE/Library/Methods/Propulsors/Converters/fan_compressor.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: Mar 2025, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from dataclasses import dataclass, field

# package imports
import numpy as np

# RCAIDE imports
import RCAIDE.Framework as rcf

# ----------------------------------------------------------------------------------------------------------------------
#  Compressor
# ----------------------------------------------------------------------------------------------------------------------


def func_fan_compressor_performance(
        g,
        Cp,
        T_t,
        P_t,
        PR,
        n_p
):

    T_t_out = T_t * (PR ** ((g - 1.) / (g * n_p)))

    h_t     = T_t * Cp
    h_t_out = T_t_out * Cp
    work    = h_t_out - h_t

    P_t_out = P_t * PR

    return work, P_t_out, T_t_out, h_t_out


def fan_performance(state: rcf.State,
                    system: rcf.System,
                    settings: rcf.Settings):

    # Get inputs

    g = state.conditions.freestream.gamma
    Cp = state.conditions.freestream.Cp

    inputs = system.energy.converters.compression_nozzle.outputs
    T_t = inputs.stagnation_temperature
    P_t = inputs.stagnation_pressure

    PR = system.energy.converters.fan.pressure_ratio
    n_p = system.energy.converters.fan.polytropic_efficiency

    work, P_t_out, T_t_out, h_t_out = func_fan_compressor_performance(g, Cp, T_t, P_t, PR, n_p)

    # Set outputs
    cond = state.energy.converters.fan
    cond.outputs.work                   = work
    cond.outputs.stagnation_pressure    = P_t_out
    cond.outputs.stagnation_temperature = T_t_out
    cond.outputs.stagnation_enthalpy    = h_t_out

    return state, system, settings


def compressor_performance(state: rcf.State,
                           system: rcf.System,
                           settings: rcf.Settings):

    # Get inputs

    g = state.conditions.freestream.gamma
    Cp = state.conditions.freestream.Cp

    inputs = system.energy.converters.compression_nozzle.outputs
    T_t = inputs.stagnation_temperature
    P_t = inputs.stagnation_pressure

    for idx, comp in enumerate(system.energy.converters.compressors):

        PR = comp.pressure_ratio
        n_p = comp.polytropic_efficiency

        work, P_t_out, T_t_out, h_t_out = func_fan_compressor_performance(g, Cp, T_t, P_t, PR, n_p)

        # Set outputs
        cond = state.energy.compressors[idx]
        cond.outputs.work                   = work
        cond.outputs.stagnation_pressure    = P_t_out
        cond.outputs.stagnation_temperature = T_t_out
        cond.outputs.stagnation_enthalpy    = h_t_out

        # Set outputs of current compressor as inputs for next compressor
        T_t = T_t_out
        P_t = P_t_out

    return state, system, settings
