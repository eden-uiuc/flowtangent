# $NAME.py
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
#  Combustor Performance
# ----------------------------------------------------------------------------------------------------------------------


def func_combustor_performance(T_t_in,
                               P_t_in,
                               T_t_out,
                               PR,
                               n_b,
                               h_t_f):

    P_t_out = P_t_in * PR                                   # Output stagnation pressure

    h_t_in  = Cp * T_t_in                                   # Input stagnation enthalpy
    h_t_out = Cp * T_t_out                                  # Output stagnation enthalpy
    f       = (h_t_out - h_t_in) / (n_b * h_t_f - h_t_out)  # Fuel-to-air ratio

    return P_t_out, T_t_out, h_t_out, f


def turbojet_combustor_performance(state: rcf.State,
                                   system: rcf.System,
                                   settings: rcf.Settings):

    # Get inputs

    T_t_in = state.energy.compressors[-1].outputs.stagnation_temperature
    P_t_in = state.energy.compressors[-1].outputs.stagnation_pressure

    T_t_out = state.energy.converters.combustor.outputs.stagnation_temperature

    PR      = system.energy.converters.combustor.pressure_ratio
    n_b     = system.energy.converters.combustor.efficiency

    h_t_f   = system.energy.fuel.specific_enthalpy

    # Call the function
    P_t_out, T_t_out, h_t_out, f = func_combustor_performance(T_t_in, P_t_in, T_t_out, PR, n_b, h_t_f)

    # Set outputs

    cond = state.energy.converters.combustor
    cond.outputs.stagnation_pressure            = P_t_out
    cond.outputs.stagnation_temperature         = T_t_out
    cond.outputs.stagnation_enthalpy            = h_t_out
    cond.fuel_air_ratio                         = f

