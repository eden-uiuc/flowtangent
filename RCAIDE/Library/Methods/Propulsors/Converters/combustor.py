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
                               Cp,
                               PR,
                               n_b,
                               h_t_f):

    P_t_out = P_t_in * PR                                   # Output stagnation pressure

    h_t_in  = Cp * T_t_in                                   # Input stagnation enthalpy
    h_t_out = Cp * T_t_out                                  # Output stagnation enthalpy
    f       = (h_t_out - h_t_in) / (n_b * h_t_f - h_t_out)  # Fuel-to-air ratio

    return P_t_out, T_t_out, h_t_out, f


def turbojet_combustor_performance(
    state: rcf.State,
    system: rcf.System,
    settings: rcf.Settings
):

    for idx, jet in system.energy.propulsors:

        comp_outputs = state.energy.propulsors[idx].compressors[-1].outputs
        T_t_in = comp_outputs.stagnation_temperature
        P_t_in = comp_outputs.stagnation_pressure

        T_t_out = jet.turbines[0].design_intake_temperature

        Cp      = state.freestream.Cp

        combustor   = jet.cconverters.ombustor
        PR          = combustor.pressure_ratio
        n_b         = combustor.efficiency

        h_t_f       = jet.fuel.specific_energy

        # Call the function
        P_t_out, T_t_out, h_t_out, f = func_combustor_performance(T_t_in, P_t_in, T_t_out, Cp, PR, n_b, h_t_f)

        # Set Input State
        combustor_state = state.energy.propulsors[idx].combustor

        inputs = combustor_state.inputs
        inputs.freestream_Cp                            = Cp
        inputs.stagnation_temperature                   = T_t_in
        inputs.stagnation_pressure                      = P_t_in

        # Set Output State

        outputs = combustor_state.outputs
        outputs.stagnation_pressure            = P_t_out
        outputs.stagnation_temperature         = T_t_out
        outputs.stagnation_enthalpy            = h_t_out
        outputs.fuel_air_ratio                 = f

    return state, system, settings

