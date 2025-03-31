# RCAIDE/Library/Methods/Propulsors/Converters/compression_nozzle.py
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
#  Compression Nozzle Functional Methods
# ----------------------------------------------------------------------------------------------------------------------

def func_compression_nozzle_performance(T_t,
                                        P_t,
                                        P0,
                                        M0,
                                        Cp,
                                        g,
                                        PR,
                                        n_r,
                                        n_p):

    # Isentropic Outputs

    P_t_out = np.max(P_t * PR * n_r, P0)                   # Output stagnation pressure, minimum is freestream pressure
    T_t_out  = T_t * (PR * n_r) ** ((g - 1.) / (g * n_p))  # Output stagnation temperature

    M_out   = np.sqrt((((P_t_out / P0) ** ((g - 1.) / g)) - 1.) * 2. / (g - 1.))  # Output Mach number
    T_out   = T_t_out / (1. + (g - 1.) / 2. * M_out ** 2)                         # Output static temperature

    # Normal Shock Outputs

    ns_M    = np.sqrt((1. + (g - 1.) / 2. * M0 ** 2.) / (g * M0 ** 2 - (g - 1.) / 2.))
    ns_T    = T_t_out / (1. + (g - 1.) / 2 * ns_M ** 2)
    ns_P_t  = (PR *
               P_t *
               ((((g + 1.) * (M0 ** 2.)) / ((g - 1.) * M0 ** 2. + 2.)) ** (g / (g - 1.))) *
               ((g + 1.) / (2. * g * M0 ** 2.-(g - 1.))) ** (1./(g - 1.))
               )

    # Combine Outputs

    P_t_out.at(M0 > 1.0).set(ns_P_t)
    T_out.at(M0 > 1.0).set(ns_T)
    M_out.at(M0 > 1.0).set(ns_M)

    h_out   = Cp * T_out                        # Output static enthalpy
    h_t_out = Cp * T_t_out                      # Output stagnation enthalpy
    u_out   = np.sqrt(2. * (h_t_out - h_out))   # Output velocity

    return M_out, u_out, P_t_out, T_t_out, T_out, h_t_out, h_out


def compression_nozzle_performance(State: rcf.State,
                                   System: rcf.System,
                                   Settings: rcf.Settings):

    # Get inputs

    T_t = State.conditions.freestream.stagnation_temperature
    P_t = State.conditions.freestream.stagnation_pressure
    P0  = State.conditions.freestream.pressure
    M0  = State.conditions.freestream.mach_number
    Cp  = State.conditions.freestream.Cp
    g   = State.conditions.freestream.gamma

    PR  = System.energy.converters.compression_nozzle.pressure_ratio
    n_r = System.energy.converters.compression_nozzle.recovery_efficiency
    n_p = System.energy.converters.compression_nozzle.polytropic_efficiency

    # Call function
    M_out, u_out, P_t_out, T_t_out, T_out, h_t_out, h_out = func_compression_nozzle_performance(T_t,
                                                                                                P_t,
                                                                                                P0,
                                                                                                M0,
                                                                                                Cp,
                                                                                                g,
                                                                                                PR,
                                                                                                n_r,
                                                                                                n_p)

    # Set outputs

    State.energy.converters.compression_nozzle.outputs.mach_number             = M_out
    State.energy.converters.compression_nozzle.outputs.velocity                = u_out

    State.energy.converters.compression_nozzle.outputs.stagnation_pressure     = P_t_out

    State.energy.converters.compression_nozzle.outputs.stagnation_temperature  = T_t_out
    State.energy.converters.compression_nozzle.outputs.static_temperature      = T_out

    State.energy.converters.compression_nozzle.outputs.stagnation_enthalpy     = h_t_out
    State.energy.converters.compression_nozzle.outputs.static_enthalpy         = h_out



