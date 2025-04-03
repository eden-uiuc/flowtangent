# RCAIDE/Library/Methods/Propulsors/Converters/turbine.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: Mar, 2025, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
# IMPORT 
# ----------------------------------------------------------------------------------------------------------------------

import RCAIDE.Framework as rcf

# ----------------------------------------------------------------------------------------------------------------------
# turbine
# ----------------------------------------------------------------------------------------------------------------------


def func_turbine_performance(
        g,
        Cp,
        f,
        a,
        w_c,
        w_s,
        w_f,
        n_m,
        T_t,
        P_t,
        n_p,
):

        d_h_t = -1 / (1 + f) * (w_c + w_s + (a * w_f)) / n_m            # Total enthalpy drop across the turbine

        T_t_out = T_t + d_h_t / Cp                                      # Total temperature out
        P_t_out = P_t * (T_t_out / T_t) ** (g / ((g - 1) * n_p))        # Total pressure out

        h_t_out = Cp * T_t_out                                          # Total enthalpy out

        return T_t_out, P_t_out, h_t_out


def turbine_performance(
    state: rcf.State,
    system: rcf.System,
    settings: rcf.Settings
):

    # Get inputs

    g       = state.freestream.gamma
    Cp      = state.freestream.Cp

    f       = state.energy.converters.combustor.outputs.fuel_air_ratio

    w_s     = state.energy.converters.shaft_offtake.outputs.work
    w_f     = state.energy.converters.fan.outputs.work

    T_t     = state.energy.converters.combustor.outputs.stagnation_temperature
    P_t     = state.energy.converters.combustor.outputs.stagnation_pressure

    for idx, turb in enumerate(system.energy.converters.turbines)[:-1]:

        w_c     = state.energy.converters.compressors[-(idx + 1)].outputs.work

        n_m     = turb.mechanical_efficiency
        n_p     = turb.polytropic_efficiency

        # Call Functions
        T_t_out, P_t_out, h_t_out = func_turbine_performance(g, Cp, f, 0., w_c, w_s, w_f, n_m, T_t, P_t, n_p)

        turb_state = state.energy.converters.turbines[idx]

        # Set Input State
        inputs = turb_state.inputs
        inputs.gamma = g
        inputs.Cp = Cp
        inputs.fuel_air_ratio = f
        inputs.bypass_ratio = 0.
        inputs.shaft_work = w_s
        inputs.fan_work = w_f
        inputs.compressor_work = w_c
        inputs.stagnation_temperature = T_t
        inputs.stagnation_pressure = P_t

        # Set Output State
        outputs = turb_state.outputs
        outputs.stagnation_temperature = T_t_out
        outputs.stagnation_pressure = P_t_out
        outputs.stagnation_enthalpy = h_t_out

        T_t = T_t_out
        P_t = P_t_out

    # Run final turbine with fan bypass
    a = system.energy.bypass_ratio

    w_c = state.energy.converters.compressors[0].outputs.work

    n_m = system.energy.converters.turbines[-1].mechanical_efficiency
    n_p = system.energy.converters.turbines[-1].polytropic_efficiency

    # Call Functions
    T_t_out, P_t_out, h_t_out = func_turbine_performance(g, Cp, f, a, w_c, w_s, w_f, n_m, T_t, P_t, n_p)

    turb_state = state.energy.converters.turbines[-1]

    # Set Input State
    inputs = turb_state.inputs

    inputs.gamma                    = g
    inputs.Cp                       = Cp
    inputs.fuel_air_ratio           = f
    inputs.bypass_ratio             = a
    inputs.shaft_work               = w_s
    inputs.fan_work                 = w_f
    inputs.compressor_work          = w_c
    inputs.stagnation_temperature   = T_t
    inputs.stagnation_pressure      = P_t

    # Set Output State
    outputs = turb_state.outputs

    outputs.stagnation_temperature  = T_t_out
    outputs.stagnation_pressure     = P_t_out
    outputs.stagnation_enthalpy     = h_t_out

    return state, system, settings
