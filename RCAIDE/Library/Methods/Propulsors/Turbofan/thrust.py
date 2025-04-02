# RCAIDE/Library/Methods/Propulsors/Turbofan/thrust.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: Apr 2025, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from dataclasses import dataclass, field

# package imports
import numpy as np

# RCAIDE imports
import RCAIDE.Framework as rcf
from RCAIDE.Framework.Core import Units as U


# ----------------------------------------------------------------------------------------------------------------------
#  Turbofan thrust
# ----------------------------------------------------------------------------------------------------------------------

def func_thrust(
        g,
        u0,
        a0,
        M0,
        P0,
        G0,
        T_ref,
        T_t_ref,
        P_ref,
        P_t_ref,
        mdhc,
        delta_SFC,
        f,
        v_fan_nozzle,
        AR_fan_nozzle,
        P_fan_nozzle,
        v_core_nozzle,
        AR_core_nozzle,
        P_core_nozzle,
        alpha,
        throttle
):

    f_fan = alpha / (1 + alpha)
    f_core = 1 / (1 + alpha)

    # Fan and Core Components of Non-Dimensional Thrust
    F_fan   = f_fan  * (g * M0 ** 2 * (v_fan_nozzle  / u0 - 1) + AR_fan_nozzle  * (P_fan_nozzle  / P0 - 1))
    F_core  = f_core * (g * M0 ** 2 * (v_core_nozzle / u0 - 1) + AR_core_nozzle * (P_core_nozzle / P0 - 1))
    F_total = F_fan + F_core                                                # Total Non-Dimensional Thrust
    F_sp    = 1 / (g * M0) * F_total                                        # Specific Thrust
    I_sp    = F_sp * a0 * (1 + alpha) / (f * g)                             # Specific Impulse
    TSFC    = f * g / (F_sp * a0 * (1 + alpha)) * (1 - delta_SFC) * U.hour  # Thrust-Specific Fuel Consumption (TSFC)
    mdot_c  = mdhc * np.sqrt(T_ref / T_t_ref) * (P_t_ref / P_ref)           # Core flow rate
    F       = F_sp * a0 * (1 + alpha) * mdot_c * throttle                   # Dimensional Thrust
    p       = F * u0                                                        # Power
    ff      = np.maximum(F * TSFC / G0, 0.) * 1 / U.hour                    # Fuel Flow Rate

    return F, F_sp, I_sp, TSFC, mdot_c, p, ff


def thrust(
        state: rcf.State,
        system: rcf.System,
        settings: rcf.Settings
):

    # Get inputs

    fs  = state.freestream
    g   = fs.gamma
    u0  = fs.u
    a0  = fs.speed_of_sound
    M0  = fs.mach_number
    P0  = fs.pressure
    G0  = fs.gravity

    tf = system.energy
    T_ref               = tf.reference_temperature
    T_t_ref             = tf.reference_total_temperature
    P_ref               = tf.reference_pressure
    P_t_ref             = tf.reference_total_pressure
    mdhc                = tf.compressor_nondimensional_massflow
    delta_SFC           = tf.specific_fuel_consumption_adjustment
    alpha               = tf.bypass_ratio

    tf_state = state.energy
    f                   = tf_state.compressor.fuel_air_ratio
    throttle            = tf_state.throttle

    fn_out = tf_state.fan_nozzle.outputs
    v_fan_nozzle        = fn_out.velocity
    AR_fan_nozzle       = fn_out.area_ratio
    P_fan_nozzle        = fn_out.pressure

    cn_out = tf_state.core_nozzle.outputs
    v_core_nozzle       = cn_out.velocity
    AR_core_nozzle      = cn_out.area_ratio
    P_core_nozzle       = cn_out.pressure

    # Call function
    F, F_sp, I_sp, TSFC, mdot_c, p, ff = func_thrust(
        g,
        u0,
        a0,
        M0,
        P0,
        G0,
        T_ref,
        T_t_ref,
        P_ref,
        P_t_ref,
        mdhc,
        delta_SFC,
        f,
        v_fan_nozzle,
        AR_fan_nozzle,
        P_fan_nozzle,
        v_core_nozzle,
        AR_core_nozzle,
        P_core_nozzle,
        alpha,
        throttle
    )

    # Update Energy State

    tf_state.thrust                             = F
    tf_state.non_dimensional_thrust             = F_sp
    tf_state.specific_impulse                   = I_sp
    tf_state.thrust_specific_fuel_consumption   = TSFC
    tf_state.core_mass_flow_rate                = mdot_c
    tf_state.power                              = p
    tf_state.fuel_flow_rate                     = ff
                   
    return state, system, settings
