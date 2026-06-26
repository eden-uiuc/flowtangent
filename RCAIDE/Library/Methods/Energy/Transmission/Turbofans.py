# RCAIDE/Library/Methods/Propulsors/Turbofan/thrust.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: Apr 2025, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

# package imports

import jax.numpy as jnp

# RCAIDE imports


# ----------------------------------------------------------------------------------------------------------------------
#  Turbofan thrust
# ----------------------------------------------------------------------------------------------------------------------


def func_thrust_and_power(
    gamma_0,  # Explicitly the freestream gamma from the atmospheric model
    u0,
    a0,
    M0,
    P0,
    g,  # Standard gravity
    F_ref,
    delta_SFC,
    v_fan_nozzle,
    AR_fan_nozzle,
    P_fan_nozzle,
    v_core_nozzle,
    AR_core_nozzle,
    P_core_nozzle,
    fuel_air_ratio,
    BPR,
):

    f_fan = BPR / (1.0 + BPR)
    f_core = 1.0 / (1.0 + BPR)

    # 1. Fan and Core Components of Non-Dimensional Thrust
    # Normalized by Freestream Dynamic Pressure * Capture Area
    F_fan = f_fan * (gamma_0 * M0**2 * (v_fan_nozzle / u0 - 1.0) + AR_fan_nozzle * (P_fan_nozzle / P0 - 1.0))
    F_core = f_core * (gamma_0 * M0**2 * (v_core_nozzle / u0 - 1.0) + AR_core_nozzle * (P_core_nozzle / P0 - 1.0))
    F_total = F_fan + F_core

    # 2. Specific Thrust (F_sp_nondim is dimensionless)
    F_sp_nondim = 1.0 / (gamma_0 * M0) * F_total

    # Dimensional specific thrust evaluated for the whole engine relative to CORE mass flow
    # Units: N / (kg/s) or (m/s)
    specific_thrust_core = F_sp_nondim * a0 * (1.0 + BPR)

    # 3. Performance Metrics (Corrected to use gravity 'g' instead of 'gamma')
    I_sp = specific_thrust_core / (fuel_air_ratio * g)
    TSFC = fuel_air_ratio / specific_thrust_core * (1.0 - delta_SFC) * 3600.0

    # 4. Engine Sizing and Throttle State
    mdot_core = (F_ref * f_core) / (F_sp_nondim * a0)

    F = specific_thrust_core * mdot_core
    p = F * u0

    # 5. Fuel Flow Rate (Using JAX maximum)
    # TSFC is expected to be mass / (Thrust * hr)
    ff = jnp.maximum(F * TSFC, 0.0) / 3600.0

    return F, F_sp_nondim, I_sp, TSFC, mdot_core, p, ff


def func_sea_level_static_thrust(
    F_ref,
    delta_SFC,
    v_fan_nozzle,
    AR_fan_nozzle,
    P_fan_nozzle,
    v_core_nozzle,
    AR_core_nozzle,
    P_core_nozzle,
    f,  # Fuel-Air Ratio
    alpha,  # Bypass Ratio
    gamma_0,
):

    sls_thrust, _, _, _, _, _, _ = func_thrust_and_power(
        u0=0.0,
        a0=0.0,
        M0=0.01,
        P0=101325.0,
        g=9.81,
        F_ref=F_ref,
        delta_SFC=delta_SFC,
        fuel_air_ratio=f,
        v_fan_nozzle=v_fan_nozzle,
        AR_fan_nozzle=AR_fan_nozzle,
        P_fan_nozzle=P_fan_nozzle,
        v_core_nozzle=v_core_nozzle,
        AR_core_nozzle=AR_core_nozzle,
        P_core_nozzle=P_core_nozzle,
        BPR=alpha,
        gamma_0=gamma_0,
        throttle=1.0,
    )

    return sls_thrust
