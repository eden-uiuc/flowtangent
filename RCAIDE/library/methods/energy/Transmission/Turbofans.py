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
from RCAIDE.library import units

# ----------------------------------------------------------------------------------------------------------------------
#  Turbofan thrust
# ----------------------------------------------------------------------------------------------------------------------


def func_thrust_and_power(
    gamma_0, 
    u0,
    a0,
    M0,
    P0,
    g, 
    delta_SFC,
    v_fan_nozzle,
    AR_fan_nozzle,
    P_fan_nozzle,
    v_core_nozzle,
    AR_core_nozzle,
    P_core_nozzle,
    fuel_air_ratio,
    mdot_core,
    BPR,
):

    f_fan = BPR / (1.0 + BPR)
    f_core = 1.0 / (1.0 + BPR)

    # Specific Thrust calculations remain perfectly valid
    F_fan = f_fan * (gamma_0 * M0**2 * (v_fan_nozzle / u0 - 1.0) + AR_fan_nozzle * (P_fan_nozzle / P0 - 1.0))
    F_core = f_core * (gamma_0 * M0**2 * (v_core_nozzle / u0 - 1.0) + AR_core_nozzle * (P_core_nozzle / P0 - 1.0))
    F_total = F_fan + F_core

    F_sp_nondim = 1.0 / (gamma_0 * M0) * F_total
    specific_thrust_core = F_sp_nondim * a0 * (1.0 + BPR)

    F_actual = specific_thrust_core * mdot_core
    
    p = F_actual * u0

    I_sp = specific_thrust_core / (fuel_air_ratio * g)
    TSFC = fuel_air_ratio / specific_thrust_core * (1.0 - delta_SFC) * 3600.0
    
    ff = jnp.maximum(F_actual * TSFC, 0.0) / units.hr

    return F_actual, F_sp_nondim, I_sp, TSFC, p, ff
