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
from src.eden_trace.library import units

# ----------------------------------------------------------------------------------------------------------------------
#  Turbofan thrust
# ----------------------------------------------------------------------------------------------------------------------


def func_thrust_and_power(
    u0,
    P0,
    g, 
    delta_SFC,
    v_fan_nozzle,
    A_fan_nozzle, # Physical exit area, not Area Ratio!
    P_fan_nozzle,
    v_core_nozzle,
    A_core_nozzle, # Physical exit area, not Area Ratio!
    P_core_nozzle,
    fuel_air_ratio,
    mdot_core,
    BPR,
):

    # 1. Calculate mass flows
    mdot_fan = mdot_core * BPR
    mdot_in_core = mdot_core / (1.0 + fuel_air_ratio) # Strip fuel for inlet momentum
    mdot_in_total = mdot_in_core + mdot_fan
    
    # 2. Raw Dimensional Thrust (Gross Thrust - Ram Drag)
    # Core
    gross_thrust_core = (mdot_core * v_core_nozzle) + (P_core_nozzle - P0) * A_core_nozzle
    ram_drag_core = mdot_in_core * u0
    F_core = gross_thrust_core - ram_drag_core
    
    # Fan
    gross_thrust_fan = (mdot_fan * v_fan_nozzle) + (P_fan_nozzle - P0) * A_fan_nozzle
    ram_drag_fan = mdot_fan * u0
    F_fan = gross_thrust_fan - ram_drag_fan
    
    # 3. Total Actual Thrust (in Newtons)
    F_actual = F_core + F_fan
    
    # 4. Power and Efficiency (Calculated directly from F_actual to avoid singularities)
    p = F_actual * u0
    
    mdot_fuel = mdot_in_core * fuel_air_ratio
    
    # Protect against divide-by-zero if fuel flow is exactly 0.0
    safe_mdot_fuel = jnp.maximum(mdot_fuel, 1e-9)
    safe_F_actual = jnp.maximum(F_actual, 1e-9)
    
    I_sp = F_actual / (safe_mdot_fuel * g)
    TSFC = (safe_mdot_fuel / safe_F_actual) * (1.0 - delta_SFC) * 3600.0
    
    # Fuel flow in kg/hr
    ff = mdot_fuel * 3600.0 
    
    # Back-calculate non-dimensional terms only if downstream code strictly requires it
    # (Though it's highly recommended to purge F_sp_nondim entirely if possible)
    specific_thrust_core = F_actual / mdot_core

    return F_actual, specific_thrust_core, I_sp, TSFC, p, ff
