# Trace/Library/Methods/Propulsors/Converters/turbine.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: Mar, 2025, Trace Team

# ----------------------------------------------------------------------------------------------------------------------
# IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import eden_trace.framework as rcf

# ----------------------------------------------------------------------------------------------------------------------
# turbine
# ----------------------------------------------------------------------------------------------------------------------


# def func_turbine_performance(
#     gas,  # The BurnedGas mixture
#     FAR,  # Fuel-to-air ratio from the combustor
#     input_work,  # Work required by the compressor (per kg of core air)
#     n_mech,  # Mechanical efficiency of the shaft
#     n_flow,  # Polytropic flow efficiency
#     T_t,
#     P_t,
# ):
#     # Calculate the target specific enthalpy drop across the turbine
#     # Mass flow through the turbine is (1 + f) times the compressor flow
#     d_h_t = -1.0 / (1.0 + FAR) * (input_work / n_mech)

#     # Set the target exit enthalpy
#     h_t_in = gas.compute_enthalpy(T_t)
#     h_t_out = h_t_in + d_h_t

#     # Newton-Raphson Root Solver for T_t_out
#     # Initial guess using the inlet Cp
#     Cp_in = gas.compute_Cp(T_t)
#     T_t_out = T_t + (d_h_t / Cp_in)

#     for _ in range(4):
#         # Evaluate current guess
#         h_guess = gas.compute_enthalpy(T_t_out)
#         Cp_guess = gas.compute_Cp(T_t_out)

#         # Newton step: T_new = T_old - (h_guess - h_target) / (dh/dT)
#         T_t_out = T_t_out - ((h_guess - h_t_out) / Cp_guess)

#     # Total pressure out using polytropic efficiency, averaging gamma across the expansion
#     gamma_in = gas.compute_gamma(T_t)
#     gamma_out = gas.compute_gamma(T_t_out)
#     gamma_avg = 0.5 * (gamma_in + gamma_out)

#     P_t_out = P_t * (T_t_out / T_t) ** (gamma_avg / ((gamma_avg - 1.0) * n_flow))

#     return T_t_out, P_t_out, h_t_out


def func_turbine_performance(
    gas,  # The BurnedGas mixture
    FAR,  # Fuel-to-air ratio from the combustor
    PR,   # Pressure Ratio (guessed by global solver or map)
    n_isn,  # Isentropic efficiency (from the TurbineMap or PyCycle)
    n_mech, # Mechanical work transmission efficiency
    T_t,
    P_t,
):
    # Target exit pressure based on the given PR
    P_t_out = P_t / PR

    # 1. Find the IDEAL exit temperature (100% isentropic expansion)
    # We use a quick 3-step fixed-point iteration to get a highly accurate average gamma 
    # across the massive temperature drop of the turbine.
    gamma_guess = gas.compute_gamma(T_t)
    T_t_out_ideal = T_t * (1.0 / PR) ** ((gamma_guess - 1.0) / gamma_guess)
    
    for _ in range(3):
        gamma_out = gas.compute_gamma(T_t_out_ideal)
        gamma_avg = 0.5 * (gamma_guess + gamma_out)
        T_t_out_ideal = T_t * (1.0 / PR) ** ((gamma_avg - 1.0) / gamma_avg)

    # 2. Apply ISENTROPIC efficiency to find the ACTUAL exit temperature
    # A real turbine extracts less energy, so the temperature drop is smaller than ideal.
    T_t_out = T_t - (T_t - T_t_out_ideal) * n_isn

    # 3. Calculate actual work extracted per kg of core air
    h_t_in = gas.compute_enthalpy(T_t)
    h_t_out = gas.compute_enthalpy(T_t_out)

    # Turbine mass flow is higher than compressor due to added fuel
    # Work will be a negative value (energy leaving the fluid)
    work = (1.0 + FAR) * (h_t_out - h_t_in) * n_mech

    return T_t_out, P_t_out, h_t_out, work
