# $NAME.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: Mar 2025, Trace Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.eden_trace.framework import Aircraft, Settings, State

# package imports
import jax.numpy as jnp

# Trace imports
from src.eden_trace.library.gases import CO2, H2O, O2, IdealGas
from src.eden_trace.library.methods.energy.transmission import Rayleigh, fM

# ----------------------------------------------------------------------------------------------------------------------
#  Combustor Performance
# ----------------------------------------------------------------------------------------------------------------------


def func_combustor_design(
    gas: IdealGas,
    T_t: jnp.ndarray,
    P_t: jnp.ndarray,
    T_t_out: jnp.ndarray,
    mdot_in: jnp.ndarray,
    LHV: jnp.ndarray | float,
    h_t_f: jnp.ndarray | float,
    PR: jnp.ndarray | float,
    n_b: jnp.ndarray | float,
):
    
    h_t_in = gas.compute_enthalpy(T_t)
    P_t_out = P_t * PR

    # Target exit enthalpy based on the commanded exit temperature
    h_t_out = gas.compute_enthalpy(T_t_out)

    # Simple First-Law FAR calculation using LHV
    numerator = h_t_out - h_t_in
    denominator = (LHV * n_b) + h_t_f - h_t_out
    
    FAR = numerator / denominator
    
    # Calculate explicit mass flow additions
    mdot_fuel = mdot_in * FAR
    mdot_out = mdot_in + mdot_fuel

    return P_t_out, h_t_out, jnp.atleast_2d(FAR), mdot_out

def func_combustor_performance(
    gas: IdealGas,            # IdealGas or BurnedGas model
    T_t: jnp.ndarray,
    P_t: jnp.ndarray,
    mdot_in: jnp.ndarray,
    FAR: jnp.ndarray,
    LHV: jnp.ndarray | float,
    h_t_f: jnp.ndarray | float,
    PR: jnp.ndarray | float,
    n_b: jnp.ndarray | float,
):
    # 1. Pressure and Mass Flow additions
    P_t_out = P_t * PR
    mdot_fuel = mdot_in * FAR
    mdot_out = mdot_in + mdot_fuel

    # 2. Forward Energy Balance to find Exit Enthalpy
    h_t_in = gas.compute_enthalpy(T_t)
    
    # Derivation: m_in*h_in + m_fuel*h_fuel + m_fuel*LHV*n_b = m_out*h_out
    h_t_out = (h_t_in + FAR * (LHV * n_b + h_t_f)) / (1.0 + FAR)

    # 3. Newton-Raphson to invert Enthalpy back to Temperature
    # Initial guess using inlet Cp to get us in the ballpark
    Cp_guess = gas.compute_Cp(T_t)
    T_t_out = T_t + (h_t_out - h_t_in) / Cp_guess

    # 5 steps is more than enough for NASA polynomials to converge perfectly
    for _ in range(5):
        h_current = gas.compute_enthalpy(T_t_out)
        Cp_current = gas.compute_Cp(T_t_out)
        
        error = h_current - h_t_out
        
        # True Newton Step: x_new = x_old - f(x)/f'(x)
        T_t_out = T_t_out - (error / Cp_current)

    return P_t_out, T_t_out, h_t_out, mdot_out


# ----------------------------------------------------------------------------------------------------------------------
# Rayleigh Line Flow
# ----------------------------------------------------------------------------------------------------------------------


def func_rayleigh_line_flow(
    g,
    Cp,
    Tt_in,
    Pt_in,
    M0,
    Tt_4,
    eta_b,
    ht_f,
    AR,
):

    # Isentropic deceleration through divergent nozzle
    M1 = jnp.atleast_2d(fM(AR, M0[:, 0], g[:, 0])).T

    # Max stagnation temperature to thermally choke flow
    Tt_out_Rayleigh = Tt_in * (1.0 + g * M1**2) ** 2.0 / ((2.0 * (1.0 + g) * M1**2) * (1.0 + (g - 1.0) / 2.0 * M1**2))

    # Limit Tt_out
    Tt_out = jnp.ones_like(Tt_out_Rayleigh) * Tt_4
    Tt_out = jnp.minimum(Tt_out, Tt_out_Rayleigh)

    # Rayleigh calculations
    M_out = jnp.zeros_like(Pt_in)
    Pt_R = jnp.zeros_like(Pt_in)
    M_out[:, 0], Pt_R[:, 0] = Rayleigh(g[:, 0], M1[:, 0], Tt_out[:, 0] / Tt_in[:, 0])
    Pt_out = Pt_R * Pt_in

    # Stagnation enthalpies from stagnation temperatures
    ht_in = Tt_in * Cp
    ht_out = Tt_out * Cp

    # Fuel air ratio
    f = (ht_out - ht_in) / (eta_b * ht_f - ht_out)

    return Tt_out, Pt_out, ht_out, M_out, f


def rayleigh_line_flow(state: State, system: Aircraft, settings: Settings):

    for l_idx, line in system.energy.lines:
        for p_idx, propulsor in enumerate(line.propulsors):
            combustor = propulsor.propulsors.combustor
            combustor_state = state.energy.lines[l_idx].propulsors[p_idx].combustor

            Tt_in = combustor_state.inputs.stagnation_temperature
            Pt_in = combustor_state.inputs.stagnation_pressure
            M0 = combustor_state.inputs.mach_number

            eta_b = combustor.efficiency
            AR = combustor.area_ratio

            ht_f = propulsor.fuel.specific_energy
            Tt_4 = propulsor.propulsors.turbines[0].design_intake_temperature

            g = state.freestream.gamma
            Cp = state.freestream.Cp

            Tt_out, Pt_out, ht_out, M_out, f = func_rayleigh_line_flow(g, Cp, Tt_in, Pt_in, M0, Tt_4, eta_b, ht_f, AR)

            combustor_state.outputs.stagnation_temperature = Tt_out
            combustor_state.outputs.stagnation_pressure = Pt_out
            combustor_state.outputs.stagnation_enthalpy = ht_out
            combustor_state.outputs.mach_number = M_out
            combustor_state.outputs.fuel_to_air_ratio = f

    return state, system, settings
