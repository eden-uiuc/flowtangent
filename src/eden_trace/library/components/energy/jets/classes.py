# Trace/Library/Components/Energy/Propulsors.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: May 2025, Trace Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING, Callable
if TYPE_CHECKING:
    from eden_trace.framework import Settings, State, System
    from eden_trace.framework.conditions.energy import TurbojetNetworkConditions

import csv
import json
import os

# package imports
import equinox as eqx
import jax
import jax.numpy as jnp

import eden_trace.utils as tu

# Trace imports
from eden_trace.utils import init_field, register, get_trace_root

from eden_trace.library import units

from .maps import data as map_data
from ..maps.classes import CompressorMap, TurbineMap
from eden_trace.library.components.energy.nodes import GraphInput, GraphNode, Splitter, FlowNode, FlowDesign, BleedFlow
from eden_trace.library.gases import Air, BurnedJetA, IdealGas
from eden_trace.library.propellants import JetA, Propellant

# ----------------------------------------------------------------------------------------------------------------------
# Turbojet Components
# ----------------------------------------------------------------------------------------------------------------------

# Inlet ------------------------------------------------------------------------

def _inlet_stagnation(gas, T_t, P_t, M0, PR, P_rec):
    gamma = gas.compute_gamma(T_t)
    
    # Base isentropic recovery
    P_t_out_sub = P_t * PR * P_rec
    
    # Normal Shock Recovery
    ns_P_t = (
        PR * P_t
        * ((((gamma + 1.0) * (M0**2.0)) / ((gamma - 1.0) * M0**2.0 + 2.0)) ** (gamma / (gamma - 1.0)))
        * ((gamma + 1.0) / (2.0 * gamma * M0**2.0 - (gamma - 1.0))) ** (1.0 / (gamma - 1.0))
    )
    
    T_t_out = T_t # Adiabatic
    P_t_out = jnp.where(M0 > 1.0, ns_P_t, P_t_out_sub)
    
    return T_t_out, P_t_out

def _inlet_performance(gas, T_t, P_t, M0, PR, P_rec, mdot, A_exit):
    
    T_t_out, P_t_out = _inlet_stagnation(gas, T_t, P_t, M0, PR, P_rec)
    gamma = gas.compute_gamma(T_t)
    R = gas.R_specific
    
    # The non-dimensional mass flow parameter we need to match
    Q = (mdot * jnp.sqrt(R * T_t_out)) / (P_t_out * A_exit * jnp.sqrt(gamma))
    
    # Newton loop to find subsonic Mach number
    def step(M_out, _):
        term = 1.0 + (gamma - 1.0) / 2.0 * M_out**2
        power = - (gamma + 1.0) / (2.0 * (gamma - 1.0))
        
        f = M_out * (term ** power) - Q
        
        # Derivative df/dM
        df_dM = (term ** power) + M_out * power * (term ** (power - 1.0)) * (gamma - 1.0) * M_out
        
        M_out = jnp.clip(M_out - f / df_dM, 1e-6, 0.99)
        
        return M_out, None
    
    M_out, _ = jax.lax.scan(step, 0.5 * jnp.ones_like(gamma), jnp.arange(5))
        
    # Calculate static properties using the solved Mach
    T_out, P_out, h_t_out, h_out, u_out, M_out = FlowNode.statics(gas, T_t_out, P_t_out, mdot, A_exit)
    
    return M_out, u_out, P_t_out, T_t_out, P_out, T_out, h_t_out, h_out

@register
class Inlet(FlowNode):
    tag: str = init_field("Inlet", static=True)

    @tu.inputs(
        "state.freestream.stagnation_temperature",
        "state.freestream.stagnation_pressure",
        "state.freestream.pressure",
        "state.freestream.mach_number",
        "state.freestream.Cp",
        "state.freestream.gamma",
        "system.energy.nodes[InletNozzle].pressure_ratio",
        "system.energy.nodes[InletNozzle].pressure_recovery",
        "system.energy.nodes[InletNozzle].design_parameters.eff.flow",
    )
    @tu.outputs(
        "state.energy.nodes[InletNozzle].outputs.flow.mach_number",
        "state.energy.nodes[InletNozzle].outputs.flow.speed",
        "state.energy.nodes[InletNozzle].outputs.flow.stagnation_pressure",
        "state.energy.nodes[InletNozzle].outputs.flow.temperature",
        "state.energy.nodes[InletNozzle].outputs.flow.stagnation_temperature",
        "state.energy.nodes[InletNozzle].outputs.flow.enthalpy",
        "state.energy.nodes[InletNozzle].outputs.flow.stagnation_enthalpy",
    )
    def transmit(self, state: State, system: Aircraft, settings: Settings):  # type: ignore

        updated_system = system
        fs = state.freestream
        
        network_state: TurbojetNetworkConditions = state.energy

        gas = fs.atmosphere.fluid
        T_t = fs.stagnation_temperature
        P_t = fs.stagnation_pressure
        M0  = fs.mach_number

        PR    = jnp.atleast_2d(self.design_parameters.pressure_ratio)
        P_rec = jnp.atleast_2d(self.design_parameters.pressure_recovery)
        M_out = jnp.atleast_2d(self.design_parameters.exit_mach_number)
        
        T_t_out, P_t_out = self.stagnation(gas, T_t, P_t, M0, PR, P_rec)

        if settings.analysis.energy.design_mode:

            A_out, u_out, P_out, T_out, h_t_out, h_out = self.kinematics(
                gas=gas,
                T_t_out=T_t_out,
                P_t_out=P_t_out,
                M_out=M_out,
                mdot=network_state.mass_flow_rate
            )
        
            updated_design_paramters = eqx.tree_at(
                lambda d: d.A_exit,
                self.design_parameters,
                A_out.squeeze()
            )
            updated_system = eqx.tree_at(
                lambda s: s.energy.nodes[self.network_ID].design_parameters,
                updated_system,
                updated_design_paramters
            )

        else:
            T_out, P_out, h_t_out, h_out, u_out, M_out = self.statics(
                gas=fs.atmosphere.fluid,
                T_t=fs.stagnation_temperature,
                P_t=fs.stagnation_pressure,
                mdot=jnp.atleast_2d(network_state.mass_flow_rate),
                area=self.design_parameters.A_exit,
            )

        outputs = state.energy.nodes[self.network_ID].outputs.flow

        outputs = eqx.tree_at(lambda o: o.mass_flow_rate, outputs, network_state.mass_flow_rate)
        outputs = eqx.tree_at(lambda o: o.mach_number, outputs, M_out)
        outputs = eqx.tree_at(lambda o: o.speed, outputs, u_out)
        outputs = eqx.tree_at(lambda o: o.stagnation_pressure, outputs, P_t_out)
        outputs = eqx.tree_at(lambda o: o.stagnation_temperature, outputs, T_t_out)
        outputs = eqx.tree_at(lambda o: o.temperature, outputs, T_out)
        outputs = eqx.tree_at(lambda o: o.pressure, outputs, P_out)
        outputs = eqx.tree_at(lambda o: o.stagnation_enthalpy, outputs, h_t_out)
        outputs = eqx.tree_at(lambda o: o.enthalpy, outputs, h_out)

        updated_state = eqx.tree_at(lambda s: s.energy.nodes[self.network_ID].outputs.flow, state, outputs)

        return updated_state, updated_system, settings


# Compressor -------------------------------------------------------------------

def _alpha_c(Nc, Nc_design):
    """
    Schedules alpha (inlet guide vane angle in degrees) according to rotation speed.
    """
    return jnp.where(Nc_design > 0.0, jnp.maximum(0.0, 90.0 - (Nc / Nc_design) * 90.0), jnp.zeros_like(Nc))

def _compressor_performance(
        gas,
        T_t,
        P_t,
        PR,
        n_isn,
):
    """
    Computes power and output thermal quantities for a compressor.
    """

    T_t_out, P_t_out = FlowNode.stagnation(gas, T_t, P_t, PR, 1.0 / n_isn)

    # 3. Calculate work using absolute enthalpies
    h_t = gas.compute_enthalpy(T_t)
    h_t_out = gas.compute_enthalpy(T_t_out)
    d_power = h_t_out - h_t

    return d_power, P_t_out, T_t_out, h_t_out

@register
class Compressor(FlowNode):
    tag: str = init_field("Compressor", static=True)

    inputs: tuple | GraphInput =init_field(GraphInput("flow", "Inlet"), static=True)

    map: CompressorMap = init_field(map_data.AXI5)

    alpha_schedule: Callable = init_field(_alpha_c, as_value=True, static=True)

    def __post_init__(self):
        if not isinstance(self.map, CompressorMap):
            raise TypeError(f"'{self.tag}' requires a CompressorMap, got {type(self.map).__name__}")
        super(Compressor, self).__post_init__()

    @tu.inputs(
        "state.freestream.Cp",
        "state.freestream.gamma",
        "state.energy.nodes[Compressor_flow_inputs].outputs.flow.stagnation_temperature",
        "state.energy.nodes[Compressor_flow_inputs].outputs.flow.stagnation_pressure",
        "system.energy.nodes[Compressor].pressure_ratio",
        "system.energy.nodes[Compressor].design_parameters.eff.flow",
    )
    @tu.outputs(
        "state.energy.nodes[Compressor].flow.stagnation_temperature"
        "state.energy.nodes[Compressor].flow.stagnation_pressure",
        "state.energy.nodes[Compressor].flow.stagnation_enthalpy",
        "state.energy.nodes[Compressor].mechanical.work",
    )
    def transmit(self, state: State, system: System, settings: Settings):

        updated_system = system

        design_mode = settings.analysis.energy.design_mode
        network_state: TurbojetNetworkConditions = state.energy
        
        gas, T_t, P_t, W_in, _, _ = self.mix_inputs(state)
        W_out = W_in * (1.0 - self.bleed_MFR_frac(state))

        theta_c = T_t / 288.15
        delta_c = P_t / 101325.0
        
        if design_mode:
            M_out   = jnp.atleast_2d(self.design_parameters.exit_mach_number)
            PR      = jnp.atleast_2d(self.design_parameters.pressure_ratio)
            n_isn   = jnp.atleast_2d(self.design_parameters.eff.flow)
            N_des   = jnp.atleast_2d(self.design_parameters.rotation_speed)
            
            Nc_des  = N_des / jnp.sqrt(theta_c)

            W   = jnp.atleast_2d(network_state.mass_flow_rate)
            Wc_tgt = W * jnp.sqrt(theta_c) / delta_c
            
            PR_map, Wc_map, eff_map = self.map.evaluate(
                alpha=self.map.alpha_des,
                Nc=self.map.Nc_des,
                Rline=self.map.Rline_des
            )
            
            s_Wc =  (Wc_tgt / Wc_map).squeeze()
            s_PR = (PR - 1.0)/(PR_map - 1.0)
            s_eff = n_isn / eff_map
            s_Nc = (Nc_des/self.map.Nc_des).squeeze()

            T_t_out, P_t_out = self.stagnation(gas, T_t, P_t, PR, 1.0 / n_isn)

            A_out, u_out, P_out, T_out, h_t_out, h_out = self.kinematics(
                gas=gas,
                T_t_out=T_t_out,
                P_t_out=P_t_out,
                M_out=M_out,
                mdot=network_state.mass_flow_rate
            )

            power = h_t_out - jnp.atleast_2d(gas.compute_enthalpy(T_t))

            updated_design_paramters = eqx.tree_at(
                lambda d: d.A_exit,
                self.design_parameters,
                A_out.squeeze()
            )

            updated_map = eqx.tree_at(
                lambda m: (m.s_Wc, m.s_PR, m.s_eff, m.s_Nc),
                self.map,
                (s_Wc, s_PR, s_eff, s_Nc)
            )

            updated_system = eqx.tree_at(
                lambda s: (
                    s.energy.nodes[self.network_ID].design_parameters,
                    s.energy.nodes[self.network_ID].map,
                    s.energy.design_parameters.mass_flow_rate,
                    s.energy.design_parameters.power,
                ),
                    updated_system,
                (
                    updated_design_paramters,
                    updated_map,
                    state.energy.mass_flow_rate,
                    power
                )

            )

        else:
            N      = jnp.atleast_2d(network_state.rotation_speed)
            Nc_des = self.design_parameters.rotation_speed
            Nc     = N / jnp.sqrt(theta_c)
            
            
            alpha   = self.alpha_schedule(Nc, Nc_des)
            
            # Rline drops out of controls in variable nozzle engine
            if network_state.Rline.size == 0:
                Rline = jnp.ones_like(state.energy.mass_flow_rate) * self.map.Rline_des
                # TODO: Shift to Rline scheduling on altitude, Mach number in future
            else:
                Rline = jnp.atleast_2d(network_state.Rline)

            PR, Wc, n_isn = self.map.evaluate(alpha, Nc, Rline)
            W = Wc * delta_c / jnp.sqrt(theta_c)

            power, P_t_out, T_t_out, h_t_out = _compressor_performance(
                gas=gas,
                T_t=T_t,
                P_t=P_t,
                PR=PR,
                n_isn=n_isn,
            )

            T_out, P_out, h_t_out, h_out, u_out, M_out = self.statics(
                gas,
                T_t_out,
                P_t_out,
                W,
                self.design_parameters.A_exit
            )


        outputs = state.energy.nodes[self.network_ID].outputs

        outputs = eqx.tree_at(lambda o: o.mechanical.power, outputs, jnp.atleast_2d(power))

        outputs = eqx.tree_at(lambda o: o.flow.mass_flow_rate, outputs,         jnp.atleast_2d(W_out))
        outputs = eqx.tree_at(lambda o: o.flow.pressure, outputs,               jnp.atleast_2d(P_out))
        outputs = eqx.tree_at(lambda o: o.flow.stagnation_pressure, outputs,    jnp.atleast_2d(P_t_out))
        outputs = eqx.tree_at(lambda o: o.flow.temperature, outputs,            jnp.atleast_2d(T_out))
        outputs = eqx.tree_at(lambda o: o.flow.stagnation_temperature, outputs, jnp.atleast_2d(T_t_out))
        outputs = eqx.tree_at(lambda o: o.flow.enthalpy, outputs,               jnp.atleast_2d(h_out))
        outputs = eqx.tree_at(lambda o: o.flow.stagnation_enthalpy, outputs,    jnp.atleast_2d(h_t_out))
        outputs = eqx.tree_at(lambda o: o.flow.speed, outputs,                  jnp.atleast_2d(u_out))
        outputs = eqx.tree_at(lambda o: o.flow.mach_number, outputs,            jnp.atleast_2d(M_out))

        updated_state = eqx.tree_at(lambda s: s.energy.nodes[self.network_ID].outputs, state, outputs)

        # Residual Update
        W_des = system.energy.design_parameters.mass_flow_rate
        Wc_res = (W - state.energy.mass_flow_rate)/W_des

        updated_state = eqx.tree_at(
            lambda s:s.energy.outputs.residual.Wc,
            updated_state,
            Wc_res
        )

        return updated_state, updated_system, settings

# Combustor --------------------------------------------------------------------

def _combustor_design(
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

def _combustor_performance(
    gas: IdealGas,            # IdealGas or BurnedGas model
    fuel: Propellant,
    T_t: jnp.ndarray,
    P_t: jnp.ndarray,
    mdot_in: jnp.ndarray,
    FAR: jnp.ndarray,
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
    LHV = fuel.specific_energy
    h_t_f = fuel.enthalpy_of_formation
    h_t_out = (h_t_in + FAR * (LHV * n_b + h_t_f)) / (1.0 + FAR)

    # 3. Newton-Raphson to invert Enthalpy back to Temperature
    # Initial guess using inlet Cp to get us in the ballpark
    ox = fuel.oxidized_form(FAR)
    Cp_guess = ox.compute_Cp(T_t)
    T_t_out = T_t + (h_t_out - h_t_in) / Cp_guess

    # 5 steps is more than enough for NASA polynomials to converge perfectly
    def step(T_t_out, _):
        h_current = ox.compute_enthalpy(T_t_out)
        Cp_current = ox.compute_Cp(T_t_out)
        
        error = h_current - h_t_out
        
        # True Newton Step: x_new = x_old - f(x)/f'(x)
        T_t_out = T_t_out - (error / Cp_current)

        return T_t_out, None
    
    T_t_out, _ = jax.lax.scan(step, T_t_out, jnp.arange(5))

    return P_t_out, T_t_out, h_t_out, mdot_out

@register
class Combustor(FlowNode):
    tag: str = init_field("Combustor", static=True)

    inputs: tuple | GraphInput = init_field(GraphInput("flow", "Compressor"), static=True)
    fuel: Propellant = init_field(JetA)

    @tu.inputs(
        "state.freestream.Cp",
        "state.energy.nodes[Combustor_flow_inputs].outputs.flow.stagnation_temperature",
        "state.energy.nodes[Combustor_flow_inputs].outputs.flow.stagnation_pressure",
        "system.energy.nodes[Turbojet].design_paramters.turbine_intake_temperature",
        "system.energy.nodes[Turbojet].fuel.specific_energy",
        "system.energy.nodes[Combustor].pressure_ratio",
        "system.energy.nodes[Combustor].design_parameters.eff.flow",
    )
    @tu.outputs(
        "state.energy.nodes[Combustor].outputs.flow.stagnation_pressure",
        "state.energy.nodes[Combustor].outputs.flow.stagnation_temperature",
        "state.energy.nodes[Combustor].outputs.flow.stagnation_enthalpy",
        "state.energy.nodes[Combustor].outputs.fuel.fuel_air_ratio",
    )
    def transmit(self, state: State, system: System, settings: Settings):
        
        updated_system = system

        gas, T_t, P_t, W_in, _, _ = self.mix_inputs(state)

        LHV=self.fuel.specific_energy
        PR=self.design_parameters.pressure_ratio
        n_b=self.design_parameters.eff.flow

        if settings.analysis.energy.design_mode:
            T_t_out = jnp.atleast_2d(self.design_parameters.output_temperature)

            P_t_out, h_t_out, FAR, mdot_out = _combustor_design(
                gas=gas,
                T_t=T_t,
                P_t=P_t,
                T_t_out=T_t_out,
                mdot_in=W_in,
                LHV=LHV,
                h_t_f=0.0,
                PR=PR,
                n_b=n_b,
            )

            A_out, u_out, P_out, T_out, h_t_out, h_out = self.kinematics(
                gas=self.working_fluid,
                T_t_out=T_t_out,
                P_t_out=P_t_out,
                M_out=self.design_parameters.exit_mach_number,
                mdot=state.energy.mass_flow_rate * (1.0 + FAR)
            )

            updated_design_paramters = eqx.tree_at(
                lambda d: d.A_exit,
                self.design_parameters,
                A_out.squeeze()
            )
            updated_system = eqx.tree_at(
                lambda s: s.energy.nodes[self.network_ID].design_parameters,
                updated_system,
                updated_design_paramters
            )
        
        else:    
            FAR = state.energy.fuel_air_ratio

            P_t_out, T_t_out, h_t_out, mdot_out = _combustor_performance(
                gas=self.working_fluid,
                fuel=self.fuel,
                T_t=T_t,
                P_t=P_t,
                mdot_in=W_in,
                FAR=FAR,
                PR=PR,
                n_b=n_b
            )

        outputs = state.energy.nodes[self.network_ID].outputs.flow

        outputs = eqx.tree_at(lambda o: o.stagnation_pressure, outputs,     jnp.atleast_2d(P_t_out))
        outputs = eqx.tree_at(lambda o: o.stagnation_temperature, outputs,  jnp.atleast_2d(T_t_out))
        outputs = eqx.tree_at(lambda o: o.stagnation_enthalpy, outputs,     jnp.atleast_2d(h_t_out))
        outputs = eqx.tree_at(lambda o: o.mass_flow_rate, outputs,          jnp.atleast_2d(mdot_out))
        outputs = eqx.tree_at(lambda o: o.fuel_air_ratio, outputs,          jnp.atleast_2d(FAR))
        outputs = eqx.tree_at(lambda o: o.fluid, outputs,                   BurnedJetA(FAR))

        updated_state = eqx.tree_at(lambda s: s.energy.nodes[self.network_ID].outputs.flow, state, outputs)

        return updated_state, updated_system, settings

# Turbine ----------------------------------------------------------------------

def _turbine_performance(
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
    T_t_out, P_t_out = FlowNode.stagnation(gas, T_t, P_t, 1.0 / PR, n_isn)

    # 3. Calculate actual work extracted per kg of core air
    h_t_in = gas.compute_enthalpy(T_t)
    h_t_out = gas.compute_enthalpy(T_t_out)

    # Turbine mass flow is higher than compressor due to added fuel
    # Work will be a negative value (energy leaving the fluid)
    power = (1.0 + FAR) * (h_t_out - h_t_in) * n_mech

    return T_t_out, P_t_out, h_t_out, power

@register
class Turbine(FlowNode):
    tag: str = init_field("Turbine", static=True)

    map: TurbineMap = init_field(map_data.LPT2269)

    alpha_schedule: Callable = init_field(lambda Np, Np_des: jnp.full_like(Np, 1.0), as_value=True, static=True)

    inputs: tuple | GraphInput = init_field(
        (
            GraphInput("flow", "Combustor"),
            GraphInput("fuel", "Combustor"),
        ), static=True,
    )

    def __post_init__(self):
        if not isinstance(self.map, TurbineMap):
            raise TypeError(f"'{self.tag}' requires a TurbineMap, got {type(self.map).__name__}")
        super(Turbine, self).__post_init__()

    @tu.inputs(
        "state.freestream.gamma",
        "state.freestream.Cp",
        "state.energy.nodes[Turbine_flow_inputs].outputs.flow.stagnation_temperature",
        "state.energy.nodes[Turbine_flow_inputs].outputs.flow.stagnation_pressure",
        "state.energy.nodes[Turbine_fuel_inputs].outputs.fuel.fuel_air_ratio",
        "state.energy.nodes[Turbine_mechanical_inputs].outputs.mechanical.work",
        "system.energy.nodes[Turbine].design_parameters.eff.mechanical",
        "system.energy.nodes[Turbine].design_parameters.eff.flow",
    )
    @tu.outputs(
        "state.energy.nodes[Turbine].outputs.flow.stagnation_temperature",
        "state.energy.nodes[Turbine].outputs.flow.stagnation_pressure",
        "state.energy.nodes[Turbine].outputs.flow.stagnation_enthalpy",
    )
    def transmit(self, state: State, system: System, settings: Settings):
        
        updated_system = system

        design_mode = settings.analysis.energy.design_mode
        network_state = state.energy
        
        gas, T_t, P_t, W, FAR, _ = self.mix_inputs(state)

        if design_mode:
            if self.tag.lower() == "lpt":
                PR = jnp.atleast_2d(network_state.LPT_PR)
            elif self.tag.lower() == "hpt":
                PR = jnp.atleast_2d(network_state.HPT_PR)
            else:
                PR = jnp.atleast_2d(network_state.turbine_PR)
            
            n_isn = jnp.atleast_1d(self.design_parameters.eff.flow)
            M_out = jnp.atleast_1d(self.design_parameters.exit_mach_number)
            N_des = jnp.atleast_1d(self.design_parameters.rotation_speed)
            
            Np_des = N_des / jnp.sqrt(T_t)

            Wp_tgt = W * jnp.sqrt(T_t) / P_t

            Wp_map, eff_map = self.map.evaluate(
                alpha=self.map.alpha_des,
                Np=self.map.Np_des,
                PR=self.map.PR_des)
            
            s_Wp = (Wp_tgt / Wp_map).squeeze()
            s_PR = ((PR - 1.0)/(self.map.PR_des - 1.0)).squeeze()
            s_eff = (self.design_parameters.eff.flow / eff_map).squeeze()
            s_Np = (Np_des/self.map.Np_des).squeeze()

            # Turbine passes 1 / PR to reflect pressure drop
            safe_PR = jnp.clip(PR, min=1e-5)
            T_t_out, P_t_out = self.stagnation(gas, T_t, P_t, 1.0 / safe_PR, n_isn)
            A_out, u_out, P_out, T_out, h_t_out, h_out = self.kinematics(
                gas=gas,
                T_t_out=T_t_out,
                P_t_out=P_t_out,
                M_out=M_out,
                mdot=network_state.mass_flow_rate
            )

            power = jnp.atleast_2d(h_t_out - gas.compute_enthalpy(T_t))

            updated_map = eqx.tree_at(
                lambda m: (m.s_Wp, m.s_PR, m.s_eff, m.s_Np),
                self.map,
                (s_Wp, s_PR, s_eff, s_Np)
            )

            updated_design = eqx.tree_at(
                lambda d: (
                    d.pressure_ratio,
                    d.A_exit,
                ),
                    self.design_parameters,
                (
                    PR,
                    A_out,
                )
            )

            updated_system = eqx.tree_at(
                lambda s: (
                    s.energy.nodes[self.network_ID].map,
                    s.energy.nodes[self.network_ID].design_parameters,
                ),
                    updated_system,
                (
                    updated_map,
                    updated_design
                )
            )

        else:
            N = jnp.atleast_2d(network_state.rotation_speed)
            Np = N / jnp.sqrt(T_t)
            Np_des = self.design_parameters.rotation_speed

            PR = jnp.atleast_2d(network_state.turbine_PR)
            # PR = jnp.atleast_2d(self.design_parameters.pressure_ratio)
            alpha = self.alpha_schedule(Np, Np_des)

            Wp, n_isn = self.map.evaluate(alpha, Np, PR)
            W = Wp * P_t / jnp.sqrt(T_t) * (1. + FAR)

            T_t_out, P_t_out, h_t_out, power = _turbine_performance(
                gas=gas,
                FAR=FAR,
                PR=PR,
                n_isn=n_isn,
                n_mech=self.design_parameters.eff.mechanical,
                T_t=T_t,
                P_t=P_t,
            )

            T_out, P_out, h_t_out, h_out, u_out, M_out = self.statics(
                gas,
                T_t_out,
                P_t_out,
                W,
                self.design_parameters.A_exit
            )

        # Set Output State
        outputs = state.energy.nodes[self.network_ID].outputs

        outputs = eqx.tree_at(lambda o: o.flow.mass_flow_rate, outputs,         jnp.atleast_2d(W))
        outputs = eqx.tree_at(lambda o: o.flow.pressure, outputs,               jnp.atleast_2d(P_out))
        outputs = eqx.tree_at(lambda o: o.flow.stagnation_pressure, outputs,    jnp.atleast_2d(P_t_out))
        outputs = eqx.tree_at(lambda o: o.flow.temperature, outputs,            jnp.atleast_2d(T_out))
        outputs = eqx.tree_at(lambda o: o.flow.stagnation_temperature, outputs, jnp.atleast_2d(T_t_out))
        outputs = eqx.tree_at(lambda o: o.flow.enthalpy, outputs,               jnp.atleast_2d(h_out))
        outputs = eqx.tree_at(lambda o: o.flow.stagnation_enthalpy, outputs,    jnp.atleast_2d(h_t_out))
        outputs = eqx.tree_at(lambda o: o.flow.speed, outputs,                  jnp.atleast_2d(u_out))
        outputs = eqx.tree_at(lambda o: o.flow.mach_number, outputs,            jnp.atleast_2d(M_out))

        outputs = eqx.tree_at(lambda o: o.mechanical.power, outputs, jnp.atleast_2d(power))

        updated_state = eqx.tree_at(lambda s: s.energy.nodes[self.network_ID].outputs, state, outputs)

        # Residual Update
        W_des = system.energy.design_parameters.mass_flow_rate
        Wp_res = (W / (1. + FAR) - state.energy.mass_flow_rate)/W_des
        updated_state = eqx.tree_at(
            lambda s:s.energy.outputs.residual.Wp,
            updated_state,
            Wp_res
        )

        return updated_state, updated_system, settings

# Nozzle -----------------------------------------------------------------------

def _isentropic_expansion(
    T_t: jnp.ndarray,
    P_t: jnp.ndarray,
    P0: jnp.ndarray,
    gamma: jnp.ndarray,
    PR: jnp.ndarray | float,
    n_r: jnp.ndarray | float,
):

    # Isentropic Outputs
    P_t_out = jnp.maximum(P_t * PR * n_r, P0)  # Output stagnation pressure, minimum is freestream pressure
    T_t_out = T_t  # Output stagnation temperature, adiabatically conserved

    M_out = jnp.sqrt((((P_t_out / P0) ** ((gamma - 1.0) / gamma)) - 1.0) * 2.0 / (gamma - 1.0))  # Output Mach number
    T_out = T_t_out / (1.0 + (gamma - 1.0) / 2.0 * M_out**2)  # Output temperature

    return P_t_out, T_t_out, T_out, M_out

def _nozzle_design(
        gas: IdealGas,
        T_t: jnp.ndarray,
        P_t: jnp.ndarray,
        mdot: jnp.ndarray,
        P0: jnp.ndarray,
        PR: jnp.ndarray | float,
        n_v: jnp.ndarray | float,
):
    # Dynamic gas properties for the exhaust flow
    gamma = gas.compute_gamma(T_t)
    R = gas.R_specific

    P_t_out, T_t_out, T_out, M_isn = _isentropic_expansion(T_t, P_t, P0, gamma, PR, 1.0)

    # Supersonic Expansion / Choking Logic
    critical_PR = (1.0 + (gamma - 1.0) / 2.0) ** (gamma / (gamma - 1.0))
    is_choked = (P_t / P0) >= critical_PR

    M_out = jnp.maximum(M_isn, 0.001)

    # Recalculate static conditions
    P_out = P_t_out / (1.0 + (gamma - 1.0) / 2.0 * M_out**2) ** (gamma / (gamma - 1.0))
    P_out = jnp.where(is_choked, P_out, P0)

    T_out = T_t_out / (1.0 + (gamma - 1.0) / 2.0 * M_out**2)

    # Enthalpy and velocity
    h_t_out = gas.compute_enthalpy(T_t_out)
    h_out = gas.compute_enthalpy(T_out)
    u_out = jnp.sqrt(2.0 * (h_t_out - h_out)) * n_v

    # Exit area
    rho_out = P_out / (R * T_out)
    A_exit = mdot / (rho_out * u_out)

    # Throat area
    T_star = T_t / (1.0 + (gamma - 1.0) / 2.0)
    P_star = P_t / critical_PR
    rho_star = P_star / (R * T_star)
    u_star = jnp.sqrt(gamma * R * T_star)

    A_throat_choked = mdot / (rho_star * u_star)
    A_throat = jnp.where(is_choked, A_throat_choked, A_exit)

    return A_throat, A_exit, M_out, rho_out, u_out, P_out, P_t_out, T_out, T_t_out, h_out, h_t_out

def _nozzle_performance(
        gas: IdealGas,
        T_t: jnp.ndarray,
        P_t: jnp.ndarray,
        P0: jnp.ndarray,
        A_throat: float,
        A_exit: float,
        n_v: jnp.ndarray | float,
    ):

    gamma = gas.compute_gamma(T_t)
    R = gas.R_specific
    AR = A_exit / A_throat

    # Check for choked flow
    critical_PR = (1.0 + (gamma - 1.0) / 2.0) ** (gamma / (gamma - 1.0))
    actual_PR = P_t / P0
    safe_PR = 1.0 + jnp.sqrt((actual_PR - 1.0)**2 + 1e-4)
    choked = actual_PR >= critical_PR

    # Find exit Mach number
    def step(M_exit_sup, _):
        term = (2.0 / (gamma + 1.0)) * (1.0 + (gamma - 1.0) / 2.0 * M_exit_sup**2)
        power = (gamma + 1.0) / (2.0 * (gamma - 1.0))
        AR_calc = (1.0 / M_exit_sup) * (term ** power)
        
        # Analytical derivative: d(A/A*) / dM
        dAR_dM = AR_calc * (M_exit_sup**2 - 1.0) / (M_exit_sup * (1.0 + (gamma - 1.0) / 2.0 * M_exit_sup**2))
        
        # Newton step
        M_exit_sup = jnp.maximum(M_exit_sup - (AR_calc - AR) / dAR_dM, 1.001)

        return M_exit_sup, None
    
    M_exit_sup, _ = jax.lax.scan(step, 2.0 * jnp.ones_like(gamma), jnp.arange(5))
    
    M_exit_sub  = jnp.sqrt((2.0 / (gamma - 1.0)) * ((safe_PR)**((gamma - 1.0) / gamma) - 1.0))
    M_exit      = jnp.where(choked, M_exit_sup, M_exit_sub)
    M_throat    = jnp.where(choked, 1.0, M_exit)

    # Nozzle Mass Flow
    m_1 = (P_t * A_throat) / jnp.sqrt(R * T_t)
    m_2 = jnp.sqrt(gamma) * M_throat
    m_3 = (1.0 + (gamma - 1.0) / 2.0 * M_throat**2) ** (- (gamma + 1.0) / (2.0 * (gamma - 1.0)))

    mdot_out = m_1 * m_2 * m_3

    P_out_choked = P_t / (1.0 + (gamma - 1.0) / 2.0 * M_exit**2) ** (gamma / (gamma - 1.0))
    P_out = jnp.where(choked, P_out_choked, P0)
    P_t_out = P_out * (1.0 + (gamma - 1.0) / 2.0 * M_exit**2) ** (gamma / (gamma - 1.0))
    
    T_out = T_t / (1.0 + (gamma - 1.0) / 2.0 * M_exit**2)
    T_t_out = T_t
    
    h_t_out = gas.compute_enthalpy(T_t_out)
    h_out = gas.compute_enthalpy(T_out)
    u_out = jnp.sqrt(2.0 * (h_t_out - h_out)) * n_v

    rho_out = P_out / (R * T_out)

    return mdot_out, M_exit, u_out, rho_out, P_out, P_t_out, T_out, T_t_out, h_out, h_t_out

@register
class FixedNozzle(FlowNode):
    tag: str = init_field("Core Nozzle", static=True)

    inputs: tuple | GraphInput = (
        GraphInput("flow", "Turbine"),
        GraphInput("fuel", "Combustor"),
    )

    @tu.inputs(
        "state.freestream.stagnation_temperature",
        "state.freestream.stagnation_pressure",
        "state.freestream.pressure",
        "state.freestream.mach_number",
        "state.freestream.Cp",
        "state.freestream.gamma",
        "state.freestream.R",
        "system.energy.nodes[ExpansionNozzle].pressure_ratio",
        "system.energy.nodes[ExpansionNozzle].design_parameters.eff.flow",
    )
    @tu.outputs(
        "state.energy.nodes[ExpansionNozzle].outputs.flow",
        "state.energy.nodes[ExpansionNozzle].outputs.flow.area_ratio",
        "state.energy.nodes[ExpansionNozzle].outputs.flow.mach_number",
        "state.energy.nodes[ExpansionNozzle].outputs.flow.density",
        "state.energy.nodes[ExpansionNozzle].outputs.flow.speed",
        "state.energy.nodes[ExpansionNozzle].outputs.flow.pressure",
        "state.energy.nodes[ExpansionNozzle].outputs.flow.stagnation_pressure",
        "state.energy.nodes[ExpansionNozzle].outputs.flow.temperature",
        "state.energy.nodes[ExpansionNozzle].outputs.flow.stagnation_temperature",
        "state.energy.nodes[ExpansionNozzle].outputs.flow.enthalpy",
        "state.energy.nodes[ExpansionNozzle].outputs.flow.stagnation_enthalpy",
    )
    def transmit(self, state: State, system: System, settings: Settings):

        updated_system = system
        
        fs = state.freestream
        P0 = fs.pressure
        
        gas, T_t, P_t, _, FAR, _ = self.mix_inputs(state)
        
        if settings.analysis.energy.design_mode:
            
            mdot_out=state.energy.mass_flow_rate * (1. + FAR)

            A_t, A_x, M_out, rho_out, u_out, P_out, P_t_out, T_out, T_t_out, h_out, h_t_out = _nozzle_design(
                gas=gas,
                T_t=T_t,
                P_t=P_t,
                mdot=mdot_out,
                P0=P0,
                PR=self.design_parameters.pressure_ratio,
                n_v=self.design_parameters.eff.flow,
            )
            
            updated_design_parameters = eqx.tree_at(lambda d:(
                    d.A_throat,
                    d.A_exit,
                    d.A_ratio
                ), self.design_parameters,(
                    A_t.squeeze(),
                    A_x.squeeze(),
                    A_x.squeeze()/A_t.squeeze()
                ),
            )
            updated_system = eqx.tree_at(
                lambda s: s.energy.nodes[self.network_ID].design_parameters,
                updated_system,
                updated_design_parameters
            )
        
        else:
            A_t = self.design_parameters.A_throat
            A_x = self.design_parameters.A_exit
            
            mdot_out, M_out, u_out, rho_out, P_out, P_t_out, T_out, T_t_out, h_out, h_t_out = _nozzle_performance(
                gas=gas,
                T_t=T_t,
                P_t=P_t,
                P0=P0,
                A_throat=A_t,
                A_exit=A_x,
                n_v=self.design_parameters.eff.flow,
            )

        # Physical outflow
        outputs = state.energy.nodes[self.network_ID].outputs.flow

        outputs = eqx.tree_at(lambda o: o.area, outputs, A_x)
        outputs = eqx.tree_at(lambda o: o.mass_flow_rate, outputs, mdot_out)
        outputs = eqx.tree_at(lambda o: o.mach_number, outputs, M_out)
        outputs = eqx.tree_at(lambda o: o.density, outputs, rho_out)
        outputs = eqx.tree_at(lambda o: o.speed, outputs, u_out)
        outputs = eqx.tree_at(lambda o: o.pressure, outputs, P_out)
        outputs = eqx.tree_at(lambda o: o.stagnation_pressure, outputs, P_t_out)
        outputs = eqx.tree_at(lambda o: o.temperature, outputs, T_out)
        outputs = eqx.tree_at(lambda o: o.stagnation_temperature, outputs, T_t_out)
        outputs = eqx.tree_at(lambda o: o.enthalpy, outputs, h_out)
        outputs = eqx.tree_at(lambda o: o.stagnation_enthalpy, outputs, h_t_out)

        updated_state = eqx.tree_at(lambda s: s.energy.nodes[self.network_ID].outputs.flow, state, outputs)

        # Residual update
        updated_state = eqx.tree_at(
            lambda s: s.energy.outputs.residual.mass_flow_rate,
            updated_state,
            (mdot_out / (1. + FAR) - state.energy.mass_flow_rate)/system.energy.design_parameters.mass_flow_rate
        )

        return updated_state, updated_system, settings

def _variable_nozzle_performance(
        gas: IdealGas,
        T_t: jnp.ndarray,
        P_t: jnp.ndarray,
        P0: jnp.ndarray,
        mdot_in: jnp.ndarray,
        n_v: jnp.ndarray | float,
    ):

    gamma = gas.compute_gamma(T_t)
    R = gas.R_specific

    # 1. Determine Pressure Ratio and Choking
    critical_PR = (1.0 + (gamma - 1.0) / 2.0) ** (gamma / (gamma - 1.0))
    actual_PR = P_t / P0
    safe_PR = jnp.maximum(actual_PR, 1.00001) # Prevent div by zero or negative root
    choked = actual_PR >= critical_PR

    # 2. Perfect Expansion Exit Mach
    # Because we vary the nozzle to perfectly expand to ambient pressure (P_out = P0),
    # M_exit is explicitly a function of the total-to-ambient pressure ratio.
    M_out = jnp.sqrt((2.0 / (gamma - 1.0)) * ((safe_PR)**((gamma - 1.0) / gamma) - 1.0))
    
    # 3. Throat Mach
    M_throat = jnp.where(choked, 1.0, M_out)

    # 4. Explicitly Calculate Required Physical Areas
    def calc_area_for_mach(M):
        # Apply a floor to M to prevent NaN gradients in JAX if M approaches 0
        M_safe = jnp.maximum(M, 1e-5) 
        m_term = jnp.sqrt(gamma) * M_safe
        temp_term = (1.0 + (gamma - 1.0) / 2.0 * M_safe**2) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))
        return (mdot_in * jnp.sqrt(R * T_t) * temp_term) / (P_t * m_term)

    A_throat = calc_area_for_mach(M_throat)
    A_exit = calc_area_for_mach(M_out)

    # 5. Thermodynamics and Kinematics
    P_out = P0  # Perfect expansion assumption guarantees this
    P_t_out = P_out * (1.0 + (gamma - 1.0) / 2.0 * M_out**2) ** (gamma / (gamma - 1.0))
    
    T_out = T_t / (1.0 + (gamma - 1.0) / 2.0 * M_out**2)
    T_t_out = T_t
    
    h_t_out = gas.compute_enthalpy(T_t_out)
    h_out = gas.compute_enthalpy(T_out)
    u_out = jnp.sqrt(jnp.maximum(2.0 * (h_t_out - h_out), 0.0)) * n_v

    rho_out = P_out / (R * T_out)

    return M_out, u_out, rho_out, P_out, P_t_out, T_out, T_t_out, h_out, h_t_out, A_throat, A_exit

@register
class VariableNozzle(FlowNode):
    tag: str = init_field("Core Nozzle", static=True)

    inputs: tuple | GraphInput = (
        GraphInput("flow", "Turbine"),
        GraphInput("fuel", "Combustor"),
    )

    @tu.inputs(
        "state.freestream.stagnation_temperature",
        "state.freestream.stagnation_pressure",
        "state.freestream.pressure",
        "state.freestream.mach_number",
        "state.freestream.Cp",
        "state.freestream.gamma",
        "state.freestream.R",
        "system.energy.nodes[ExpansionNozzle].pressure_ratio",
        "system.energy.nodes[ExpansionNozzle].design_parameters.eff.flow",
    )
    @tu.outputs(
        "state.energy.nodes[ExpansionNozzle].outputs.flow",
        "state.energy.nodes[ExpansionNozzle].outputs.flow.area_ratio",
        "state.energy.nodes[ExpansionNozzle].outputs.flow.mach_number",
        "state.energy.nodes[ExpansionNozzle].outputs.flow.density",
        "state.energy.nodes[ExpansionNozzle].outputs.flow.speed",
        "state.energy.nodes[ExpansionNozzle].outputs.flow.pressure",
        "state.energy.nodes[ExpansionNozzle].outputs.flow.stagnation_pressure",
        "state.energy.nodes[ExpansionNozzle].outputs.flow.temperature",
        "state.energy.nodes[ExpansionNozzle].outputs.flow.stagnation_temperature",
        "state.energy.nodes[ExpansionNozzle].outputs.flow.enthalpy",
        "state.energy.nodes[ExpansionNozzle].outputs.flow.stagnation_enthalpy",
    )
    def transmit(self, state: State, system: System, settings: Settings):

        updated_system = system
        
        fs = state.freestream
        P0 = fs.pressure
        
        working_fluid, T_t, P_t, W_in, _, _ = self.mix_inputs(state)
        
        if settings.analysis.energy.design_mode:
            A_t, A_x, M_out, rho_out, u_out, P_out, P_t_out, T_out, T_t_out, h_out, h_t_out = _nozzle_design(
                gas=working_fluid,
                T_t=T_t,
                P_t=P_t,
                mdot=W_in,
                P0=P0,
                PR=self.design_parameters.pressure_ratio,
                n_v=self.design_parameters.eff.flow,
            )
            
            updated_design_parameters = eqx.tree_at(lambda d:(
                    d.A_throat,
                    d.A_exit,
                    d.A_ratio
                ), self.design_parameters,(
                    A_t.squeeze(),
                    A_x.squeeze(),
                    A_x.squeeze()/A_t.squeeze()
                ),
            )
            updated_system = eqx.tree_at(
                lambda s: s.energy.nodes[self.network_ID].design_parameters,
                updated_system,
                updated_design_parameters
            )
        
        else:            
            M_out, u_out, rho_out, P_out, P_t_out, T_out, T_t_out, h_out, h_t_out, A_t, A_x = (
                _variable_nozzle_performance(
                    gas=working_fluid,
                    T_t=T_t,
                    P_t=P_t,
                    P0=P0,
                    mdot_in=W_in,
                    n_v=self.design_parameters.eff.flow,
                )
            )

        # Physical outflow
        outputs = state.energy.nodes[self.network_ID].outputs.flow

        outputs = eqx.tree_at(lambda o: o.area, outputs, A_x)
        outputs = eqx.tree_at(lambda o: o.mass_flow_rate, outputs, W_in)
        outputs = eqx.tree_at(lambda o: o.mach_number, outputs, M_out)
        outputs = eqx.tree_at(lambda o: o.density, outputs, rho_out)
        outputs = eqx.tree_at(lambda o: o.speed, outputs, u_out)
        outputs = eqx.tree_at(lambda o: o.pressure, outputs, P_out)
        outputs = eqx.tree_at(lambda o: o.stagnation_pressure, outputs, P_t_out)
        outputs = eqx.tree_at(lambda o: o.temperature, outputs, T_out)
        outputs = eqx.tree_at(lambda o: o.stagnation_temperature, outputs, T_t_out)
        outputs = eqx.tree_at(lambda o: o.enthalpy, outputs, h_out)
        outputs = eqx.tree_at(lambda o: o.stagnation_enthalpy, outputs, h_t_out)

        updated_state = eqx.tree_at(lambda s: s.energy.nodes[self.network_ID].outputs.flow, state, outputs)

        # Residual update (Turbojet/Single Flow Only)
        A_t_des = self.design_parameters.A_throat
        A_t_res =  (A_t - A_t_des)/A_t_des
        updated_state = eqx.tree_at(
            lambda s: s.energy.outputs.residual.area,
            updated_state,
            A_t_res
        )

        return updated_state, updated_system, settings

# Turboshaft -------------------------------------------------------------------

@register
class Turboshaft(GraphNode):
    tag: str = init_field("Turboshaft", static=True)

    inputs: tuple | GraphInput = (
        GraphInput("mechanical", "compressor"),
        GraphInput("mechanical", "turbine"),
    )

    def transmit(self, state: State, system: System, settings: Settings):
        
        if settings.analysis.energy.design_mode:
            d_power = (self.apply_domain_op(jnp.sum, state, "mechanical", "power") / 2e7)
        else:
            d_power = (self.apply_domain_op(jnp.sum, state, "mechanical", "power") / 
                       system.energy.design_parameters.power) #type: ignore
        
        outputs = state.energy.nodes[self.network_ID].outputs
        outputs = eqx.tree_at(lambda o: o.residual.power, outputs, d_power)

        updated_state = eqx.tree_at(lambda s: s.energy.nodes[self.network_ID].outputs, state, outputs)

        return updated_state, system, settings

# ----------------------------------------------------------------------------------------------------------------------
# Engines
# ----------------------------------------------------------------------------------------------------------------------

def _engine_performance(
    u0,
    P0,
    g, 
    delta_SFC,
    v_fan_nozzle,
    A_fan_nozzle,
    P_fan_nozzle,
    v_core_nozzle,
    A_core_nozzle,
    P_core_nozzle,
    fuel_air_ratio,
    mdot_core,
    BPR,
):

    # 1. Calculate mass flows
    mdot_fan = mdot_core * BPR
    mdot_in_core = mdot_core / (1.0 + fuel_air_ratio) # Strip fuel for inlet momentum
    
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
    TSFC = (safe_mdot_fuel / safe_F_actual) * (1.0 - delta_SFC) / units.hr
    
    # Fuel flow in kg/hr
    ff = mdot_fuel * units.parse('kg/hr')
    
    specific_thrust_core = F_actual / mdot_core

    return F_actual, specific_thrust_core, I_sp, TSFC, p, ff

def _TurbojetSetup():

    inlet = Inlet()
    comp = Compressor()
    comb = Combustor()
    turb = Turbine()
    shaft = Turboshaft()
    nozz = VariableNozzle()

    return (inlet, comp, comb, turb, shaft, nozz)

@register
class JetGeometry(eqx.Module):
    xe: float = 1.0
    ye: float = 1.0
    Ce: float = 2.0

@register
class JetDesign(FlowDesign):
    thrust: float = 0.0
    delta_SFC: float = 0.0

    altitude: float = 0.0
    mach_number: float = 0.01

    temperature: float = 288.15  # Kelvin
    stagnation_temperature: float = 288.15  # Kelvin

    pressure: float = 101325.0  # Pascal
    stagnation_pressure: float = 101325.0  # Pascal

    isa_deviation: float = 0.0

    SLS_thrust: float = 0.0

    turbine_intake_temperature: float = 0.0
    bypass_ratio: float = 0.0

@register
class TurbojetEngine(FlowNode[JetDesign]):
    tag: str = init_field("Engine", static=True)
    subcomponents: tuple = init_field(_TurbojetSetup)

    plug_diameter: float = 0.0

    working_fluid: IdealGas = init_field(Air)
    design_parameters: JetDesign = init_field(JetDesign)

    inputs: tuple | GraphInput = init_field(
        (
            GraphInput("flow", "self.core_nozzle"),
            GraphInput("fuel", "self.combustor"),
            GraphInput("residual", "self.turboshaft"),
        ),
        static=True,
    )

    installation_geometry: JetGeometry = init_field(JetGeometry)

    _bookkeeping: dict = init_field(lambda: {
        "compressors": Compressor,
        "turbines": Turbine,
        "nozzles": VariableNozzle | FixedNozzle,
        "shafts": Turboshaft,
        "ducts": Splitter,
        }, static=True
    )

    @classmethod
    def build_custom(
        cls,
        variable_nozzle: bool = True,
        **kwargs
    ):
        
        inlet = FlowNode(tag="Inlet", inputs=GraphInput("flow", "freestream"))
        comp = Compressor()
        comb = Combustor()
        turb = Turbine()
        shaft = Turboshaft()

        if variable_nozzle:
            nozz = VariableNozzle()
        else:
            nozz = FixedNozzle()

        custom_subs = (inlet, comp, comb, turb, shaft, nozz)

        return cls(subcomponents=custom_subs, **kwargs)

    @tu.inputs(
        "state.freestream.gamma",
        "state.freestream.speed",
        "state.freestream.speed_of_sound",
        "state.freestream.mach_number",
        "state.freestream.pressure",
        "state.freestream.gravity",
        "state.energy.nodes[Turbojet].throttle",
        "state.energy.nodes[Turbojet_core_nozzle].outputs.flow.speed",
        "state.energy.nodes[Turbojet_core_nozzle].outputs.flow.area_ratio",
        "state.energy.nodes[Turbojet_core_nozzle].outputs.flow.pressure",
        "state.energy.nodes[Turbojet_combustor].outputs.fuel.fuel_air_ratio",
        "system.energy.nodes[Turbojet].design_parameters.total_thrust"
        "system.energy.nodes[Turbojet].design_parameters.delta_SFC",
    )
    @tu.outputs(
        "state.energy.nodes[Turbojet].outputs.force.thrust",
        "state.energy.nodes[Turbojet].outputs.force.nondimensional_thrust",
        "state.energy.nodes[Turbojet].outputs.force.specific_impulse",
        "state.energy.nodes[Turbojet].outputs.fuel.TSFC",
        "state.energy.nodes[Turbojet].outputs.fuel.flow_rate",
        "state.energy.nodes[Turbojet].outputs.flow.mass_flow_rate",
        "state.energy.nodes[Turbojet].outputs.mechanical.power",
    )
    def transmit(self, state: State, system: System, settings: Settings):        

        fs = state.freestream
        FAR = state.energy.nodes[self.network_ID + '.combustor'].outputs.flow.fuel_air_ratio
        
        # Core Flow
        core_flow = next((f for f in self.flow_inputs if "core" in f.network_ID), None)

        v_core = self.get_input_state(state, core_flow, "speed") if core_flow is not None else 0.0
        A_core = self.get_input_state(state, core_flow, "area") if core_flow is not None else 0.0
        P_core = self.get_input_state(state, core_flow, "pressure") if core_flow is not None else 0.0

        mdot_core = self.get_input_state(state, core_flow, "mass_flow_rate") if core_flow is not None else 1.0

        # Fan Flow
        fan_flow = next((f for f in self.flow_inputs if "fan" in f.network_ID), None)

        v_fan = self.get_input_state(state, fan_flow, "speed") if fan_flow is not None else 0.0
        A_fan = self.get_input_state(state, fan_flow, "area") if fan_flow is not None else 0.0
        P_fan = self.get_input_state(state, fan_flow, "pressure") if fan_flow is not None else 0.0

        BPR = getattr(state.energy, "bypass_ratio", 0.0)

        F, F_sp, I_sp, TSFC, p, ff = _engine_performance(
            u0=fs.speed,
            P0=fs.pressure,
            g=fs.gravity,
            delta_SFC=self.design_parameters.delta_SFC,
            v_fan_nozzle=v_fan,
            A_fan_nozzle=A_fan,
            P_fan_nozzle=P_fan,
            v_core_nozzle=v_core,
            A_core_nozzle=A_core,
            P_core_nozzle=P_core,
            fuel_air_ratio=FAR,
            mdot_core=mdot_core,
            BPR=BPR,
        )

        outputs = state.energy.nodes[self.network_ID].outputs

        outputs = eqx.tree_at(lambda o: o.force.thrust, outputs, F)
        outputs = eqx.tree_at(lambda o: o.force.nondimensional_thrust, outputs, F_sp)
        outputs = eqx.tree_at(lambda o: o.force.specific_impulse, outputs, I_sp)

        outputs = eqx.tree_at(lambda o: o.fuel.TSFC, outputs, TSFC)
        outputs = eqx.tree_at(lambda o: o.fuel.flow_rate, outputs, ff)

        outputs = eqx.tree_at(lambda o: o.flow.mass_flow_rate, outputs, mdot_core)

        outputs = eqx.tree_at(lambda o: o.mechanical.power, outputs, p)
        
        outputs = eqx.tree_at(
            lambda o: o.residual.power, outputs, self.apply_domain_op(jnp.sum, state, "residual", "power")
        )

        updated_state = eqx.tree_at(lambda s: s.energy.nodes[self.network_ID].outputs, state, outputs)

        return updated_state, system, settings

# Makes BPR split serializable for save/load
@register
class BPRSplit(eqx.Module):
    is_bypass: bool = init_field(True, static=True)

    def __call__(self, state):
        bpr = state.energy.bypass_ratio
        if self.is_bypass:
            return bpr / (1.0 + bpr)
        else:
            return 1.0 / (1.0 + bpr)

def _TurbofanSetup():

    inlet = Inlet()
    fan = Compressor(tag="Fan", map=map_data.Fan)
    
    # Core Flow ----------------------------------------------------------------

    core_flow = Splitter(
        tag="Core Flow",
        inputs=GraphInput("flow", "fan"),
        values=("mass_flow_rate",),
        fractions=BPRSplit(is_bypass=False)
    )
    core_duct = FlowNode(tag="Core Duct", inputs=GraphInput("flow", "core flow"))
    
    # Compressors
    lpc = Compressor(tag="LPC", map=map_data.LPC, inputs=GraphInput("flow", "core duct"))
    
    c_stat = FlowNode(tag="Compressor Stator", inputs=GraphInput("flow", "lpc"))
    
    hpc = Compressor(tag="HPC", map=map_data.HPC,
                     inputs=GraphInput("flow", "compressor stator"),
                     output_bleeds=(
                        BleedFlow(tag="outlet"),
                        BleedFlow(tag="LPT cooling"),
                        BleedFlow(tag="nozzle cooling"),))

    cooling = FlowNode(tag="Cooling Duct",
                       inputs=GraphInput("flow", "hpc"),
                       output_bleeds=(
                           BleedFlow("HPT cooling"),
                           BleedFlow("LPT cooling"),))

    # Combustor    
    comb = Combustor(inputs=GraphInput("flow", "cooling_duct"))
    
    # Turbines
    hpt = Turbine(tag="HPT", map=map_data.HPT,
                  inputs=(
                      GraphInput("flow", "combustor", primary=True),
                      GraphInput("flow", "cooling_duct.hpt_cooling"),))
    
    t_stat = FlowNode(tag="Turbine Stator", inputs=(
                        GraphInput("flow", "hpt", primary=True),
                        GraphInput("flow", "hpc.lpt_cooling"),
                        GraphInput("flow", "cooling_duct.lpt_cooling"),))
    
    lpt = Turbine(tag="LPT", map=map_data.LPT,
                  inputs=GraphInput("flow", "turbine_stator"))
    
    # Turboshafts
    lp_shaft = Turboshaft(tag="LP Shaft", inputs=(
            GraphInput("mechanical", "lpc"),
            GraphInput("mechanical", "fan"),
            GraphInput("mechanical", "lpt"),))
    
    hp_shaft = Turboshaft(tag="HP Shaft", inputs=(
            GraphInput("mechanical", "hpc"),
            GraphInput("mechanical", "hpt"),))
    
    # Core Nozzle
    cn_duct = FlowNode(tag="Core Nozzle Duct", inputs=(
        GraphInput("flow", "lpt", primary=True),
        GraphInput("flow", "hpc.nozzle_cooling"),
    ))
    c_nozz = VariableNozzle(inputs=GraphInput("flow", "core_nozzle_duct"))

    # Bypass Flow --------------------------------------------------------------
    fan_flow = Splitter(
        tag="Fan Flow",
        inputs=GraphInput("flow", "fan"),
        values=("mass_flow_rate",),
        fractions=BPRSplit(is_bypass=True))
    
    fn_duct = FlowNode(tag="Fan Duct", inputs=GraphInput("flow", "fan flow"),
                       output_bleeds=(BleedFlow(tag="outlet"),))
    
    f_nozz = VariableNozzle(tag="Fan Nozzle", inputs=(GraphInput("flow", "fan duct")))

    return (inlet, fan,
            core_flow, core_duct,
            lpc, c_stat, hpc, cooling,
            comb,
            hpt, t_stat, lpt,
            lp_shaft, hp_shaft,
            cn_duct, c_nozz,
            fan_flow,
            fn_duct, f_nozz)

def TurbofanEngine(**kwargs):
    return TurbojetEngine(
        subcomponents=_TurbofanSetup(),
        inputs = (
            GraphInput("flow", "self.core_nozzle"),
            GraphInput("flow", "self.fan_nozzle"),
            GraphInput("fuel", "self.combustor"),
            GraphInput("residual", "self.lp_shaft"),
            GraphInput("residual", "self.hp_shaft"),
        ),
        **kwargs
    )

