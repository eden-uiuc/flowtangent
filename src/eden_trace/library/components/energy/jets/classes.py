# Trace/Library/Components/Energy/Propulsors.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: May 2025, Trace Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal, Optional
if TYPE_CHECKING:
    from eden_trace.framework import Settings, State, System
    from eden_trace.framework.state_data.energy import TurbojetData
    from eden_trace.framework.analyses.energy.jets import JetSettings

import json
import warnings
from pathlib import Path
from dataclasses import replace

# package imports
import equinox as eqx
import jax
import jax.numpy as jnp

import eden_trace.utils as tu

# Trace imports
from eden_trace.utils import init_field, register
from eden_trace.library import units

from ..maps import data as map_data
from ..maps.classes import CompressorMap, TurbineMap
from ..nodes import GraphInput, GraphNode, Splitter, FlowNode, FlowDesign, BleedFlow
from eden_trace.library.gases import Air, BurnedJetA, Gas
from eden_trace.library.propellants import JetA, Propellant

# ----------------------------------------------------------------------------------------------------------------------
# Turbojet Components
# ----------------------------------------------------------------------------------------------------------------------

# Inlet ------------------------------------------------------------------------

@register
class Inlet(FlowNode):
    tag: str = init_field("inlet", static=True)

    @tu.inputs(
        "state.freestream",
        "state.energy.mass_flow_rate",
        "system.energy.nodes['{network_ID}'].design_parameters.pressure_ratio",
        "system.energy.nodes['{network_ID}'].design_parameters.pressure_recovery",
        "system.energy.nodes['{network_ID}'].design_parameters.eff.flow",
        "system.energy.nodes['{network_ID}'].design_parameters.exit_mach_number: Optional",
    )
    @tu.outputs(
        "system.energy.nodes['{network_ID}'].design_parameters.A_exit: Optional"
        "state.energy.nodes['{network_ID}'].flow"
    )
    def transmit(self, state: State, system: Aircraft, settings: Settings):  # type: ignore

        network_state   = state.energy
        state_node      = network_state.nodes[self.network_ID]
        system_node     = system.energy.nodes[self.network_ID]
        des_params      = system_node.design_parameters

        updated_system = system
        fs = state.freestream
        
        analysis_settings: JetSettings = settings.analysis.energy
        design_mode = analysis_settings.design_mode
        statics = analysis_settings.statics

        gas = fs.atmosphere.fluid
        T_t = fs.stagnation_temperature
        P_t = fs.stagnation_pressure
        M0  = fs.mach_number

        PR    = jnp.atleast_2d(des_params.pressure_ratio)
        P_rec = jnp.atleast_2d(des_params.pressure_recovery)
        M_out = jnp.atleast_2d(des_params.exit_mach_number)
        
        T_t_out, P_t_out = self.stagnation(gas, T_t, P_t, PR, 1.0, M0, P_rec)
        h_t_out = gas.compute_enthalpy(T_t_out)

        if design_mode:
            if statics:
                A_out, u_out, P_out, T_out, h_t_out, h_out = self.kinematic_design(
                    gas=gas,
                    T_t_out=T_t_out,
                    P_t_out=P_t_out,
                    M_out=M_out,
                    mdot=network_state.mass_flow_rate)
                
                updated_system = eqx.tree_at(
                    lambda s: s.energy.nodes[self.network_ID].design_parameters.A_exit,
                    updated_system,
                    A_out.squeeze())

        elif statics:
                T_out, P_out, h_t_out, h_out, u_out, M_out = self.statics(
                    gas=fs.atmosphere.fluid,
                    T_t=fs.stagnation_temperature,
                    P_t=fs.stagnation_pressure,
                    mdot=jnp.atleast_2d(network_state.mass_flow_rate),
                    area=des_params.A_exit,)

        outputs = state_node.flow

        outputs = eqx.tree_at(lambda o: o.mass_flow_rate, outputs,          jnp.atleast_2d(network_state.mass_flow_rate))
        outputs = eqx.tree_at(lambda o: o.stagnation_pressure, outputs,     jnp.atleast_2d(P_t_out))
        outputs = eqx.tree_at(lambda o: o.stagnation_temperature, outputs,  jnp.atleast_2d(T_t_out))
        outputs = eqx.tree_at(lambda o: o.stagnation_enthalpy, outputs,     jnp.atleast_2d(h_t_out))
        
        if statics:
            outputs = eqx.tree_at(lambda o: o.mach_number, outputs,         jnp.atleast_2d(M_out))
            outputs = eqx.tree_at(lambda o: o.temperature, outputs,         jnp.atleast_2d(T_out))
            outputs = eqx.tree_at(lambda o: o.pressure, outputs,            jnp.atleast_2d(P_out))
            outputs = eqx.tree_at(lambda o: o.stagnation_enthalpy, outputs, jnp.atleast_2d(h_t_out))
            outputs = eqx.tree_at(lambda o: o.enthalpy, outputs,            jnp.atleast_2d(h_out))
            outputs = eqx.tree_at(lambda o: o.speed, outputs,               jnp.atleast_2d(u_out))

        updated_state = eqx.tree_at(lambda s: s.energy.nodes[self.network_ID].flow, state, outputs)

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
    tag: str = init_field("compressor", static=True)

    inputs: tuple | GraphInput = init_field(GraphInput("flow", "inlet"), static=True)

    map: CompressorMap = init_field(map_data.AXI5)

    alpha_schedule: Callable = init_field(_alpha_c, as_value=True, static=True)

    def __post_init__(self):
        if not isinstance(self.map, CompressorMap):
            raise TypeError(f"'{self.tag}' requires a CompressorMap, got {type(self.map).__name__}")
        if self.design_parameters.eff.flow == 1.0:
            map_effs = replace(self.design_parameters.eff, flow=self.map.eff_des + 0.0)  # +0.0 trick to force new memory allocation
            map_params = replace(self.design_parameters, eff=map_effs)
            object.__setattr__(self, "design_parameters", map_params)
        super(Compressor, self).__post_init__()

    @tu.inputs(
        "state.energy.nodes['{flow_inputs.network_ID}'].flow",
        "system.energy.nodes['{network_ID}'].design_parameters.pressure_ratio",
        "system.energy.nodes['{network_ID}'].design_parameters.eff.flow",
        "system.energy.nodes['{network_ID}'].design_parameters.rotation_speed",
        "system.energy.nodes['{network_ID}'].design_parameters.exit_mach_number: Optional",
    )
    @tu.outputs(
        "state.energy.nodes['{network_ID}'].flow",
        "state.energy.nodes['{network_ID}'].residual.{tag.lower()}_Wc",
        "system.energy.nodes['{network_ID}'].design_parameters.A_exit: Optional",
        "system.energy.nodes['{network_ID}'].map.s_Wc",
        "system.energy.nodes['{network_ID}'].map.s_PR",
        "system.energy.nodes['{network_ID}'].map.s_eff",
        "system.energy.nodes['{network_ID}'].map.s_Nc",
    )
    def transmit(self, state: State, system: System, settings: Settings):

        network_state   = state.energy
        state_node      = network_state.nodes[self.network_ID]
        system_node     = system.energy.nodes[self.network_ID]
        des_params      = system_node.design_parameters

        updated_system = system
        
        analysis_settings   = settings.analysis.energy
        design_mode         = analysis_settings.design_mode
        statics             = analysis_settings.statics
        
        gas, T_t, P_t, W_in, _, _ = self.mix_inputs(state)
        W_out = W_in * (1.0 - self.bleed_MFR_frac(state))

        theta_c = T_t / 288.15
        delta_c = P_t / 101325.0
        
        if design_mode:
            # Design Parameters
            M_out   = jnp.atleast_2d(des_params.exit_mach_number)
            PR      = jnp.atleast_2d(des_params.pressure_ratio)
            n_isn   = jnp.atleast_2d(des_params.eff.flow)
            N_des   = jnp.atleast_2d(des_params.rotation_speed)

            # Corrected Inflow
            Nc_des  = N_des / jnp.sqrt(theta_c)
            Wc_tgt  = W_in * jnp.sqrt(theta_c) / delta_c

            # Map Parameters
            PR_map  = system_node.map.PR_des
            Wc_map  = system_node.map.Wc_des
            eff_map = system_node.map.eff_des
            Nc_map  = system_node.map.Nc_des
            
            s_Wc =  (Wc_tgt / Wc_map).squeeze()
            s_PR = (PR - 1.0)/(PR_map - 1.0)
            s_eff = n_isn / eff_map
            s_Nc = (Nc_des/Nc_map).squeeze()

            T_t_out, P_t_out = self.stagnation(gas, T_t, P_t, PR, 1.0 / n_isn)
            
            if statics:
                A_out, u_out, P_out, T_out, h_t_out, h_out = self.kinematic_design(
                    gas=gas,
                    T_t_out=T_t_out,
                    P_t_out=P_t_out,
                    M_out=M_out,
                    mdot=W_in,)

                updated_design_paramters = eqx.tree_at(
                    lambda d: d.A_exit,
                    system_node.design_parameters,
                    A_out.squeeze(),)
            else:
                h_t_out = gas.compute_enthalpy(T_t_out)
                updated_design_paramters = system_node.design_parameters

            power = (h_t_out - jnp.atleast_2d(gas.compute_enthalpy(T_t))) * W_in

            updated_map = eqx.tree_at(
                lambda m: (m.s_Wc, m.s_PR, m.s_eff, m.s_Nc),
                self.map,
                (s_Wc, s_PR, s_eff, s_Nc))

            updated_system = eqx.tree_at(
                lambda s: (
                    s.energy.nodes[self.network_ID].design_parameters,
                    s.energy.nodes[self.network_ID].map,
                ),
                    updated_system,
                (
                    updated_design_paramters,
                    updated_map
                ))

        else:
            if self.tag.lower() == 'lpc' or self.tag.lower() == 'fan':
                N = jnp.atleast_2d(network_state.LP_speed)
            elif self.tag.lower() == 'hpc':
                N = jnp.atleast_2d(network_state.HP_speed)
            else:
                N = jnp.atleast_2d(network_state.rotation_speed)
            Nc_des = system_node.design_parameters.rotation_speed
            Nc     = N / jnp.sqrt(theta_c)
            
            alpha   = self.alpha_schedule(Nc, Nc_des)
            Rline = jnp.atleast_2d(getattr(network_state, f"{self.tag.lower()}_Rline")) # TODO: Shift to Rline scheduling on altitude, Mach number in future

            # Reference the nodal version of the map to ensure updated scalars
            PR, Wc, n_isn = system_node.map.evaluate(alpha, Nc, Rline)
            W_in = Wc * delta_c / jnp.sqrt(theta_c)

            power, P_t_out, T_t_out, h_t_out = _compressor_performance(
                gas=gas,
                T_t=T_t,
                P_t=P_t,
                PR=PR,
                n_isn=n_isn,)
            
            if statics:
                T_out, P_out, h_t_out, h_out, u_out, M_out = self.statics(
                    gas,
                    T_t_out,
                    P_t_out,
                    W_in,
                    des_params.A_exit)

        outputs = state_node

        outputs = eqx.tree_at(lambda o: o.mechanical.power, outputs, jnp.atleast_2d(power))

        outputs = eqx.tree_at(lambda o: o.flow.mass_flow_rate, outputs,         jnp.atleast_2d(W_out))
        outputs = eqx.tree_at(lambda o: o.flow.stagnation_pressure, outputs,    jnp.atleast_2d(P_t_out))
        outputs = eqx.tree_at(lambda o: o.flow.stagnation_temperature, outputs, jnp.atleast_2d(T_t_out))
        outputs = eqx.tree_at(lambda o: o.flow.stagnation_enthalpy, outputs,    jnp.atleast_2d(h_t_out))

        if statics:
            outputs = eqx.tree_at(lambda o: o.flow.temperature, outputs,        jnp.atleast_2d(T_out))
            outputs = eqx.tree_at(lambda o: o.flow.pressure, outputs,           jnp.atleast_2d(P_out))
            outputs = eqx.tree_at(lambda o: o.flow.enthalpy, outputs,           jnp.atleast_2d(h_out))
            outputs = eqx.tree_at(lambda o: o.flow.speed, outputs,              jnp.atleast_2d(u_out))
            outputs = eqx.tree_at(lambda o: o.flow.mach_number, outputs,        jnp.atleast_2d(M_out))

        updated_state = eqx.tree_at(lambda s: s.energy.nodes[self.network_ID], state, outputs)

        # Residual Update
        if isinstance(system.energy.line.engine.design_parameters, tuple):
            eng_des = system.energy.line.engine.design_parameters[0]
        else:
            eng_des = system.energy.line.engine.design_parameters
        W_des   = eng_des.mass_flow_rate
        Wc_res  = (W_in - state.energy.mass_flow_rate)/W_des

        updated_state = eqx.tree_at(
            lambda s:getattr(s.energy.residual, f"{self.tag.lower()}_Wc"),
            updated_state,
            Wc_res)

        return updated_state, updated_system, settings

# Burner -----------------------------------------------------------------------

def _burner_design(
    gas: Gas,
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

def _burner_performance(
    gas: Gas,            # Gas or BurnedGas model
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
class Burner(FlowNode):
    tag: str = init_field("Burner", static=True)

    inputs: tuple | GraphInput = init_field(GraphInput("flow", "Compressor"), static=True)
    fuel: Propellant = init_field(JetA)

    @tu.inputs(
        "state.energy.target_temperature",
        "state.energy.nodes['{flow_inputs.network_ID}'].flow",
        "system.energy.nodes['{network_ID}'].fuel.specific_energy",
        "system.energy.nodes['{network_ID}'].design_parameters.pressure_ratio",
        "system.energy.nodes['{network_ID}'].design_parameters.eff.flow",
        "system.energy.nodes['{network_ID}'].design_parameters.exit_mach_number: Optional",
    )
    @tu.outputs(
        "state.energy.nodes['{network_ID}'].flow",
        "system.energy.nodes['{network_ID}'].design_parameters.A_exit: Optional"
    )
    def transmit(self, state: State, system: System, settings: Settings):
        
        network_state   = state.energy
        state_node      = network_state.nodes[self.network_ID]
        system_node     = system.energy.nodes[self.network_ID]
        des_params      = system_node.design_parameters

        updated_system = system

        gas, T_t, P_t, W_in, _, _ = self.mix_inputs(state)

        LHV = system_node.fuel.specific_energy
        PR  = des_params.pressure_ratio
        n_b = des_params.eff.flow

        analysis_settings   = settings.analysis.energy
        design_mode         = analysis_settings.design_mode
        statics             = analysis_settings.statics

        if design_mode:
            T_t_out = state.energy.target_temperature

            P_t_out, h_t_out, FAR, mdot_out = _burner_design(
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
            if statics:
                A_out, u_out, P_out, T_out, h_t_out, h_out = self.kinematic_design(
                    gas=gas,
                    T_t_out=T_t_out,
                    P_t_out=P_t_out,
                    M_out=des_params.exit_mach_number,
                    mdot=W_in * (1.0 + FAR)
                )
                
                updated_system = eqx.tree_at(
                    lambda s: s.energy.nodes[self.network_ID].design_parameters.A_exit,
                    updated_system,
                    A_out.squeeze(),)
        
        else:    
            FAR = state.energy.fuel_air_ratio

            P_t_out, T_t_out, h_t_out, mdot_out = _burner_performance(
                gas=gas,
                fuel=self.fuel,
                T_t=T_t,
                P_t=P_t,
                mdot_in=W_in,
                FAR=FAR,
                PR=PR,
                n_b=n_b)
            
            if statics:
                T_out, P_out, h_t_out, h_out, u_out, M_out = self.statics(
                    gas,
                    T_t_out,
                    P_t_out,
                    mdot_out,
                    des_params.A_exit)

        outputs = state_node.flow

        outputs = eqx.tree_at(lambda o: o.stagnation_pressure, outputs,     jnp.atleast_2d(P_t_out))
        outputs = eqx.tree_at(lambda o: o.stagnation_temperature, outputs,  jnp.atleast_2d(T_t_out))
        outputs = eqx.tree_at(lambda o: o.stagnation_enthalpy, outputs,     jnp.atleast_2d(h_t_out))
        outputs = eqx.tree_at(lambda o: o.mass_flow_rate, outputs,          jnp.atleast_2d(mdot_out))
        outputs = eqx.tree_at(lambda o: o.fuel_air_ratio, outputs,          jnp.atleast_2d(FAR))
        outputs = eqx.tree_at(lambda o: o.fluid, outputs,                   BurnedJetA(FAR))

        if statics:
            outputs = eqx.tree_at(lambda o: o.flow.temperature, outputs,    jnp.atleast_2d(T_out))
            outputs = eqx.tree_at(lambda o: o.flow.pressure, outputs,       jnp.atleast_2d(P_out))
            outputs = eqx.tree_at(lambda o: o.flow.enthalpy, outputs,       jnp.atleast_2d(h_out))
            outputs = eqx.tree_at(lambda o: o.flow.speed, outputs,          jnp.atleast_2d(u_out))
            outputs = eqx.tree_at(lambda o: o.flow.mach_number, outputs,    jnp.atleast_2d(M_out))

        updated_state = eqx.tree_at(lambda s: s.energy.nodes[self.network_ID].flow, state, outputs)

        return updated_state, updated_system, settings

# Turbine ----------------------------------------------------------------------

def _turbine_performance(
    gas,  # The BurnedGas mixture
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
    power = (h_t_out - h_t_in) * n_mech

    return T_t_out, P_t_out, h_t_out, power

@register
class Turbine(FlowNode):
    tag: str = init_field("Turbine", static=True)

    map: TurbineMap = init_field(map_data.LPT2269)

    alpha_schedule: Callable = init_field(lambda Np, Np_des: jnp.full_like(Np, 1.0), as_value=True, static=True)

    inputs: tuple | GraphInput = init_field(
        (
            GraphInput("flow", "Burner"),
        ), static=True,
    )

    def __post_init__(self):
        if not isinstance(self.map, TurbineMap):
            raise TypeError(f"'{self.tag}' requires a TurbineMap, got {type(self.map).__name__}")
        if self.design_parameters.eff.flow == 1.0:
            map_effs = replace(self.design_parameters.eff, flow=self.map.eff_des)
            map_params = replace(self.design_parameters, eff=map_effs)
            object.__setattr__(self, "design_parameters", map_params)
        super(Turbine, self).__post_init__()

    @tu.inputs(
        "state.energy.{tag.lower()}_PR",
        "state.energy.nodes['{flow_inputs.network_ID}'].flow",
        "system.energy.nodes['{network_ID}'].map",
        "system.energy.nodes['{network_ID}'].design_parameters.eff.flow",
        "system.energy.nodes['{network_ID}'].design_parameters.eff.mechanical",
        "system.energy.nodes['{network_ID}'].design_parameters.rotation_speed",
        "system.energy.nodes['{network_ID}'].design_parameters.exit_mach_number: Optional",
    )
    @tu.outputs(
        "state.energy.nodes['{network_ID}'].flow",
        "system.energy.nodes['{network_ID}'].map.s_Wp",
        "system.energy.nodes['{network_ID}'].map.s_PR",
        "system.energy.nodes['{network_ID}'].map.s_eff",
        "system.energy.nodes['{network_ID}'].map.s_Np",
        "system.energy.nodes['{network_ID}'].design_parameters.pressure_ratio",
        "system.energy.nodes['{network_ID}'].design_parameters.A_exit: Optional",
        
    )
    def transmit(self, state: State, system: System, settings: Settings):
        
        network_state   = state.energy
        state_node      = network_state.nodes[self.network_ID]
        system_node     = system.energy.nodes[self.network_ID]
        des_params      = system_node.design_parameters

        updated_system  = system

        analysis_settings   = settings.analysis.energy
        design_mode         = analysis_settings.design_mode
        statics             = analysis_settings.statics
        
        gas, T_t, P_t, W, FAR, _ = self.mix_inputs(state)

        if design_mode:
            PR = jnp.atleast_2d(getattr(network_state, f"{self.tag.lower()}_PR"))
            n_isn = jnp.atleast_1d(des_params.eff.flow)
            N_des = jnp.atleast_1d(des_params.rotation_speed)
            
            Np_des = N_des / jnp.sqrt(T_t)

            Wp_tgt = W * jnp.sqrt(T_t) / P_t

            Wp_map  = system_node.map.Wp_des
            eff_map = system_node.map.eff_des
            
            s_Wp = (Wp_tgt / Wp_map).squeeze()
            s_PR = ((PR - 1.0)/(system_node.map.PR_des - 1.0)).squeeze()
            s_eff = des_params.eff.flow / eff_map
            s_Np = (Np_des/system_node.map.Np_des).squeeze()

            # Turbine passes 1 / PR to reflect pressure drop
            safe_PR = jnp.clip(PR, min=1e-5)
            T_t_out, P_t_out = self.stagnation(gas, T_t, P_t, 1.0 / safe_PR, n_isn)
            
            if statics:

                M_out = jnp.atleast_1d(des_params.exit_mach_number)
                A_out, u_out, P_out, T_out, h_t_out, h_out = self.kinematic_design(
                    gas=gas,
                    T_t_out=T_t_out,
                    P_t_out=P_t_out,
                    M_out=M_out,
                    mdot=W
                )
            
            else:
                h_t_out = gas.compute_enthalpy(T_t_out)
                A_out = des_params.A_exit

            power = (jnp.atleast_2d(h_t_out - gas.compute_enthalpy(T_t))) * W

            updated_map = eqx.tree_at(
                lambda m: (m.s_Wp, m.s_PR, m.s_eff, m.s_Np),
                system_node.map,
                (s_Wp, s_PR, s_eff, s_Np)
            )

            updated_design = eqx.tree_at(
                lambda d: (
                    d.pressure_ratio,
                    d.A_exit,
                ),
                    des_params,
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
            if self.tag.lower() == 'lpt':
                N = jnp.atleast_2d(network_state.LP_speed)
            elif self.tag.lower() == 'hpt':
                N = jnp.atleast_2d(network_state.HP_speed)
            else:
                N = jnp.atleast_2d(network_state.rotation_speed)
            Np = N / jnp.sqrt(T_t)
            Np_des = des_params.rotation_speed

            PR = jnp.atleast_2d(getattr(network_state, f"{self.tag.lower()}_PR"))
            # PR = jnp.atleast_2d(system.energy.nodes[self.network_ID].design_parameters.pressure_ratio)
            alpha = self.alpha_schedule(Np, Np_des)

            # Reference nodal version of the map to ensure updated parameters
            Wp, n_isn = system_node.map.evaluate(alpha, Np, PR)
            W = Wp * P_t / jnp.sqrt(T_t) * (1. + FAR)

            T_t_out, P_t_out, h_t_out, power = _turbine_performance(
                gas=gas,
                PR=PR,
                n_isn=n_isn,
                n_mech=des_params.eff.mechanical,
                T_t=T_t,
                P_t=P_t,
            )

            if statics:
                T_out, P_out, h_t_out, h_out, u_out, M_out = self.statics(
                    gas,
                    T_t_out,
                    P_t_out,
                    W,
                    des_params.A_exit
                )

        # Set Output State
        outputs = state_node

        outputs = eqx.tree_at(lambda o: o.mechanical.power, outputs, jnp.atleast_2d(power))

        outputs = eqx.tree_at(lambda o: o.flow.mass_flow_rate, outputs,         jnp.atleast_2d(W))
        outputs = eqx.tree_at(lambda o: o.flow.fuel_air_ratio, outputs,         jnp.atleast_2d(FAR))
        outputs = eqx.tree_at(lambda o: o.flow.stagnation_pressure, outputs,    jnp.atleast_2d(P_t_out))
        outputs = eqx.tree_at(lambda o: o.flow.stagnation_temperature, outputs, jnp.atleast_2d(T_t_out))
        outputs = eqx.tree_at(lambda o: o.flow.stagnation_enthalpy, outputs,    jnp.atleast_2d(h_t_out))

        if statics:
            outputs = eqx.tree_at(lambda o: o.flow.temperature, outputs,        jnp.atleast_2d(T_out))
            outputs = eqx.tree_at(lambda o: o.flow.pressure, outputs,           jnp.atleast_2d(P_out))
            outputs = eqx.tree_at(lambda o: o.flow.enthalpy, outputs,           jnp.atleast_2d(h_out))
            outputs = eqx.tree_at(lambda o: o.flow.speed, outputs,              jnp.atleast_2d(u_out))
            outputs = eqx.tree_at(lambda o: o.flow.mach_number, outputs,        jnp.atleast_2d(M_out))

        updated_state = eqx.tree_at(lambda s: s.energy.nodes[self.network_ID], state, outputs)

        # Residual Update
        if isinstance(system.energy.line.engine.design_parameters, tuple):
            eng_des = system.energy.line.engine.design_parameters[0]
        else:
            eng_des = system.energy.line.engine.design_parameters
        W_des = eng_des.mass_flow_rate
        Wp_res = (W / (1. + FAR) - state.energy.mass_flow_rate)/W_des
        updated_state = eqx.tree_at(
            lambda s: getattr(s.energy.residual, f"{self.tag.lower()}_Wp"),
            updated_state,
            Wp_res)

        return updated_state, updated_system, settings

# Nozzle -----------------------------------------------------------------------

# Helpers ------------------------------------------------------------

def _isentropic_expansion(
    T_t: jnp.ndarray,
    P_t: jnp.ndarray,
    P0: jnp.ndarray,
    gamma: jnp.ndarray,
    PR: jnp.ndarray | float,
):

    # Isentropic Outputs
    P_t_out = jnp.maximum(P_t * PR, P0)  # Output stagnation pressure, minimum is freestream pressure
    T_t_out = T_t  # Output stagnation temperature, adiabatically conserved

    M_out = jnp.sqrt((((P_t_out / P0) ** ((gamma - 1.0) / gamma)) - 1.0) * 2.0 / (gamma - 1.0))  # Output Mach number
    T_out = T_t_out / (1.0 + (gamma - 1.0) / 2.0 * M_out**2)  # Output temperature

    return P_t_out, T_t_out, T_out, M_out

def _mass_flux(
        gas: Gas,
        T_t: jnp.ndarray,
        P_t: jnp.ndarray,
        M: jnp.ndarray
):
    gamma = gas.compute_gamma(T_t)
    R = gas.R_specific

    M_safe = jnp.maximum(M, 1e-6)
    m_term = jnp.sqrt(gamma * M_safe)

    temp_term = (1.0 + (gamma - 1.0) / 2.0 * M_safe**2) ** (- (gamma + 1.0) / (2.0 * (gamma - 1.0)))

    return (P_t / jnp.sqrt(R * T_t)) * m_term * temp_term

# Sections -----------------------------------------------------------

def _nozzle_design(
        gas: Gas,
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

    P_t_out, T_t_out, T_out, M_isn = _isentropic_expansion(T_t, P_t, P0, gamma, PR)

    # Supersonic Expansion / Choking Logic
    critical_PR = (1.0 + (gamma - 1.0) / 2.0) ** (gamma / (gamma - 1.0))
    is_choked = (P_t_out / P0) >= critical_PR

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

def _fixed_nozzle_performance(
        gas: Gas,
        T_t: jnp.ndarray,
        P_t: jnp.ndarray,
        P0: jnp.ndarray,
        diverging_section: bool,
        A_throat: jnp.ndarray | float,
        A_exit: jnp.ndarray | float,
        n_v: jnp.ndarray | float,
    ):

    gamma = gas.compute_gamma(T_t)
    R = gas.R_specific
    
    # Check for choked flow
    critical_PR = (1.0 + (gamma - 1.0) / 2.0) ** (gamma / (gamma - 1.0))
    actual_PR = P_t / P0
    choked = actual_PR >= critical_PR
    
    safe_PR = 1.0 + jnp.sqrt((actual_PR - 1.0)**2 + 1e-4)

    if diverging_section:
        # Find exit Mach number
        AR = A_exit / A_throat

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
    
    else:
        M_exit = jnp.where(choked, 1.0, jnp.sqrt((2.0 / (gamma - 1.0)) * (safe_PR**((gamma - 1.0) / gamma) - 1.0)))
        M_throat = M_exit
        A_throat = A_exit

    # Nozzle Mass Flow
    Q_dot = _mass_flux(gas, T_t, P_t, M_throat)
    mdot_out = Q_dot * A_throat

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

def _variable_nozzle_performance(
        gas: Gas,
        T_t: jnp.ndarray,
        P_t: jnp.ndarray,
        P0: jnp.ndarray,
        mdot_in: jnp.ndarray,
        n_v: jnp.ndarray | float,
    ):

    gamma = gas.compute_gamma(T_t)
    R = gas.R_specific

    # Determine Pressure Ratio and Choking
    critical_PR = (1.0 + (gamma - 1.0) / 2.0) ** (gamma / (gamma - 1.0))
    actual_PR = P_t / P0
    safe_PR = jnp.maximum(actual_PR, 1.00001) # Prevent div by zero or negative root
    choked = actual_PR >= critical_PR

    # Perfect Expansion Exit Mach
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
class Nozzle(FlowNode):
    tag: str = init_field("Core Nozzle", static=True)
    variable_exit: bool = init_field(False, static=True)
    diverging_section: bool = init_field(False, static=True)

    inputs: tuple | GraphInput = (
        GraphInput("flow", "Turbine"),
    )

    def __post_init__(self):
        super(Nozzle, self).__post_init__()
        if self.variable_exit:
            if not self.diverging_section:
                warnings.warn(f"Variable exit for nozzle '{self.tag}' requires diverging section. "
                              "Setting diverging section to True.")
                object.__setattr__(self, "diverging_section", True)

    @tu.inputs(
        "state.freestream",
        "system.energy.nodes['{network_ID}'].design_parameters.pressure_ratio",
        "system.energy.nodes['{network_ID}'].design_parameters.eff.flow",
        "system.energy.nodes['{network_ID}'].design_parameters.A_throat",
        "system.energy.nodes['{network_ID}'].design_parameters.A_exit",
    )
    @tu.outputs(
        "state.energy.nodes['{network_ID}'].flow",
        "state.energy.nodes['{network_ID}'].residual.area",
    )
    def transmit(self, state: State, system: System, settings: Settings):

        network_state   = state.energy
        state_node      = network_state.nodes[self.network_ID]
        system_node     = system.energy.nodes[self.network_ID]
        des_params      = system_node.design_parameters

        updated_state = state
        updated_system = system
        
        fs = state.freestream
        P0 = fs.pressure

        analysis_settings   = settings.analysis.energy
        design_mode         = analysis_settings.design_mode
        
        gas, T_t, P_t, W_in, _, _ = self.mix_inputs(state)
        
        if design_mode:
            
            mdot_out = W_in

            A_t, A_x, M_out, rho_out, u_out, P_out, P_t_out, T_out, T_t_out, h_out, h_t_out = _nozzle_design(
                gas=gas,
                T_t=T_t,
                P_t=P_t,
                mdot=mdot_out,
                P0=P0,
                PR=des_params.pressure_ratio,
                n_v=des_params.eff.flow,)
            
            updated_design_parameters = eqx.tree_at(lambda d:(
                    d.A_throat,
                    d.A_exit,
                ), des_params,(
                    A_t.squeeze(),
                    A_x.squeeze(),))
            
            updated_system = eqx.tree_at(
                lambda s: s.energy.nodes[self.network_ID].design_parameters,
                updated_system,
                updated_design_parameters)
        
        else:
            A_x = des_params.A_exit
            A_t = des_params.A_throat
            
            if self.variable_exit:
                M_out, u_out, rho_out, P_out, P_t_out, T_out, T_t_out, h_out, h_t_out, A_t, A_x = (
                    _variable_nozzle_performance(
                        gas=gas,
                        T_t=T_t,
                        P_t=P_t,
                        P0=P0,
                        mdot_in=W_in,
                        n_v=des_params.eff.flow,))
                
                mdot_out = W_in
                
                # Residual update (Turbojet/Single Flow Only)
                A_t_des = des_params.A_throat
                A_t_res =  (A_t - A_t_des)/A_t_des
                updated_state = eqx.tree_at(
                    lambda s: s.energy.residual.area,
                    updated_state,
                    A_t_res)

            else:
                mdot_out, M_out, u_out, rho_out, P_out, P_t_out, T_out, T_t_out, h_out, h_t_out = (
                    _fixed_nozzle_performance(
                        gas=gas,
                        T_t=T_t,
                        P_t=P_t,
                        P0=P0,
                        diverging_section=system_node.diverging_section,
                        A_throat=A_t,
                        A_exit=A_x,
                        n_v=des_params.eff.flow,))
                
                # Residual update
                if isinstance(system.energy.line.engine.design_parameters, tuple):
                    eng_des = system.energy.line.engine.design_parameters[0]
                else:
                    eng_des = system.energy.line.engine.design_parameters
                updated_state = eqx.tree_at(
                    lambda s: s.energy.nodes[self.network_ID].residual.mass_flow_rate,
                    updated_state,
                    ((mdot_out - W_in)/ eng_des.mass_flow_rate)
                )

        # Physical outflow
        outputs = state_node.flow

        outputs = eqx.tree_at(lambda o: o.area, outputs,                    jnp.atleast_2d(A_x))
        outputs = eqx.tree_at(lambda o: o.mass_flow_rate, outputs,          jnp.atleast_2d(mdot_out))
        outputs = eqx.tree_at(lambda o: o.mach_number, outputs,             jnp.atleast_2d(M_out))
        outputs = eqx.tree_at(lambda o: o.density, outputs,                 jnp.atleast_2d(rho_out))
        outputs = eqx.tree_at(lambda o: o.speed, outputs,                   jnp.atleast_2d(u_out))
        outputs = eqx.tree_at(lambda o: o.pressure, outputs,                jnp.atleast_2d(P_out))
        outputs = eqx.tree_at(lambda o: o.stagnation_pressure, outputs,     jnp.atleast_2d(P_t_out))
        outputs = eqx.tree_at(lambda o: o.temperature, outputs,             jnp.atleast_2d(T_out))
        outputs = eqx.tree_at(lambda o: o.stagnation_temperature, outputs,  jnp.atleast_2d(T_t_out))
        outputs = eqx.tree_at(lambda o: o.enthalpy, outputs,                jnp.atleast_2d(h_out))
        outputs = eqx.tree_at(lambda o: o.stagnation_enthalpy, outputs,     jnp.atleast_2d(h_t_out))

        updated_state = eqx.tree_at(lambda s: s.energy.nodes[self.network_ID].flow, updated_state, outputs)

        return updated_state, updated_system, settings

# Turboshaft -------------------------------------------------------------------

@register
class Turboshaft(GraphNode):
    tag: str = init_field("Turboshaft", static=True)

    inputs: tuple | GraphInput = (
        GraphInput("mechanical", "compressor"),
        GraphInput("mechanical", "turbine"),
    )

    @tu.inputs(
        "state.energy.nodes['{mechanical_inputs.network_ID}'].mechanical.power",
        "system.energy.nodes['network.line.engine'].design_parameters",
    )
    @tu.outputs(
        "state.energy.nodes['{network_ID}'].residual.power"
    )
    def transmit(self, state: State, system: System, settings: Settings):
        
        if settings.analysis.energy.design_mode:
            d_power = (self.apply_domain_op(jnp.sum, state, "mechanical", "power") / 2e7)
        
        else:
            if isinstance(system.energy.line.engine.design_parameters, tuple):
                des = system.energy.nodes['network.line.engine'].design_parameters[0]
            else:
                des = system.energy.nodes['network.line.engine'].design_parameters
            
            d_power = (self.apply_domain_op(jnp.sum, state, "mechanical", "power") / des.power) #type: ignore
        
        outputs = state.energy.nodes[self.network_ID]
        outputs = eqx.tree_at(lambda o: o.residual.power, outputs, d_power)

        updated_state = eqx.tree_at(lambda s: s.energy.nodes[self.network_ID], state, outputs)

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
    burn = Burner()
    turb = Turbine()
    shaft = Turboshaft()
    nozz = Nozzle()

    return (inlet, comp, burn, turb, shaft, nozz)

def _ABTurbojetSetup():

    base_components = _TurbojetSetup()
    ab = Burner(
        tag="Afterburner",
        inputs=(
            GraphInput("flow", "turbine"),
        ),
    )
    nozz = replace(base_components[-1],
                   inputs=(
        GraphInput("flow", "afterburner"),
        GraphInput("fuel", "afterburner"),
    ))
    return base_components[:-1] + (ab, nozz)

@register
class JetGeometry(eqx.Module):
    xe: float = 1.0
    ye: float = 1.0
    Ce: float = 2.0

@register
class JetKinematics(eqx.Module):
    """
    Exit Mach numbers for turbojet components
    """

    inlet: float = 0.6

    compressor: float = 0.3
    burner: float = 0.1
    turbine: float = 0.4

@register
class TurbojetDesign[KinType: JetKinematics | FanKinematics](FlowDesign):
    
    tag: str = init_field("TOC", static=True) # Top-of-Climb design point by default
    
    # Performance Parameters
    thrust: float = 0.0
    SLS_thrust: float = 0.0
    delta_SFC: float = 0.0

    # Flight Conditions
    altitude: float = 0.0
    mach_number: float = 1e-6

    temperature: float = 288.15  # Kelvin
    stagnation_temperature: float = 288.15  # Kelvin

    pressure: float = 101325.0  # Pascal
    stagnation_pressure: float = 101325.0  # Pascal

    # Component Parameters
    inlet_pressure_recovery: float = 0.999
    overall_pressure_ratio: float = 20.0 # Compressor PR, 'OPR' by convention
    burner_pressure_ratio: float = 0.97
    
    turbine_intake_temperature: float = 0.0
    afterburner_exit_temperature: float = 0.0
    
    # Control Values
    TSFC: float = 0.0
    mass_flow_rate: float = 100 * units.kg/units.s
    rotation_speed: float = 8_000 * units.rev/units.mins    # Single spool
    turbine_PR: float = 5.0                                 # Single spool
    power: float = 2e7 * units.W

    exit_mach_numbers: KinType = init_field(JetKinematics, static=True)

@register
class TurbojetEngine(FlowNode[TurbojetDesign | tuple]):
    tag: str = init_field("Engine", static=True)
    subcomponents: tuple = init_field(_TurbojetSetup)

    plug_diameter: float = 0.0

    working_fluid: Gas = init_field(Air)
    design_parameters: TurbojetDesign | tuple = init_field(TurbojetDesign)

    inputs: tuple | GraphInput = init_field(
        (
            GraphInput("flow", "self.core_nozzle"),
            GraphInput("fuel", "self.burner"),
            GraphInput("residual", "self.turboshaft"),
        ),
        static=True,
    )

    installation_geometry: JetGeometry = init_field(JetGeometry)

    _bookkeeping: dict = init_field(lambda: {
        "compressors": Compressor,
        "turbines": Turbine,
        "nozzles": Nozzle | Nozzle,
        "shafts": Turboshaft,
        "ducts": Splitter,
        }, static=True
    )

    @classmethod
    def build_custom(
        cls,
        variable_nozzle: bool = True,
        cd_nozzle: bool = True,
        afterburner: bool = False,
        turbofan: bool = False,
        **kwargs
    ):
        
        if turbofan:
            if afterburner:
                base_components = _ABTurbofanSetup()
            else:
                base_components = _TurbofanSetup()
        else:
            if afterburner:
                base_components = _ABTurbojetSetup()
            else:
                base_components = _TurbojetSetup()
        

        inlet = Inlet()
        comp = Compressor()
        comb = Burner()
        turb = Turbine()
        shaft = Turboshaft()
        diverging = variable_nozzle or cd_nozzle
        nozz = Nozzle(variable_exit=variable_nozzle, diverging_section=diverging)

        custom_subs = (inlet, comp, comb, turb, shaft, nozz)

        return cls(subcomponents=custom_subs, **kwargs)

    @classmethod
    def from_json(cls, filepath: str | Path):

        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Extract Type
        engine_cat = data.get("category", "civil").lower()
        engine_type = data.get("type", "turbojet").lower()
        engine_ab = data.get("AB", False)

        if engine_type == "turbofan":
            if engine_cat == "civil":
                # Create Synthetic TOC point
                des_kwargs = {
                    'tag': "TOC",
                    'mach_number': data.get("Cruise Mach", 0.8),
                    'altitude': data.get("Cruise Alt (kft)", 35.0) * 1000. * units.ft,
                    'thrust': data.get("Takeoff Thrust (lbf)", 0.0) * 0.25 * units.lbf,
                    'SLS_thrust': data.get("Takeoff Thrust (lbf)", 0.0) * units.lbf,
                    'bypass_ratio': data.get("Takeoff BR", 0.0),
                    'overall_pressure_ratio': data.get("Takeoff OPR", 0.0),
                    'turbine_intake_temperature': (data.get("TIT (F)", 2300.0) + 459.67) * units.R,
                    'mass_flow_rate': data.get("Takeoff Airflow (lbm/s)", 0.0) * units.lbm / units.s,
                    'TSFC': data.get("Cruise TSFC", 0.0) * units.liter / units.hr
                }

                cruise_kwargs = {
                    'tag': "Cruise",
                    'mach_number': data.get("Cruise Mach", 0.8),
                    'altitude': data.get("Cruise Alt (kft)", 35.0) * 1000. * units.ft,
                    'thrust': data.get("Cruise Thrust (lbf)", 0.0) * 0.25 * units.lbf,
                    'SLS_thrust': data.get("Takeoff Thrust (lbf)", 0.0) * units.lbf,
                    'bypass_ratio': data.get("Takeoff BR", 0.0),
                    'overall_pressure_ratio': data.get("Takeoff OPR", 0.0),
                    'turbine_intake_temperature': (data.get("TIT (F)", 2300.0) + 459.67) * units.R,
                    'mass_flow_rate': data.get("Takeoff Airflow (lbm/s)", 0.0) * 0.1 * units.lbm / units.s,
                    'TSFC': data.get("Cruise TSFC", 0.0) * units.liter / units.hr
                }

                takeoff_kwargs = {
                    'tag': "Takeoff",
                    'mach_number': 1e-6,
                    'altitude': 0.0,
                    'thrust': data.get("Takeoff Thrust (lbf)", 0.0) * units.lbf,
                    'SLS_thrust': data.get("Takeoff Thrust (lbf)", 0.0) * units.lbf,
                    'bypass_ratio': data.get("Takeoff BR", 0.0),
                    'overall_pressure_ratio': data.get("Takeoff OPR", 0.0),
                    'turbine_intake_temperature': (data.get("TIT (F)", 2300.0) + 459.67) * units.R,
                    'mass_flow_rate': data.get("Takeoff Airflow (lbm/s)", 0.0) * units.lbm / units.s,
                    'TSFC': 0.0
                }

                des_params = tuple(TurbofanDesign(**k) for k in [des_kwargs, cruise_kwargs, takeoff_kwargs])

            else:
                des_kwargs = {
                    'mach_number':1e-6,
                    'altitude':0.0,
                    'thrust': data.get("Thrust (lbf)", 0.0) * units.lbf,
                    'SLS_thrust': data.get("Thrust (lbf)", 0.0) * units.lbf,
                    'bypass_ratio': data.get("BR", 0.0),
                    'overall_pressure_ratio': data.get("OPR", 0.0),
                    'fan_pressure_ratio': data.get("FPR", 0.0),
                    'turbine_intake_temperature': (data.get("TIT (F)", 2300.0) + 459.67) * units.R,
                    'afterburner_exit_temperature': (data.get("AET (F)", -459.67) + 459.67) * units.R,
                    'mass_flow_rate': data.get("Takeoff Airflow (lbm/s)", 0.0) * units.lbm / units.s,
                }

                des_params = TurbofanDesign(**des_kwargs)

            return TurbojetEngine(
                subcomponents=_ABTurbofanSetup() if engine_ab else _TurbofanSetup(),
                inputs = 
                (
                    GraphInput("flow", "self.afterburner"),
                    GraphInput("fuel", "self.afterburner"),
                    GraphInput("fuel", "self.burner"),
                    GraphInput("residual", "self.lp_shaft"),
                    GraphInput("residual", "self.hp_shaft"),
                ) if engine_ab else
                (
                    GraphInput("flow", "self.core_nozzle"),
                    GraphInput("flow", "self.fan_nozzle"),
                    GraphInput("fuel", "self.burner"),
                    GraphInput("residual", "self.lp_shaft"),
                    GraphInput("residual", "self.hp_shaft"),
                ),
                design_parameters=des_params)
        
        elif engine_type == "turbojet":
            return TurbojetEngine(
                subcomponents=_ABTurbojetSetup() if engine_ab else _TurbojetSetup(),
                inputs = 
                (
                    GraphInput("flow", "self.afterburner"),
                    GraphInput("fuel", "self.afterburner"),
                    GraphInput("fuel", "self.burner"),
                    GraphInput("residual", "self.turboshaft"),
                ) if engine_ab else
                (
                    GraphInput("flow", "self.core_nozzle"),
                    GraphInput("flow", "self.fan_nozzle"),
                    GraphInput("fuel", "self.burner"),
                    GraphInput("residual", "self.turboshaft"),
                ),
                design_parameters=TurbojetDesign(
                    mach_number=1e-6,
                    altitude=0.0,
                    thrust=data.get("Takeoff Thrust (lbf)", 0.0) * units.lbf,
                    SLS_thrust=data.get("Takeoff Thrust (lbf)", 0.0) * units.lbf,
                    overall_pressure_ratio=data.get("Takeoff OPR", 0.0),
                    turbine_intake_temperature=(data.get("TIT (F)", 2300.0) + 459.67) * units.R,
                    mass_flow_rate=data.get("Takeoff Airflow (lbm/s)", 0.0) * units.lbm / units.s,
                ))

    @tu.inputs(
        "state.freestream",
        "state.energy.throttle",
        "state.energy.nodes['{flow_inputs.network_ID}'].flow",
        "system.energy.nodes['{network_ID}'].design_parameters",
    )
    @tu.outputs(
        "state.energy.nodes['{network_ID}'].force.thrust",
        "state.energy.nodes['{network_ID}'].force.nondimensional_thrust",
        "state.energy.nodes['{network_ID}'].force.specific_impulse",
        "state.energy.nodes['{network_ID}'].fuel.TSFC",
        "state.energy.nodes['{network_ID}'].fuel.flow_rate",
        "state.energy.nodes['{network_ID}'].flow.mass_flow_rate",
        "state.energy.nodes['{network_ID}'].mechanical.power",
    )
    def transmit(self, state: State, system: System, settings: Settings):        

        fs = state.freestream
        FAR = state.energy.nodes[self.network_ID + '.burner'].flow.fuel_air_ratio
        
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
        des = system.energy.nodes[self.network_ID].design_parameters
        if isinstance(des, tuple):
            des = des[0]
        else:
            des = des

        F, F_sp, I_sp, TSFC, p, ff = _engine_performance(
            u0=fs.speed,
            P0=fs.pressure,
            g=fs.gravity,
            delta_SFC=des.delta_SFC,
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

        outputs = state.energy.nodes[self.network_ID]

        outputs = eqx.tree_at(lambda o: o.force.thrust, outputs, F)
        outputs = eqx.tree_at(lambda o: o.force.nondimensional_thrust, outputs, F_sp)
        outputs = eqx.tree_at(lambda o: o.force.specific_impulse, outputs, I_sp)
        outputs = eqx.tree_at(lambda o: o.fuel.TSFC, outputs, TSFC)
        outputs = eqx.tree_at(lambda o: o.fuel.flow_rate, outputs, ff)
        outputs = eqx.tree_at(lambda o: o.flow.mass_flow_rate, outputs, mdot_core)
        outputs = eqx.tree_at(lambda o: o.mechanical.power, outputs, p)
        
        outputs = eqx.tree_at(lambda o: o.residual.thrust, outputs, (F - des.thrust)/des.thrust)
        outputs = eqx.tree_at(
            lambda o: o.residual.power, outputs, self.apply_domain_op(jnp.sum, state, "residual", "power"))

        updated_state = eqx.tree_at(lambda s: s.energy.nodes[self.network_ID], state, outputs)

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
    fan = Compressor(
        tag="Fan",
        map=map_data.Fan,)
    
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
                        BleedFlow(tag="outlet", fractions_dict={
                            'mass_flow_rate':0.05,
                            'stagnation_pressure': 0.5,
                            'stagnation_enthalpy': 0.5}),
                        BleedFlow(tag="LPT cooling", fractions_dict={
                            'mass_flow_rate':0.05,
                            'stagnation_pressure': 0.5,
                            'stagnation_enthalpy': 0.5}),
                        BleedFlow(tag="nozzle cooling", fractions_dict={
                            'mass_flow_rate':0.02,
                            'stagnation_pressure': 0.5,
                            'stagnation_enthalpy': 0.5}),))

    cooling = FlowNode(tag="Cooling Duct",
                       inputs=GraphInput("flow", "hpc"),
                       output_bleeds=(
                           BleedFlow(tag="HPT cooling", fractions_dict={
                            'mass_flow_rate':0.05,
                            'stagnation_pressure': 0.5,
                            'stagnation_enthalpy': 0.5}),
                           BleedFlow(tag="LPT cooling", fractions_dict={
                            'mass_flow_rate':0.10,
                            'stagnation_pressure': 0.5,
                            'stagnation_enthalpy': 0.5}),))

    # burner    
    comb = Burner(inputs=GraphInput("flow", "cooling_duct"))
    
    # Turbines
    hpt = Turbine(tag="HPT", map=map_data.HPT,
                  inputs=(
                      GraphInput("flow", "burner", primary=True),
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
    c_nozz = Nozzle(inputs=GraphInput("flow", "core_nozzle_duct"))

    # Bypass Flow --------------------------------------------------------------
    fan_flow = Splitter(
        tag="Fan Flow",
        inputs=GraphInput("flow", "fan"),
        values=("mass_flow_rate",),
        fractions=BPRSplit(is_bypass=True))
    
    fn_duct = FlowNode(tag="Fan Duct", inputs=GraphInput("flow", "fan flow"),
                       output_bleeds=(BleedFlow(tag="outlet", fractions_dict={"mass_flow_rate":0.005 }),))
    
    f_nozz = Nozzle(tag="Fan Nozzle", inputs=(GraphInput("flow", "fan duct")))

    return (inlet, fan,
            core_flow, core_duct,
            lpc, c_stat, hpc, cooling,
            comb,
            hpt, t_stat, lpt,
            lp_shaft, hp_shaft,
            cn_duct, c_nozz,
            fan_flow,
            fn_duct, f_nozz)

def _ABTurbofanSetup():

    base_components = _TurbofanSetup()
    ab = Burner(
        tag="Afterburner",
        inputs=(
            GraphInput("flow", "fan_nozzle"),
            GraphInput("flow", "core_nozzle"),
        ),
        add_mixer=True
    )
    return base_components + (ab,)

@register
class FanKinematics(eqx.Module):
    """
    Exit Mach Numbers for turbofan components
    """

    inlet: float = 0.75
    fan: float = 0.45

    core_duct: float = 0.35
    fan_duct: float = 0.45

    lpc: float = 0.3
    compressor_stator: float = 0.35
    hpc: float = 0.25
    cooling_duct: float = 0.3
    
    burner: float = 0.1
    
    hpt: float = 0.35
    turbine_stator: float = 0.3
    lpt: float = 0.4

    core_nozzle_duct: float = 0.45

@register
class TurbofanDesign(TurbojetDesign[FanKinematics]):

    # Control Values
    bypass_ratio: float = 0.0
    fan_pressure_ratio: float = 0.0

    lp_rotation_speed: float = 5_000 * units.rev / units.mins
    hp_rotation_speed: float = 15_000 * units.rev / units.mins

    HPT_PR: float = 5.0
    LPT_PR: float = 3.0

    exit_mach_numbers: FanKinematics = init_field(FanKinematics, static=True)
    

def TurbofanEngine(**kwargs):

    return TurbojetEngine(
        subcomponents=_TurbofanSetup(),
        inputs = (
            GraphInput("flow", "self.core_nozzle"),
            GraphInput("flow", "self.fan_nozzle"),
            GraphInput("fuel", "self.burner"),
            GraphInput("residual", "self.lp_shaft"),
            GraphInput("residual", "self.hp_shaft"),
        ),
        design_parameters=TurbofanDesign(),
        **kwargs
    )