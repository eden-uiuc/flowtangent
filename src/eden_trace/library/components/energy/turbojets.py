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


# package imports
import equinox as eqx
import jax.numpy as jnp

import eden_trace.utils as tu

# Trace imports
from eden_trace.utils import init_field, register

from eden_trace.library import units
from eden_trace.library.components.energy import maps
from eden_trace.library.components.energy.maps import CompressorMap, TurbineMap
from eden_trace.library.components.energy.nodes import GraphInput, GraphNode, GraphSplitter, FlowNode, FlowDesign
from eden_trace.library.gases import Air, BurnedJetA, IdealGas
from eden_trace.library.propellants import JetA, Propellant

# ----------------------------------------------------------------------------------------------------------------------
# General Helpers
# ----------------------------------------------------------------------------------------------------------------------

def _average_gamma_stagnation(
        gas: IdealGas,
        T_t: jnp.ndarray | float,
        P_t: jnp.ndarray | float,
        PR: jnp.ndarray | float,
        n_isn: jnp.ndarray | float,
):
    """
    Averages gamma across T_t and T_t_out to converge T_t_out
    """
    
    gamma_in = gas.compute_gamma(T_t)
    gamma_avg = gamma_in
    T_t_out_ideal = T_t * (PR ** ((gamma_in - 1.0) / gamma_in))

    for _ in range(3):
        gamma_out = gas.compute_gamma(T_t_out_ideal)
        gamma_avg = 0.5 * (gamma_in + gamma_out)
        T_t_out_ideal = T_t * (PR ** ((gamma_avg - 1.0) / gamma_avg))
    
    # Compressor passes 1 / n_isn, so T_t_out is higher, Turbine passes n_isn, so T_t_out is lower
    T_t_out = T_t + (T_t_out_ideal - T_t) * n_isn

    return T_t_out, P_t * PR

def _station_kinematic_design(
        gas: IdealGas,
        T_t_out,
        P_t_out,
        M_design,
        mdot,
):
    
    # Unpack boundary stagnation properties
    R = gas.R_specific
    gamma = gas.compute_gamma(T_t_out)
    
    # Compute exit static properties
    T_out = T_t_out / (1.0 + ((gamma - 1.0) / 2.0) * M_design ** 2)
    P_out = P_t_out * (T_out / T_t_out) ** (gamma / (gamma - 1.0))

    # Compute exit kinematic properties
    h_out = gas.compute_absolute_enthalpy(T_out)
    h_t_out = gas.compute_absolute_enthalpy(T_t_out)
    u_out = jnp.sqrt(jnp.maximum(2.0 * (h_t_out - h_out), 1e-10))

    rho_out = P_out / (R * T_out)
    
    A_out = mdot / (rho_out * u_out)

    return A_out, u_out, P_out, T_out, h_t_out, h_out

# ----------------------------------------------------------------------------------------------------------------------
# Turbojet Components
# ----------------------------------------------------------------------------------------------------------------------

# Inlet ------------------------------------------------------------------------

def _inlet_stagnation(gas, T_t, P_t, M0, PR, n_r):
    gamma = gas.compute_gamma(T_t)
    
    # Base isentropic recovery
    P_t_out_sub = P_t * PR * n_r
    
    # Normal Shock Recovery
    ns_P_t = (
        PR * P_t
        * ((((gamma + 1.0) * (M0**2.0)) / ((gamma - 1.0) * M0**2.0 + 2.0)) ** (gamma / (gamma - 1.0)))
        * ((gamma + 1.0) / (2.0 * gamma * M0**2.0 - (gamma - 1.0))) ** (1.0 / (gamma - 1.0))
    )
    
    P_t_out = jnp.where(M0 > 1.0, ns_P_t, P_t_out_sub)
    T_t_out = T_t # Adiabatic
    
    return P_t_out, T_t_out

def _inlet_performance(gas, T_t, P_t, M0, PR, n_r, mdot, A_exit):
    
    P_t_out, T_t_out = _inlet_stagnation(gas, T_t, P_t, M0, PR, n_r)
    gamma = gas.compute_gamma(T_t)
    R = gas.R_specific
    
    # The non-dimensional mass flow parameter we need to match
    Q = (mdot * jnp.sqrt(R * T_t_out)) / (P_t_out * A_exit * jnp.sqrt(gamma))
    
    # Newton loop to find subsonic Mach number
    M_out = 0.5 # Subsonic initial guess
    for _ in range(5):
        term = 1.0 + (gamma - 1.0) / 2.0 * M_out**2
        power = - (gamma + 1.0) / (2.0 * (gamma - 1.0))
        
        f = M_out * (term ** power) - Q
        
        # Derivative df/dM
        df_dM = (term ** power) + M_out * power * (term ** (power - 1.0)) * (gamma - 1.0) * M_out
        
        M_out = jnp.clip(M_out - f / df_dM, 1e-6, 0.99)
        
    # Calculate static properties using the solved Mach
    T_out = T_t_out / (1.0 + (gamma - 1.0) / 2.0 * M_out**2)
    P_out = P_t_out / (1.0 + (gamma - 1.0) / 2.0 * M_out**2) ** (gamma / (gamma - 1.0))
    
    h_t_out = gas.compute_enthalpy(T_t_out)
    h_out = gas.compute_enthalpy(T_out)
    u_out = jnp.sqrt(2.0 * (h_t_out - h_out))
    
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
        "system.energy.nodes[InletNozzle].efficiencies.flow",
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

        if settings.analysis.energy.design_mode:
            gas = fs.atmosphere.fluid
            T_t = fs.stagnation_temperature
            P_t = fs.stagnation_pressure
            M0  = fs.mach_number

            PR    = self.design_parameters.pressure_ratio
            n_r   = self.design_parameters.pressure_recovery
            M_out = self.design_parameters.exit_mach_number
            
            P_t_out, T_t_out = _inlet_stagnation(gas, T_t, P_t, M0, PR, n_r)
            A_out, u_out, P_out, T_out, h_t_out, h_out = _station_kinematic_design(
                gas=gas,
                T_t_out=T_t_out,
                P_t_out=P_t_out,
                M_design=M_out,
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
            M_out, u_out, P_t_out, T_t_out, P_out, T_out, h_t_out, h_out = _inlet_performance(
                gas=fs.atmosphere.fluid,
                T_t=fs.stagnation_temperature,
                P_t=fs.stagnation_pressure,
                M0=fs.mach_number,
                PR=self.design_parameters.pressure_ratio,
                n_r=self.design_parameters.pressure_recovery,
                mdot=jnp.atleast_2d(network_state.mass_flow_rate),
                A_exit=self.design_parameters.A_exit,
            )

        outputs = state.energy.nodes[self.network_ID].outputs.flow

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

def _comp_alpha_schedule(Nc, Nc_design):
    """
    Schedules alpha (inlet guide vane angle in degrees) according to rotation speed.
    """
    return jnp.where(Nc_design > 0.0, jnp.maximum(0.0, 90.0 - (Nc / Nc_design) * 90.0), jnp.zeros_like(Nc))

def _fan_compressor_performance(
        gas,
        T_t,
        P_t,
        PR,
        n_isn,
):
    """
    Computes power and output thermal quantities for a fan or compressor.
    """

    T_t_out, P_t_out = _average_gamma_stagnation(gas, T_t, P_t, PR, 1.0 / n_isn)

    # 3. Calculate work using absolute enthalpies
    h_t = gas.compute_enthalpy(T_t)
    h_t_out = gas.compute_enthalpy(T_t_out)
    d_power = h_t_out - h_t

    return d_power, P_t_out, T_t_out, h_t_out

@register
class Compressor(FlowNode):
    tag: str = init_field("Compressor", static=True)

    inputs: tuple | GraphInput =init_field(GraphInput("flow", "Inlet"), static=True)

    map: CompressorMap = init_field(maps.AXI5)

    alpha_schedule: Callable = init_field(_comp_alpha_schedule, as_value=True, static=True)

    def __post_init__(self):
        if not isinstance(self.map, CompressorMap):
            raise TypeError(f"'{self.tag}' requires a CompressorMap, got {type(self.map).__name__}")

    @tu.inputs(
        "state.freestream.Cp",
        "state.freestream.gamma",
        "state.energy.nodes[Compressor_flow_inputs].outputs.flow.stagnation_temperature",
        "state.energy.nodes[Compressor_flow_inputs].outputs.flow.stagnation_pressure",
        "system.energy.nodes[Compressor].pressure_ratio",
        "system.energy.nodes[Compressor].efficiencies.flow",
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
        
        T_t = self.average_domain_inputs(state, "flow", "stagnation_temperature")
        P_t = self.average_domain_inputs(state, "flow", "stagnation_pressure")

        theta_c = T_t / 288.15
        delta_c = P_t / 101325.0

        gas = self.working_fluid
        
        if design_mode:
            M_out   = self.design_parameters.exit_mach_number
            PR      = self.design_parameters.pressure_ratio
            n_isn   = self.efficiencies.flow

            N_des   = self.design_parameters.rotation_speed
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

            T_t_out, P_t_out = _average_gamma_stagnation(gas, T_t, P_t, PR, 1.0 / n_isn)

            A_out, u_out, P_out, T_out, h_t_out, h_out = _station_kinematic_design(
                gas=gas,
                T_t_out=T_t_out,
                P_t_out=P_t_out,
                M_design=M_out,
                mdot=network_state.mass_flow_rate
            )

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
                ),
                    updated_system,
                (
                    updated_design_paramters,
                    updated_map,
                    state.energy.mass_flow_rate,
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

        power, P_t_out, T_t_out, h_t_out = _fan_compressor_performance(
            gas=gas,
            T_t=T_t,
            P_t=P_t,
            PR=PR,
            n_isn=n_isn,
        )

        if settings.analysis.energy.design_mode:
            updated_system = eqx.tree_at(
                lambda s: s.energy.design_parameters.power,
                updated_system,
                power
            )

        outputs = state.energy.nodes[self.network_ID].outputs

        outputs = eqx.tree_at(lambda o: o.mechanical.power, outputs, jnp.atleast_2d(power))

        outputs = eqx.tree_at(lambda o: o.flow.mass_flow_rate, outputs, network_state.mass_flow_rate)
        outputs = eqx.tree_at(lambda o: o.flow.stagnation_pressure, outputs, P_t_out)
        outputs = eqx.tree_at(lambda o: o.flow.stagnation_temperature, outputs, T_t_out)
        outputs = eqx.tree_at(lambda o: o.flow.stagnation_enthalpy, outputs, h_t_out)

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
    
    h_t_in = gas.compute_absolute_enthalpy(T_t)
    P_t_out = P_t * PR

    # Target exit enthalpy based on the commanded exit temperature
    h_t_out = gas.compute_absolute_enthalpy(T_t_out)

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
    for _ in range(5):
        h_current = ox.compute_enthalpy(T_t_out)
        Cp_current = ox.compute_Cp(T_t_out)
        
        error = h_current - h_t_out
        
        # True Newton Step: x_new = x_old - f(x)/f'(x)
        T_t_out = T_t_out - (error / Cp_current)

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
        "system.energy.nodes[Combustor].efficiencies.flow",
    )
    @tu.outputs(
        "state.energy.nodes[Combustor].outputs.flow.stagnation_pressure",
        "state.energy.nodes[Combustor].outputs.flow.stagnation_temperature",
        "state.energy.nodes[Combustor].outputs.flow.stagnation_enthalpy",
        "state.energy.nodes[Combustor].outputs.fuel.fuel_air_ratio",
    )
    def transmit(self, state: State, system: System, settings: Settings):
        
        updated_system = system

        T_t=self.average_domain_inputs(state, "flow", "stagnation_temperature")
        P_t=self.average_domain_inputs(state, "flow", "stagnation_pressure")
        mdot_in=self.sum_domain_inputs(state, "flow", "mass_flow_rate")

        LHV=self.fuel.specific_energy
        PR=self.design_parameters.pressure_ratio
        n_b=self.efficiencies.flow

        if settings.analysis.energy.design_mode:
            T_t_out = jnp.atleast_2d(self.design_parameters.output_temperature)

            P_t_out, h_t_out, FAR, mdot_out = _combustor_design(
                gas=self.working_fluid,
                T_t=T_t,
                P_t=P_t,
                T_t_out=T_t_out,
                mdot_in=mdot_in,
                LHV=LHV,
                h_t_f=0.0,
                PR=PR,
                n_b=n_b,
            )

            A_out, u_out, P_out, T_out, h_t_out, h_out = _station_kinematic_design(
                gas=self.working_fluid,
                T_t_out=T_t_out,
                P_t_out=P_t_out,
                M_design=self.design_parameters.exit_mach_number,
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
                mdot_in=mdot_in,
                FAR=FAR,
                PR=PR,
                n_b=n_b
            )

        outputs = state.energy.nodes[self.network_ID].outputs

        outputs = eqx.tree_at(lambda o: o.flow.stagnation_pressure, outputs, P_t_out)
        outputs = eqx.tree_at(lambda o: o.flow.stagnation_temperature, outputs, T_t_out)
        outputs = eqx.tree_at(lambda o: o.flow.stagnation_enthalpy, outputs, h_t_out)
        outputs = eqx.tree_at(lambda o: o.flow.mass_flow_rate, outputs, mdot_out)

        outputs = eqx.tree_at(lambda o: o.fuel.fuel_air_ratio, outputs, FAR)

        updated_state = eqx.tree_at(lambda s: s.energy.nodes[self.network_ID].outputs, state, outputs)

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
    T_t_out, P_t_out = _average_gamma_stagnation(gas, T_t, P_t, 1.0 / PR, n_isn)

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

    map: TurbineMap = init_field(maps.LPT2269)

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

    @tu.inputs(
        "state.freestream.gamma",
        "state.freestream.Cp",
        "state.energy.nodes[Turbine_flow_inputs].outputs.flow.stagnation_temperature",
        "state.energy.nodes[Turbine_flow_inputs].outputs.flow.stagnation_pressure",
        "state.energy.nodes[Turbine_fuel_inputs].outputs.fuel.fuel_air_ratio",
        "state.energy.nodes[Turbine_mechanical_inputs].outputs.mechanical.work",
        "system.energy.nodes[Turbine].efficiencies.mechanical",
        "system.energy.nodes[Turbine].efficiencies.flow",
    )
    @tu.outputs(
        "state.energy.nodes[Turbine].outputs.flow.stagnation_temperature",
        "state.energy.nodes[Turbine].outputs.flow.stagnation_pressure",
        "state.energy.nodes[Turbine].outputs.flow.stagnation_enthalpy",
    )
    def transmit(self, state: State, system: System, settings: Settings):
        
        updated_system = system

        design_mode = settings.analysis.energy.design_mode
        network_state: TurbojetNetworkConditions = state.energy
        
        T_t = self.average_domain_inputs(state, "flow", "stagnation_temperature")
        P_t = self.average_domain_inputs(state, "flow", "stagnation_pressure")
        
        FAR = self.average_domain_inputs(state, "fuel", "fuel_air_ratio")
        gas = BurnedJetA(FAR)

        if design_mode:
            PR = jnp.atleast_2d(network_state.turbine_PR)
            n_isn = self.efficiencies.flow

            N_des = self.design_parameters.rotation_speed
            Np_des = N_des / jnp.sqrt(T_t)

            W = self.sum_domain_inputs(state, "flow", "mass_flow_rate")
            Wp_tgt = W * jnp.sqrt(T_t) / P_t

            Wp_map, eff_map = self.map.evaluate(
                alpha=self.map.alpha_des,
                Np=self.map.Np_des,
                PR=self.map.PR_des)
            
            s_Wp = (Wp_tgt / Wp_map).squeeze()
            s_PR = ((PR - 1.0)/(self.map.PR_des - 1.0)).squeeze()
            s_eff = (self.efficiencies.flow / eff_map).squeeze()
            s_Np = (Np_des/self.map.Np_des).squeeze()

            # Turbine passes 1 / PR to reflect pressure drop
            T_t_out, P_t_out = _average_gamma_stagnation(gas, T_t, P_t, 1.0 / PR, n_isn)
            A_out, u_out, P_out, T_out, h_t_out, h_out = _station_kinematic_design(
                gas=BurnedJetA(FAR),
                T_t_out=T_t_out,
                P_t_out=P_t_out,
                M_design=self.design_parameters.exit_mach_number,
                mdot=network_state.mass_flow_rate
            )

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
            n_mech=self.efficiencies.mechanical,
            T_t=T_t,
            P_t=P_t,
        )

        # Set Output State
        outputs = state.energy.nodes[self.network_ID].outputs

        outputs = eqx.tree_at(lambda o: o.flow.stagnation_temperature, outputs, T_t_out)
        outputs = eqx.tree_at(lambda o: o.flow.stagnation_pressure, outputs, P_t_out)
        outputs = eqx.tree_at(lambda o: o.flow.stagnation_enthalpy, outputs, h_t_out)
        outputs = eqx.tree_at(lambda o: o.flow.mass_flow_rate, outputs, network_state.mass_flow_rate * (1 + FAR))

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
    h_t_out = gas.compute_absolute_enthalpy(T_t_out)
    h_out = gas.compute_absolute_enthalpy(T_out)
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
    M_exit_sup = 2.0
    for _ in range(5):
        term = (2.0 / (gamma + 1.0)) * (1.0 + (gamma - 1.0) / 2.0 * M_exit_sup**2)
        power = (gamma + 1.0) / (2.0 * (gamma - 1.0))
        AR_calc = (1.0 / M_exit_sup) * (term ** power)
        
        # Analytical derivative: d(A/A*) / dM
        dAR_dM = AR_calc * (M_exit_sup**2 - 1.0) / (M_exit_sup * (1.0 + (gamma - 1.0) / 2.0 * M_exit_sup**2))
        
        # Newton step
        M_exit_sup = jnp.maximum(M_exit_sup - (AR_calc - AR) / dAR_dM, 1.001)
    
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
    
    h_t_out = gas.compute_absolute_enthalpy(T_t_out)
    h_out = gas.compute_absolute_enthalpy(T_out)
    u_out = jnp.sqrt(2.0 * (h_t_out - h_out)) * n_v

    rho_out = P_out / (R * T_out)

    return mdot_out, M_exit, u_out, rho_out, P_out, P_t_out, T_out, T_t_out, h_out, h_t_out

@register
class FixedNozzle(FlowNode):
    tag: str = init_field("Core Nozzle", static=True)

    inputs: tuple = (
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
        "system.energy.nodes[ExpansionNozzle].efficiencies.flow",
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
        
        FAR = self.average_domain_inputs(state, "fuel", "fuel_air_ratio")
        working_fluid = BurnedJetA(FAR)

        T_t = self.average_domain_inputs(state, "flow", "stagnation_temperature")
        P_t = self.average_domain_inputs(state, "flow", "stagnation_pressure")
        
        if settings.analysis.energy.design_mode:
            
            mdot_out=state.energy.mass_flow_rate * (1. + FAR)

            A_t, A_x, M_out, rho_out, u_out, P_out, P_t_out, T_out, T_t_out, h_out, h_t_out = _nozzle_design(
                gas=working_fluid,
                T_t=T_t,
                P_t=P_t,
                mdot=mdot_out,
                P0=P0,
                PR=self.design_parameters.pressure_ratio,
                n_v=self.efficiencies.flow,
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
                gas=working_fluid,
                T_t=T_t,
                P_t=P_t,
                P0=P0,
                A_throat=A_t,
                A_exit=A_x,
                n_v=self.efficiencies.flow,
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
    
    h_t_out = gas.compute_absolute_enthalpy(T_t_out)
    h_out = gas.compute_absolute_enthalpy(T_out)
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
        "system.energy.nodes[ExpansionNozzle].efficiencies.flow",
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
        
        FAR = self.average_domain_inputs(state, "fuel", "fuel_air_ratio")
        working_fluid = BurnedJetA(FAR)

        T_t = self.average_domain_inputs(state, "flow", "stagnation_temperature")
        P_t = self.average_domain_inputs(state, "flow", "stagnation_pressure")

        mdot=state.energy.mass_flow_rate * (1. + FAR)
        
        if settings.analysis.energy.design_mode:
            A_t, A_x, M_out, rho_out, u_out, P_out, P_t_out, T_out, T_t_out, h_out, h_t_out = _nozzle_design(
                gas=working_fluid,
                T_t=T_t,
                P_t=P_t,
                mdot=mdot,
                P0=P0,
                PR=self.design_parameters.pressure_ratio,
                n_v=self.efficiencies.flow,
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
                    mdot_in=mdot,
                    n_v=self.efficiencies.flow,
                )
            )

        # Physical outflow
        outputs = state.energy.nodes[self.network_ID].outputs.flow

        outputs = eqx.tree_at(lambda o: o.area, outputs, A_x)
        outputs = eqx.tree_at(lambda o: o.mass_flow_rate, outputs, mdot)
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
            d_power = (self.sum_domain_inputs(state, "mechanical", "power") / 2e7)
        else:
            d_power = (self.sum_domain_inputs(state, "mechanical", "power") / 
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
        "ducts": GraphSplitter,
        }, static=True
    )

    @classmethod
    def build_custom(
        cls,
        variable_nozzle: bool = True,
        **kwargs
    ):
        
        inlet = Inlet()
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
        mdot_core = self.sum_domain_inputs(state, "flow", "mass_flow_rate")

        F, F_sp, I_sp, TSFC, p, ff = _engine_performance(
            u0=fs.speed,
            P0=fs.pressure,
            g=fs.gravity,
            delta_SFC=self.design_parameters.delta_SFC,
            v_fan_nozzle=0.0,
            A_fan_nozzle=0.0,
            P_fan_nozzle=0.0,
            v_core_nozzle=self.average_domain_inputs(state, "flow", "speed"),
            A_core_nozzle=self.average_domain_inputs(state, "flow", "area"),
            P_core_nozzle=self.average_domain_inputs(state, "flow", "pressure"),
            fuel_air_ratio=self.average_domain_inputs(state, "fuel", "fuel_air_ratio"),
            mdot_core=mdot_core,
            BPR=0.0,
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
            lambda o: o.residual.power, outputs, self.sum_domain_inputs(state, "residual", "power")
        )
        outputs = eqx.tree_at(
            lambda o: o.residual.mass_flow_rate, outputs, self.sum_domain_inputs(state, "residual", "mass_flow_rate")
        )

        updated_state = eqx.tree_at(lambda s: s.energy.nodes[self.network_ID].outputs, state, outputs)

        return updated_state, system, settings

def _TurbofanSetup(BPR: float):

    alpha = BPR / (1.0 + BPR)
    beta = 1.0 / (1.0 + BPR)

    inlet = Inlet()
    fan = Compressor(tag="Fan", map=maps.Fan)

    fan_flow = GraphSplitter(tag="Bypass Duct", inputs=GraphInput("flow", "fan"), split_fraction=alpha)
    core_flow = GraphSplitter(tag="Core Duct", inputs=GraphInput("flow", "fan"), split_fraction=beta)

    lpc = Compressor(tag="LPC", map=maps.LPC, inputs=GraphInput("flow", "core duct"))
    hpc = Compressor(tag="HPC", map=maps.HPC, inputs=GraphInput("flow", "lpc"))
    
    comb = Combustor(inputs=GraphInput("flow", "hpc"))
    
    hpt = Turbine(tag="HPT", map=maps.HPT)
    lpt = Turbine(tag="LPT", map=maps.LPT, inputs=(GraphInput("flow", "hpt"), GraphInput("fuel", "combustor")))
    
    lp_shaft = Turboshaft(tag="Low Power Shaft", inputs=(
            GraphInput("mechanical", "lpc"),
            GraphInput("mechanical", "fan"),
            GraphInput("mechanical", "lpt"),
        )
    )
    
    hp_shaft = Turboshaft(tag="High Power Shaft", inputs=(
            GraphInput("mechanical", "hpc"),
            GraphInput("mechanical", "hpt"),
        )
    )
    
    f_nozz = VariableNozzle(tag="Fan Nozzle", inputs=(GraphInput("flow", "bypass duct")))
    c_nozz = VariableNozzle(inputs=(GraphInput("flow", "lpt"), GraphInput("fuel", "combustor")))

    return (inlet, fan, core_flow, fan_flow, lpc, hpc, comb, hpt, lpt, lp_shaft, hp_shaft, f_nozz, c_nozz)

def TurbofanEngine(BPR: float = 4.0, **kwargs):
    return TurbojetEngine(
        subcomponents=_TurbofanSetup(BPR),
        inputs = (None,), # TODO: Setup Nozzle and Shaft Inputs
        **kwargs)

    
# ----------------------------------------------------------------------------------------------------------------------
# Turbofan Engine
# ----------------------------------------------------------------------------------------------------------------------


# class Fan(FlowNode):
#     tag: str = init_field("Fan", static=True)
#     inputs: tuple = init_field((EnergyInput("flow", "Inlet Nozzle"),), static=True)

#     map: CompressorMap = init_field(maps.Fan)

#     @tu.inputs(
#         "state.freestream.Cp",
#         "state.freestream.gamma",
#         "state.energy.nodes[Fan_flow_inputs].outputs.flow.stagnation_temperature",
#         "state.energy.nodes[Fan_flow_inputs].outputs.flow.stagnation_pressure",
#         "system.energy.nodes[Fan].pressure_ratio",
#         "system.energy.nodes[Fan].efficiencies.flow",
#     )
#     @tu.outputs(
#         "state.energy.nodes[Fan].outputs.flow.stagnation_pressure",
#         "state.energy.nodes[Fan].outputs.flow.stagnation_temperature",
#         "state.energy.nodes[Fan].outputs.flow.stagnation_enthalpy",
#         "state.energy.nodes[Fan].outputs.mechanical.work",
#     )
#     def transmit(self, state: State, system: System, settings: Settings):

#         power, P_t_out, T_t_out, h_t_out = fan_compressor_performance(
#             gas=self.working_fluid,
#             T_t=self.average_domain_inputs(state, "flow", "stagnation_temperature"),
#             P_t=self.average_domain_inputs(state, "flow", "stagnation_pressure"),
#             PR=self.pressure_ratio,
#             n_p=self.efficiencies.flow,
#         )

#         # Set Output State
#         outputs = state.energy.nodes[self.network_ID].outputs

#         outputs = eqx.tree_at(lambda o: o.mechanical.power, outputs, power)

#         outputs = eqx.tree_at(lambda o: o.flow.stagnation_pressure, outputs, P_t_out)
#         outputs = eqx.tree_at(lambda o: o.flow.stagnation_temperature, outputs, T_t_out)
#         outputs = eqx.tree_at(lambda o: o.flow.stagnation_enthalpy, outputs, h_t_out)

#         updated_state = eqx.tree_at(lambda s: s.energy.nodes[self.network_ID].outputs, state, outputs)

#         return updated_state, system, settings


# def _TurbofanSetup(BPR):

#     inlet = InletNozzle()
#     fan = Fan()

#     core_flow = EnergySplitter(tag="Core Duct", inputs=(EnergyInput("flow", "Fan"),), extraction_fraction=1.0 / (1.0 + BPR))
#     bypass_flow = EnergySplitter(tag="Bypass Duct", inputs=(EnergyInput("flow", "Fan"),), extraction_fraction=BPR / (1.0 + BPR))

#     LPC = Compressor(tag="LPC", flow_inputs=("Core Duct",))
#     HPC = Compressor(tag="HPC", flow_inputs=("LPC",))

#     comb = TurbojetCombustor()

#     HPT = Turbine(tag="HPT", mechanical_inputs=("HPC",), flow_inputs=("Combustor",))
#     LPT = Turbine(tag="LPT", mechanical_inputs=("LPC", "Fan"), flow_inputs=("HPT",))

#     core_nozz = ExpansionNozzle(tag="Core Nozzle", flow_inputs=("LPT",))
#     fan_nozz = ExpansionNozzle(tag="Fan Nozzle", flow_inputs=("Bypass Duct",))

#     return (inlet, fan, core_flow, bypass_flow, LPC, HPC, comb, HPT, LPT, core_nozz, fan_nozz)


# class TurbofanEngine(TurbojetEngine):
#     tag: str = init_field("Turbofan", static=True)

#     bypass_ratio: float = 1.0
#     exa: float = 1.0  # Fan Face-to-Exit Distance

#     def __post_init__(self):
#         object.__setattr__(self, "subcomponents", _TurbofanSetup(self.bypass_ratio))
#         # super(TurbofanEngine, self).__post_init__()

#     @tu.inputs(
#         "state.freestream.gamma",
#         "state.freestream.speed",
#         "state.freestream.speed_of_sound",
#         "state.freestream.mach_number",
#         "state.freestream.pressure",
#         "state.freestream.gravity",
#         "state.energy.nodes[Turbofan].throttle",
#         "state.energy.nodes[Turbofan_core_nozzle].outputs.flow.speed",
#         "state.energy.nodes[Turbofan_core_nozzle].outputs.flow.area_ratio",
#         "state.energy.nodes[Turbofan_core_nozzle].outputs.flow.pressure",
#         "state.energy.nodes[Turbofan_fan_nozzle].outputs.flow.speed",
#         "state.energy.nodes[Turbofan_fan_nozzle].outputs.flow.area_ratio",
#         "state.energy.nodes[Turbofan_fan_nozzle].outputs.flow.pressure",
#         "state.energy.nodes[Turbofan_combustor].outputs.fuel.fuel_air_ratio",
#         "system.energy.nodes[Turbofan].bypass_ratio",
#     )
#     @tu.outputs(
#         "state.energy.nodes[Turbofan].outputs.force.thrust",
#         "state.energy.nodes[Turbofan].outputs.force.nondimensional_thrust",
#         "state.energy.nodes[Turbofan].outputs.force.specific_impulse",
#         "state.energy.nodes[Turbofan].outputs.fuel.TSFC",
#         "state.energy.nodes[Turbofan].outputs.fuel.flow_rate",
#         "state.energy.nodes[Turbofan].outputs.flow.mass_flow_rate",
#         "state.energy.nodes[Turbofan].outputs.mechanical.power",
#     )
#     def transmit(self, state: State, system: System, settings: Settings):

#         cn_out = state.energy.nodes[self.network_ID + ".core_nozzle"].outputs.flow
#         fn_out = state.energy.nodes[self.network_ID + ".fan_nozzle"].outputs.flow
#         comb_out = state.energy.nodes[self.network_ID + ".combustor"].outputs.fuel

#         fs = state.freestream

#         F, F_sp, I_sp, TSFC, mdot_c, p, ff = func_thrust_and_power(
#             gamma=fs.gamma,
#             u0=fs.speed,
#             a0=fs.speed_of_sound,
#             M0=fs.mach_number,
#             P0=fs.pressure,
#             g=fs.gravity,
#             F_ref=self.design_parameters.total_thrust,
#             delta_SFC=self.design_parameters.delta_SFC,
#             v_fan_nozzle=fn_out.speed,
#             AR_fan_nozzle=fn_out.area_ratio,
#             P_fan_nozzle=fn_out.pressure,
#             v_core_nozzle=cn_out.speed,
#             AR_core_nozzle=cn_out.area_ratio,
#             P_core_nozzle=cn_out.pressure,
#             fuel_air_ratio=comb_out.fuel_air_ratio,
#             BPR=self.bypass_ratio,
#             throttle=state.energy.throttle,
#         )

#         outputs = state.energy.nodes[self.network_ID].outputs

#         outputs = eqx.tree_at(lambda o: o.force.thrust, outputs, F)
#         outputs = eqx.tree_at(lambda o: o.force.nondimensional_thrust, outputs, F_sp)
#         outputs = eqx.tree_at(lambda o: o.force.specific_impulse, outputs, I_sp)

#         outputs = eqx.tree_at(lambda o: o.fuel.TSFC, outputs, TSFC)
#         outputs = eqx.tree_at(lambda o: o.fuel.flow_rate, outputs, ff)

#         outputs = eqx.tree_at(lambda o: o.flow.mass_flow_rate, outputs, mdot_c)

#         outputs = eqx.tree_at(lambda o: o.mechanical.power, outputs, p)

#         updated_state = eqx.tree_at(lambda s: s.energy.nodes[self.network_ID].outputs, state, outputs)

#         return updated_state, system, settings
