# RCAIDE/Library/Components/Energy/Propulsors.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: May 2025, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING, Callable


if TYPE_CHECKING:
    from src.eden_trace.framework import Settings, State, System
    from src.eden_trace.framework.conditions.Energy import TurbojetNetworkConditions


# package imports
import equinox as eqx
import jax.numpy as jnp

import src.eden_trace.utils as ru

# RCAIDE imports
from src.eden_trace.utils import init_field, register

from src.eden_trace.library.components.energy import maps
from src.eden_trace.library.components.energy.maps import CompressorMap, TurbineMap
from src.eden_trace.library.components.energy.nodes import EnergyInput, EnergyNode, FlowNode
from src.eden_trace.library.gases import Air, BurnedJetA, IdealGas
from src.eden_trace.library.methods.energy.Transmission.Combustors import (
    func_combustor_design as combustor_design,
    func_combustor_performance as combustor_performance,
)
from src.eden_trace.library.methods.energy.Transmission.Fan_Compressors import func_fan_compressor_performance as fan_compressor_performance
from src.eden_trace.library.methods.energy.Transmission.Nozzles import (
    func_inlet_design as inlet_design,
    func_inlet_performance as inlet_performance,
    func_nozzle_design as nozzle_design,
    func_nozzle_performance as nozzle_performance,
    func_variable_nozzle_performance as variable_nozzle_performance
)
from src.eden_trace.library.methods.energy.Transmission.Turbines import func_turbine_performance as turbine_performance
from src.eden_trace.library.methods.energy.Transmission.Turbofans import func_thrust_and_power as engine_performance
from src.eden_trace.library.propellants import JetA, Propellant

# ----------------------------------------------------------------------------------------------------------------------
# Turbojet Components
# ----------------------------------------------------------------------------------------------------------------------
@register
class InletNozzle(FlowNode):
    tag: str = init_field("Inlet Nozzle", static=True)

    @ru.inputs(
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
    @ru.outputs(
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
            A_exit, M_out, u_out, P_t_out, T_t_out, P_out, T_out, h_t_out, h_out = inlet_design(
                gas=fs.atmosphere.fluid,
                T_t=fs.stagnation_temperature,
                P_t=fs.stagnation_pressure,
                M0=fs.mach_number,
                mdot=jnp.atleast_2d(network_state.mass_flow_rate),
                PR=self.design_parameters.pressure_ratio,
                n_r=self.design_parameters.pressure_recovery,
                M_design=self.design_parameters.exit_mach_number,
            )

            updated_design_paramters = eqx.tree_at(
                lambda d: d.A_exit,
                self.design_parameters,
                A_exit.squeeze()
            )
            updated_system = eqx.tree_at(
                lambda s: s.energy.nodes[self.network_ID].design_parameters,
                updated_system,
                updated_design_paramters
            )

        else:
            M_out, u_out, P_t_out, T_t_out, P_out, T_out, h_t_out, h_out = inlet_performance(
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


def _comp_alpha_schedule(Nc, Nc_design):
    return jnp.where(Nc_design > 0.0, jnp.maximum(0.0, 90.0 - (Nc / Nc_design) * 90.0), jnp.zeros_like(Nc))

@register
class Compressor(FlowNode):
    tag: str = init_field("Compressor", static=True)

    inputs: tuple=init_field((EnergyInput("flow", "Inlet Nozzle"),))

    map: CompressorMap = init_field(maps.AXI5)

    alpha_schedule: Callable = init_field(_comp_alpha_schedule, as_value=True, static=True)

    @ru.inputs(
        "state.freestream.Cp",
        "state.freestream.gamma",
        "state.energy.nodes[Compressor_flow_inputs].outputs.flow.stagnation_temperature",
        "state.energy.nodes[Compressor_flow_inputs].outputs.flow.stagnation_pressure",
        "system.energy.nodes[Compressor].pressure_ratio",
        "system.energy.nodes[Compressor].efficiencies.flow",
    )
    @ru.outputs(
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

        theta_c = T_t/288.15
        delta_c = P_t/101325.0
        
        if design_mode:
            PR  = self.design_parameters.pressure_ratio
            n_isn = self.efficiencies.flow

            W   = jnp.atleast_2d(network_state.mass_flow_rate)
            Wc_tgt = W * jnp.sqrt(theta_c) / delta_c
            
            PR_map, Wc_map, eff_map = self.map.evaluate(
                alpha=0.0,
                Nc=1.0,
                Rline=self.map.Rline_des
            )
            
            s_Wc =  (Wc_tgt / Wc_map).squeeze()
            s_PR = (PR - 1.0)/(PR_map - 1.0)
            s_eff = self.efficiencies.flow / eff_map

            updated_map = eqx.tree_at(
                lambda m: (m.s_Wc, m.s_PR, m.s_eff, m.s_Nc),
                self.map,
                (s_Wc, s_PR, s_eff, self.design_parameters.rotation_speed)
            )

            updated_system = eqx.tree_at(
                lambda s: s.energy.nodes[self.network_ID].map,
                updated_system,
                updated_map
            )

        else:
            N      = jnp.atleast_2d(network_state.rotation_speed)
            Nc_des = self.design_parameters.rotation_speed
            Nc     = N / jnp.sqrt(T_t / 288.15)
            
            Rline   = jnp.atleast_2d(network_state.Rline)
            alpha   = self.alpha_schedule(Nc, Nc_des)

            PR, Wc, n_isn = self.map.evaluate(alpha, Nc, Rline)
            W = Wc * delta_c / jnp.sqrt(theta_c)

        d_work, P_t_out, T_t_out, h_t_out = fan_compressor_performance(
            gas=self.working_fluid,
            T_t=T_t,
            P_t=P_t,
            PR=PR,
            n_isn=n_isn,
        )

        outputs = state.energy.nodes[self.network_ID].outputs

        outputs = eqx.tree_at(lambda o: o.mechanical.work, outputs, jnp.atleast_2d(d_work))

        outputs = eqx.tree_at(lambda o: o.flow.mass_flow_rate, outputs, network_state.mass_flow_rate)
        outputs = eqx.tree_at(lambda o: o.flow.stagnation_pressure, outputs, P_t_out)
        outputs = eqx.tree_at(lambda o: o.flow.stagnation_temperature, outputs, T_t_out)
        outputs = eqx.tree_at(lambda o: o.flow.stagnation_enthalpy, outputs, h_t_out)

        updated_state = eqx.tree_at(lambda s: s.energy.nodes[self.network_ID].outputs, state, outputs)

        # Residual Update
        updated_state = eqx.tree_at(
            lambda s:s.energy.outputs.residual.Wc,
            updated_state,
            (W - state.energy.mass_flow_rate)/100.
        )

        return updated_state, updated_system, settings

@register
class TurbojetCombustor(FlowNode):
    tag: str = init_field("Combustor", static=True)

    inputs: tuple = init_field((EnergyInput("flow", "Compressor"),), static=True)
    fuel: Propellant = init_field(JetA)

    @ru.inputs(
        "state.freestream.Cp",
        "state.energy.nodes[Combustor_flow_inputs].outputs.flow.stagnation_temperature",
        "state.energy.nodes[Combustor_flow_inputs].outputs.flow.stagnation_pressure",
        "system.energy.nodes[Turbojet].design_paramters.turbine_intake_temperature",
        "system.energy.nodes[Turbojet].fuel.specific_energy",
        "system.energy.nodes[Combustor].pressure_ratio",
        "system.energy.nodes[Combustor].efficiencies.flow",
    )
    @ru.outputs(
        "state.energy.nodes[Combustor].outputs.flow.stagnation_pressure",
        "state.energy.nodes[Combustor].outputs.flow.stagnation_temperature",
        "state.energy.nodes[Combustor].outputs.flow.stagnation_enthalpy",
        "state.energy.nodes[Combustor].outputs.fuel.fuel_air_ratio",
    )
    def transmit(self, state: State, system: System, settings: Settings):
        
        T_t=self.average_domain_inputs(state, "flow", "stagnation_temperature")
        P_t=self.average_domain_inputs(state, "flow", "stagnation_pressure")
        mdot_in=self.sum_domain_inputs(state, "flow", "mass_flow_rate")

        LHV=self.fuel.specific_energy
        PR=self.design_parameters.pressure_ratio
        n_b=self.efficiencies.flow

        if settings.analysis.energy.design_mode:
            T_t_out = jnp.atleast_2d(self.design_parameters.output_temperature)

            P_t_out, h_t_out, FAR, mdot_out = combustor_design(
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
        
        else:    
            FAR = state.energy.fuel_air_ratio

            P_t_out, T_t_out, h_t_out, mdot_out = combustor_performance(
                gas=self.working_fluid,
                T_t=T_t,
                P_t=P_t,
                mdot_in=mdot_in,
                FAR=FAR,
                LHV=LHV,
                h_t_f=0.0,
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

        return updated_state, system, settings

@register
class Turbine(FlowNode):
    tag: str = init_field("Turbine", static=True)

    map: TurbineMap = init_field(maps.LPT2269)

    alpha_schedule: Callable = init_field(lambda Np, Np_des: jnp.full_like(Np, 1.0), as_value=True, static=True)

    inputs: tuple = init_field(
        (
            EnergyInput("flow", "Combustor"),
            EnergyInput("fuel", "Combustor"),
        ), static=True,
    )

    @ru.inputs(
        "state.freestream.gamma",
        "state.freestream.Cp",
        "state.energy.nodes[Turbine_flow_inputs].outputs.flow.stagnation_temperature",
        "state.energy.nodes[Turbine_flow_inputs].outputs.flow.stagnation_pressure",
        "state.energy.nodes[Turbine_fuel_inputs].outputs.fuel.fuel_air_ratio",
        "state.energy.nodes[Turbine_mechanical_inputs].outputs.mechanical.work",
        "system.energy.nodes[Turbine].efficiencies.mechanical",
        "system.energy.nodes[Turbine].efficiencies.flow",
    )
    @ru.outputs(
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

        theta_t = T_t/288.15
        delta_t = P_t/101325.0
        
        FAR = self.average_domain_inputs(state, "fuel", "fuel_air_ratio")

        if design_mode:
            PR = jnp.atleast_2d(network_state.turbine_PR)
            n_isn = self.efficiencies.flow

            W = self.sum_domain_inputs(state, "flow", "mass_flow_rate")
            Wp_tgt = W * jnp.sqrt(theta_t) / delta_t

            Wp_map, eff_map = self.map.evaluate(
                alpha=jnp.zeros_like(PR),
                Np=jnp.ones_like(PR),
                PR=PR)
            
            s_Wp = (Wp_tgt / Wp_map).squeeze()
            s_eff = (self.efficiencies.flow / eff_map).squeeze()

            updated_map = eqx.tree_at(
                lambda m: (m.s_Wp, m.s_PR, m.s_eff, m.s_Np),
                self.map,
                (s_Wp, PR, s_eff, self.design_parameters.rotation_speed)
            )

            updated_system = eqx.tree_at(
                lambda s: s.energy.nodes[self.network_ID].map,
                updated_system,
                updated_map
            )

        else:
            N = jnp.atleast_2d(network_state.rotation_speed)
            Np = N / jnp.sqrt(T_t / 288.15)
            Np_des = self.design_parameters.rotation_speed

            PR = jnp.atleast_2d(network_state.turbine_PR)
            alpha = self.alpha_schedule(Np, Np_des)

            Wp, n_isn = self.map.evaluate(alpha, Np, PR)
            W = Wp * delta_t / jnp.sqrt(theta_t) * (1. + FAR)

        T_t_out, P_t_out, h_t_out, work = turbine_performance(
            gas=BurnedJetA(FAR),
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

        outputs = eqx.tree_at(lambda o: o.mechanical.work, outputs, jnp.atleast_2d(work))

        updated_state = eqx.tree_at(lambda s: s.energy.nodes[self.network_ID].outputs, state, outputs)

        # Residual Update
        updated_state = eqx.tree_at(
            lambda s:s.energy.outputs.residual.Wp,
            updated_state,
            (W / (1. + FAR) - state.energy.mass_flow_rate)/100.
        )

        return updated_state, updated_system, settings

@register
class FixedNozzle(FlowNode):
    tag: str = init_field("Nozzle", static=True)

    @ru.inputs(
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
    @ru.outputs(
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

            A_t, A_x, M_out, rho_out, u_out, P_out, P_t_out, T_out, T_t_out, h_out, h_t_out = nozzle_design(
                gas=working_fluid,
                T_t=T_t,
                P_t=P_t,
                mdot=mdot_out,
                P0=P0,
                PR=self.design_parameters.pressure_ratio,
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
            
            mdot_out, M_out, u_out, rho_out, P_out, P_t_out, T_out, T_t_out, h_out, h_t_out = nozzle_performance(
                gas=working_fluid,
                T_t=T_t,
                P_t=P_t,
                P0=P0,
                A_throat=A_t,
                A_exit=A_x,
            )

        # Physical outflow
        outputs = state.energy.nodes[self.network_ID].outputs.flow

        outputs = eqx.tree_at(lambda o: o.area_ratio, outputs, A_x / A_t)
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
            (mdot_out / (1. + FAR) - state.energy.mass_flow_rate)/100.
        )

        return updated_state, updated_system, settings
    
# @register
# class VariableNozzle(FlowNode):
#     tag: str = init_field("Nozzle", static=True)

#     @ru.inputs(
#         "state.freestream.stagnation_temperature",
#         "state.freestream.stagnation_pressure",
#         "state.freestream.pressure",
#         "state.freestream.mach_number",
#         "state.freestream.Cp",
#         "state.freestream.gamma",
#         "state.freestream.R",
#         "system.energy.nodes[ExpansionNozzle].pressure_ratio",
#         "system.energy.nodes[ExpansionNozzle].efficiencies.flow",
#     )
#     @ru.outputs(
#         "state.energy.nodes[ExpansionNozzle].outputs.flow",
#         "state.energy.nodes[ExpansionNozzle].outputs.flow.area_ratio",
#         "state.energy.nodes[ExpansionNozzle].outputs.flow.mach_number",
#         "state.energy.nodes[ExpansionNozzle].outputs.flow.density",
#         "state.energy.nodes[ExpansionNozzle].outputs.flow.speed",
#         "state.energy.nodes[ExpansionNozzle].outputs.flow.pressure",
#         "state.energy.nodes[ExpansionNozzle].outputs.flow.stagnation_pressure",
#         "state.energy.nodes[ExpansionNozzle].outputs.flow.temperature",
#         "state.energy.nodes[ExpansionNozzle].outputs.flow.stagnation_temperature",
#         "state.energy.nodes[ExpansionNozzle].outputs.flow.enthalpy",
#         "state.energy.nodes[ExpansionNozzle].outputs.flow.stagnation_enthalpy",
#     )
#     def transmit(self, state: State, system: System, settings: Settings):

#         updated_system = system
        
#         fs = state.freestream
#         P0 = fs.pressure
        
#         FAR = self.average_domain_inputs(state, "fuel", "fuel_air_ratio")
#         working_fluid = BurnedJetA(FAR)

#         T_t = self.average_domain_inputs(state, "flow", "stagnation_temperature")
#         P_t = self.average_domain_inputs(state, "flow", "stagnation_pressure")
        
#         if settings.analysis.energy.design_mode:
            
#             mdot_out=state.energy.mass_flow_rate * (1. + FAR)

#             A_t, A_x, M_out, rho_out, u_out, P_out, P_t_out, T_out, T_t_out, h_out, h_t_out = nozzle_design(
#                 gas=working_fluid,
#                 T_t=T_t,
#                 P_t=P_t,
#                 mdot=mdot_out,
#                 P0=P0,
#                 PR=self.design_parameters.pressure_ratio,
#             )
            
#             updated_design_parameters = eqx.tree_at(lambda d:(
#                     d.A_throat,
#                     d.A_exit,
#                     d.A_ratio
#                 ), self.design_parameters,(
#                     A_t.squeeze(),
#                     A_x.squeeze(),
#                     A_x.squeeze()/A_t.squeeze()
#                 ),
#             )
#             updated_system = eqx.tree_at(
#                 lambda s: s.energy.nodes[self.network_ID].design_parameters,
#                 updated_system,
#                 updated_design_parameters
#             )
        
#         else:
#             A_t = self.design_parameters.A_throat
#             A_x = self.design_parameters.A_exit
            
#             mdot_in, M_exit, u_out, rho_out, P_out, P_t_out, T_out, T_t_out, h_out, h_t_out, A_throat, A_exit = variable_nozzle_performance(
#                 gas=working_fluid,
#                 T_t=T_t,
#                 P_t=P_t,
#                 P0=P0,
#                 A_throat=A_t,
#                 A_exit=A_x,
#             )

#         # Physical outflow
#         outputs = state.energy.nodes[self.network_ID].outputs.flow

#         outputs = eqx.tree_at(lambda o: o.area_ratio, outputs, A_x / A_t)
#         outputs = eqx.tree_at(lambda o: o.mass_flow_rate, outputs, mdot_out)
#         outputs = eqx.tree_at(lambda o: o.mach_number, outputs, M_out)
#         outputs = eqx.tree_at(lambda o: o.density, outputs, rho_out)
#         outputs = eqx.tree_at(lambda o: o.speed, outputs, u_out)
#         outputs = eqx.tree_at(lambda o: o.pressure, outputs, P_out)
#         outputs = eqx.tree_at(lambda o: o.stagnation_pressure, outputs, P_t_out)
#         outputs = eqx.tree_at(lambda o: o.temperature, outputs, T_out)
#         outputs = eqx.tree_at(lambda o: o.stagnation_temperature, outputs, T_t_out)
#         outputs = eqx.tree_at(lambda o: o.enthalpy, outputs, h_out)
#         outputs = eqx.tree_at(lambda o: o.stagnation_enthalpy, outputs, h_t_out)

#         updated_state = eqx.tree_at(lambda s: s.energy.nodes[self.network_ID].outputs.flow, state, outputs)

#         # Residual update
#         updated_state = eqx.tree_at(
#             lambda s: s.energy.outputs.residual.mass_flow_rate,
#             updated_state,
#             (mdot_out / (1. + FAR) - state.energy.mass_flow_rate)/100.
#         )

#         return updated_state, updated_system, settings

@register
class Turboshaft(EnergyNode):
    tag: str = init_field("Turboshaft", static=True)

    inputs: tuple = (
        EnergyInput("flow", "compressor"),
        EnergyInput("flow", "turbine"),
        EnergyInput("mechanical", "compressor"),
        EnergyInput("mechanical", "turbine"),
    )

    def transmit(self, state: State, system: System, settings: Settings):
        
        # c_norm = [i for i in self.inputs if "compressor" in i.network_ID and i.domain == "mechanical"][0]

        d_work = (self.sum_domain_inputs(state, "mechanical", "work") / 2e7)
        if settings.analysis.energy.design_mode:
            d_mass = jnp.atleast_2d(0.)
        else:
            d_mass = (self.diff_domain_inputs(state, "flow", "mass_flow_rate") / 
                    self.average_domain_inputs(state, "flow", "mass_flow_rate"))
        
        outputs = state.energy.nodes[self.network_ID].outputs

        outputs = eqx.tree_at(lambda o: o.residual.work, outputs, d_work)
        outputs = eqx.tree_at(lambda o: o.residual.mass_flow_rate, outputs, d_mass)

        updated_state = eqx.tree_at(lambda s: s.energy.nodes[self.network_ID].outputs, state, outputs)


        return updated_state, system, settings

# ----------------------------------------------------------------------------------------------------------------------
# Turbojet Engine
# ----------------------------------------------------------------------------------------------------------------------

def _TurbojetSetup():

    inlet = InletNozzle()
    comp = Compressor()
    comb = TurbojetCombustor()
    turb = Turbine()
    bal = Turboshaft()

    nozz = FixedNozzle(
        tag="Core Nozzle",
        inputs=(
            EnergyInput("flow", "Turbine"),
            EnergyInput("fuel", "Combustor"),
        ),
    )

    return (inlet, comp, comb, turb, bal, nozz)

@register
class JetGeometry(eqx.Module):
    xe: float = 1.0
    ye: float = 1.0
    Ce: float = 2.0

@register
class JetDesign(eqx.Module):
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
class TurbojetEngine(FlowNode):
    tag: str = init_field("Turbojet", static=True)
    subcomponents: tuple = init_field(_TurbojetSetup)

    plug_diameter: float = 0.0

    working_fluid: IdealGas = init_field(Air)
    design_parameters: JetDesign = init_field(JetDesign)

    inputs: tuple = init_field(
        (
            EnergyInput("flow", "self.core_nozzle"),
            EnergyInput("fuel", "self.combustor"),
            EnergyInput("residual", "self.turboshaft"),
        ),
        static=True,
    )

    installation_geometry: JetGeometry = init_field(JetGeometry)

    _bookkeeping: dict = init_field(lambda: {"compressors": Compressor, "turbines": Turbine}, static=True)

    @ru.inputs(
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
    @ru.outputs(
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

        F, F_sp, I_sp, TSFC, p, ff = engine_performance(
            u0=fs.speed,
            P0=fs.pressure,
            g=fs.gravity,
            delta_SFC=self.design_parameters.delta_SFC,
            v_fan_nozzle=0.0,
            A_fan_nozzle=0.0,
            P_fan_nozzle=0.0,
            v_core_nozzle=self.average_domain_inputs(state, "flow", "speed"),
            A_core_nozzle=self.core_nozzle.design_parameters.A_exit,
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
            lambda o: o.residual.work, outputs, self.sum_domain_inputs(state, "residual", "work")
        )
        outputs = eqx.tree_at(
            lambda o: o.residual.mass_flow_rate, outputs, self.sum_domain_inputs(state, "residual", "mass_flow_rate")
        )

        updated_state = eqx.tree_at(lambda s: s.energy.nodes[self.network_ID].outputs, state, outputs)

        return updated_state, system, settings


# ----------------------------------------------------------------------------------------------------------------------
# Turbofan Engine
# ----------------------------------------------------------------------------------------------------------------------


class Fan(FlowNode):
    tag: str = init_field("Fan", static=True)
    inputs: tuple = init_field((EnergyInput("flow", "Inlet Nozzle"),), static=True)

    map: CompressorMap = init_field(maps.Fan)

    @ru.inputs(
        "state.freestream.Cp",
        "state.freestream.gamma",
        "state.energy.nodes[Fan_flow_inputs].outputs.flow.stagnation_temperature",
        "state.energy.nodes[Fan_flow_inputs].outputs.flow.stagnation_pressure",
        "system.energy.nodes[Fan].pressure_ratio",
        "system.energy.nodes[Fan].efficiencies.flow",
    )
    @ru.outputs(
        "state.energy.nodes[Fan].outputs.flow.stagnation_pressure",
        "state.energy.nodes[Fan].outputs.flow.stagnation_temperature",
        "state.energy.nodes[Fan].outputs.flow.stagnation_enthalpy",
        "state.energy.nodes[Fan].outputs.mechanical.work",
    )
    def transmit(self, state: State, system: System, settings: Settings):

        work, P_t_out, T_t_out, h_t_out = fan_compressor_performance(
            gas=self.working_fluid,
            T_t=self.average_domain_inputs(state, "flow", "stagnation_temperature"),
            P_t=self.average_domain_inputs(state, "flow", "stagnation_pressure"),
            PR=self.pressure_ratio,
            n_p=self.efficiencies.flow,
        )

        # Set Output State
        outputs = state.energy.nodes[self.network_ID].outputs

        outputs = eqx.tree_at(lambda o: o.mechanical.work, outputs, work)

        outputs = eqx.tree_at(lambda o: o.flow.stagnation_pressure, outputs, P_t_out)
        outputs = eqx.tree_at(lambda o: o.flow.stagnation_temperature, outputs, T_t_out)
        outputs = eqx.tree_at(lambda o: o.flow.stagnation_enthalpy, outputs, h_t_out)

        updated_state = eqx.tree_at(lambda s: s.energy.nodes[self.network_ID].outputs, state, outputs)

        return updated_state, system, settings


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

#     @ru.inputs(
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
#     @ru.outputs(
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
