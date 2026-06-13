# RCAIDE/Library/Components/Energy/Propulsors.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: May 2025, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from __future__ import annotations
from typing import Optional

import warnings

# package imports
import jax.numpy as jnp
import equinox as eqx

# RCAIDE imports
from RCAIDE.utils import init_field, inputs, outputs

from RCAIDE.Framework.Settings import Settings
from RCAIDE.Framework.State import State
from RCAIDE.Framework.System import System, Aircraft
from RCAIDE.Framework.Missions.Initialize import initialize_energy

from RCAIDE.Library.Atmospheres import USStandard1976
from RCAIDE.Library.Propellants import Propellant, JetA
from RCAIDE.Library.Gases import Gas, Air
from RCAIDE.Library.Components.Energy.Nodes import FlowNode, FlowSplitter


from RCAIDE.Library.Methods.Energy.Transmission.Nozzles import func_compression_nozzle_performance
from RCAIDE.Library.Methods.Energy.Transmission.Fan_Compressors import func_fan_compressor_performance
from RCAIDE.Library.Methods.Energy.Transmission.Combustors import func_combustor_performance
from RCAIDE.Library.Methods.Energy.Transmission.Turbines import func_turbine_performance
from RCAIDE.Library.Methods.Energy.Transmission.Nozzles import func_expansion_nozzle_performance, func_compression_nozzle_performance
from RCAIDE.Library.Methods.Energy.Transmission.Turbofans import func_thrust_and_power
from RCAIDE.Library.Methods.Energy.Transmission.Turbofans import func_sea_level_static_thrust

# ----------------------------------------------------------------------------------------------------------------------
# Propulsors
# ----------------------------------------------------------------------------------------------------------------------


class DesignParameters(eqx.Module):

    total_thrust:                   float = 0.0
    delta_SFC:                      float = 0.0

    altitude:                       float = 0.0
    mach_number:                    float = 0.01
    
    temperature:                    float = 288.15      # Kelvin
    total_temperature:              float = 288.15      # Kelvin
    
    pressure:                       float = 101325.0    # Pascal
    total_pressure:                 float = 101325.0    # Pascal
    
    isa_deviation:                  float = 0.0

    SLS_thrust:                     float = 0.0

    mass_flow_through_rate:         float = 0.0
    fuel_air_ratio:                 float = 0.0
    turbine_intake_temperature:     float = 0.0


class Propulsor(FlowNode):

    design_parameters:  DesignParameters    = init_field(DesignParameters)
        

# ----------------------------------------------------------------------------------------------------------------------
# Turbojet Engine
# ----------------------------------------------------------------------------------------------------------------------

class JetInstallationGeometry(eqx.Module):

    xe: float = 1.
    ye: float = 1.
    Ce: float = 2.

class InletNozzle(FlowNode):
    
    tag: str = init_field("Inlet Nozzle", static=True)
    
    @inputs(
        "state.freestream.stagnation_temperature",
        "state.freestream.stagnation_pressure",
        "state.freestream.pressure",
        "state.freestream.mach_number",
        "state.freestream.Cp",
        "state.freestream.gamma"
        "system.energy.[InletNozzle].pressure_ratio",
        "system.energy.nodes[InletNozzle].pressure_recovery",
        "system.energy.nodes[InletNozzle].efficiencies.flow"
    )
    @outputs(
        "state.energy.nodes[InletNozzle].outputs.flow.mach_number",
        "state.energy.nodes[InletNozzle].outputs.flow.speed",
        "state.energy.nodes[InletNozzle].outputs.flow.stagnation_pressure",
        "state.energy.nodes[InletNozzle].outputs.flow.temperature",
        "state.energy.nodes[InletNozzle].outputs.flow.stagnation_temperature",
        "state.energy.nodes[InletNozzle].outputs.flow.enthalpy",
        "state.energy.nodes[InletNozzle].outputs.flow.stagnation_enthalpy",
        
    )
    def transmit(self, state: State, system: Aircraft, settings: Settings): #type: ignore
        
        fs = state.freestream

        M_out, u_out, P_t_out, T_t_out, T_out, h_t_out, h_out = func_compression_nozzle_performance(
            T_t=fs.stagnation_temperature,
            P_t=fs.stagnation_pressure,
            P0=fs.pressure,
            M0=fs.mach_number,
            Cp=fs.Cp,
            gamma=fs.gamma,
            PR=self.pressure_ratio,
            n_r=self.pressure_recovery,
            n_p=self.efficiencies.flow,
        )

        outputs = state.energy.nodes[self.tag].outputs.flow
        
        outputs = eqx.tree_at(lambda o: o.mach_number           , outputs, M_out)
        outputs = eqx.tree_at(lambda o: o.speed                 , outputs, u_out)
        outputs = eqx.tree_at(lambda o: o.stagnation_pressure   , outputs, P_t_out)
        outputs = eqx.tree_at(lambda o: o.stagnation_temperature, outputs, T_t_out)
        outputs = eqx.tree_at(lambda o: o.temperature           , outputs, T_out)
        outputs = eqx.tree_at(lambda o: o.stagnation_enthalpy   , outputs, h_t_out)
        outputs = eqx.tree_at(lambda o: o.enthalpy              , outputs, h_out)

        updated_state = eqx.tree_at(lambda s: s.energy.nodes[self.tag].outputs.flow, state, outputs)

        return updated_state, system, settings

class Compressor(FlowNode):

    tag: str = init_field("Compressor", static=True)

    flow_inputs: list[str] = init_field(list, static=True)
    
    @inputs(
        "state.freestream.Cp",
        "state.freestream.gamma",
        "state.energy.nodes[Compressor_flow_inputs].outputs.flow.stagnation_temperature,"
        "state.energy.nodes[Compressor_flow_inputs].outputs.flow.stagnation_pressure,"
        "system.energy.nodes[Compressor].pressure_ratio",
        "system.energy.nodes[Compressor].efficiencies.flow"
    )
    @outputs(
        "state.energy.nodes[Compressor].flow.stagnation_temperature"
        "state.energy.nodes[Compressor].flow.stagnation_pressure",
        "state.energy.nodes[Compressor].flow.stagnation_enthalpy",
        "state.energy.nodes[Compressor].mechanical.work"
    )
    def transmit(self, state: State, system: System, settings: Settings):
        fs  = state.freestream

        work, P_t_out, T_t_out, h_t_out = func_fan_compressor_performance(
            gamma=fs.gamma,
            Cp=fs.Cp,
            T_t=self.sum_inputs(state, "flow", "stagnation_temperature"),
            P_t=self.sum_inputs(state, "flow", "stagnation_pressure"),
            PR=self.pressure_ratio,
            n_p=self.efficiencies.flow
        )

        outputs = state.energy.nodes[self.tag].outputs
        
        outputs = eqx.tree_at(lambda o: o.mechanical.work             , work)
        
        outputs = eqx.tree_at(lambda o: o.flow.stagnation_pressure    , P_t_out)
        outputs = eqx.tree_at(lambda o: o.flow.stagnation_temperature , T_t_out)
        outputs = eqx.tree_at(lambda o: o.flow.stagnation_enthalpy    , h_t_out)

        updated_state = eqx.tree_at(lambda s: s.energy.nodes[self.tag].outputs, state, outputs)

        return updated_state, system, settings

class TurbojetCombustor(FlowNode):

    tag = init_field("Combustor", static=True)

    flow_inputs = init_field(['HPC'], static=True)

    @inputs(
        "state.freestream.Cp"
        "state.energy.nodes[Combustor_flow_inputs].outputs.flow.stagnation_temperature",
        "state.energy.nodes[Combustor_flow_inputs].outputs.flow.stagnation_pressure",
        "system.energy.nodes[Turbojet].design_paramters.turbine_intake_temperature",
        "system.energy.nodes[Turbojet].fuel.specific_energy",
        "system.energy.nodes[Combustor].pressure_ratio",
        "system.energy.nodes[Combustor].efficiencies.flow",
    )
    @outputs(
        "state.energy.nodes[Combustor].outputs.flow.stagnation_pressure",
        "state.energy.nodes[Combustor].outputs.flow.stagnation_temperature",
        "state.energy.nodes[Combustor].outputs.flow.stagnation_enthalpy",
        "state.energy.nodes[Combustor].outputs.fuel.fuel_air_ratio",
    )
    def transmit(self, state: State, system: System, settings: Settings):
        
        jet_tag     = '.'.join(self.tag.split('.')[:-1])
        jet         = system.energy[jet_tag]

        P_t_out, h_t_out, f = func_combustor_performance(
            T_t_in = self.sum_inputs(state, "flow", "stagnation_temperature"),
            P_t_in=self.sum_inputs(state, "flow", "stagnation_pressure"),
            T_t_out=jet.design_parameters.turbine_intake_temperature,
            h_t_f=jet.fuel.specific_energy,
            Cp=state.freestream.Cp,
            PR=self.pressure_ratio,
            n_b=self.efficiencies.flow,
        )

        outputs = state.energy.nodes[self.tag].outputs
        
        outputs = eqx.tree_at(lambda o: o.flow.stagnation_pressure   , outputs, P_t_out)
        outputs = eqx.tree_at(lambda o: o.flow.stagnation_temperature, outputs, jet.design_parameters.turbine_intake_temperature)
        outputs = eqx.tree_at(lambda o: o.flow.stagnation_enthalpy   , outputs, h_t_out)
        
        outputs = eqx.tree_at(lambda o: o.fuel.fuel_air_ratio        , outputs, f)

class Turbine(FlowNode):

    tag = init_field("Turbine", static=True)
    
    mechanical_inputs = init_field(["Offtake Shaft"], static=True)
    flow_inputs: list[str] = init_field(list, static=True)
    fuel_inputs: list[str] = init_field(["Combustor"], static=True)

    @inputs(
        "state.freestream.gamma",
        "state.freestream.Cp",
        "state.energy.nodes[Turbine_flow_inputs].outputs.flow.stagnation_temperature",
        "state.energy.nodes[Turbine_flow_inputs].outputs.flow.stagnation_pressure",
        "state.energy.nodes[Turbine_fuel_inputs].outputs.fuel.fuel_air_ratio",
        "state.energy.nodes[Turbine_mechanical_inputs].outputs.mechanical.work",
        "system.energy.nodes[Turbine].efficiencies.mechanical",
        "system.energy.nodes[Turbine].efficiencies.flow",
    )
    @outputs(
        "state.energy.nodes[Turbine].outputs.flow.stagnation_temperature",
        "state.energy.nodes[Turbine].outputs.flow.stagnation_pressure",
        "state.energy.nodes[Turbine].outputs.flow.stagnation_enthalpy",
    )
    def transmit(self, state: State, system: System, settings: Settings):

        T_t_out, P_t_out, h_t_out = func_turbine_performance(
            gamma=state.freestream.gamma,
            Cp=state.freestream.Cp,
            f=self.average_inputs(state, "fuel", "fuel_air_ratio"),
            input_work=self.sum_inputs(state, "mechanical", "work"),
            n_mech=self.efficiencies.mechanical,
            n_flow=self.efficiencies.flow,
            T_t=self.average_inputs(state, "flow", "stagnation_temperature"),
            P_t=self.average_inputs(state, "flow", "stagnation_pressure"),
        )

        # Set Output State
        outputs = state.energy.nodes[self.tag].outputs.flow
        
        outputs = eqx.tree_at(lambda o: o.stagnation_temperature, outputs, T_t_out)
        outputs = eqx.tree_at(lambda o: o.stagnation_pressure, outputs, P_t_out)
        outputs = eqx.tree_at(lambda o: o.stagnation_enthalpy, outputs, h_t_out)

        updated_state = eqx.tree_at(lambda s: s.energy.nodes[self.tag].outputs.flow, state, outputs)

        return updated_state, system, settings

class ExpansionNozzle(FlowNode):

    tag = init_field("Nozzle", static=True)

    @inputs(
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
    @outputs(
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

        fs = state.freestream

        AR, M, r_out, u_out, P_out, P_t_out, T_out, T_t_out, h_out, h_t_out = func_expansion_nozzle_performance(
            T_t=self.average_inputs(state, "flow", "stagnation_temperature"),
            T_t0=fs.stagnation_temperature,
            P_t=self.average_inputs(state, "flow", "stagnation_pressure"),
            P_t0=fs.stagnation_pressure,
            P0=fs.pressure,
            M0=fs.mach_number,
            Cp=fs.Cp,
            gamma=fs.gamma,
            R=fs.R,
            PR=self.pressure_ratio,
            n_p=self.efficiencies.flow
        )

        outputs = state.energy.nodes[self.tag].flow

        outputs = eqx.tree_at(lambda o: o.area_ratio            , outputs , AR)
        outputs = eqx.tree_at(lambda o: o.mach_number           , outputs , M)
        outputs = eqx.tree_at(lambda o: o.density               , outputs , r_out)
        outputs = eqx.tree_at(lambda o: o.speed                 , outputs , u_out)
        outputs = eqx.tree_at(lambda o: o.pressure              , outputs , P_out)
        outputs = eqx.tree_at(lambda o: o.stagnation_pressure   , outputs , P_t_out)
        outputs = eqx.tree_at(lambda o: o.temperature           , outputs , T_out)
        outputs = eqx.tree_at(lambda o: o.stagnation_temperature, outputs , T_t_out)
        outputs = eqx.tree_at(lambda o: o.enthalpy              , outputs , h_out)
        outputs = eqx.tree_at(lambda o: o.stagnation_enthalpy   , outputs , h_t_out)

        updated_state = eqx.tree_at(lambda s: s.energy.nodes[self.tag].outputs.flow, state, outputs)

        return updated_state, system, settings

def _TurbojetSetup():
    
    inlet = InletNozzle()
    LPC = Compressor(tag="LPC", flow_inputs=["Inlet Nozzle"])
    HPC = Compressor(tag="HPC", flow_inputs=["LPC"])

    comb = TurbojetCombustor()

    HPT = Turbine(tag="HPT", mechanical_inputs=["HPC"], flow_inputs=["Combustor"])
    LPT = Turbine(tag="LPT", mechanical_inputs=["LPC"], flow_inputs=["HPT"])

    nozz = ExpansionNozzle(tag="Core Nozzle", flow_inputs=["LPC"])

    return (inlet, LPC, HPC, comb, HPT, LPT, nozz)

class TurbojetEngine(Propulsor):

    tag: str        = init_field('Turbojet', static=True)
    subcomponents   = tuple = init_field(_TurbojetSetup())

    plug_diameter:                  float   = 0.0

    fuel:                           Propellant      = init_field(JetA)
    working_fluid:                  Gas             = init_field(Air)

    installation_geometry:          JetInstallationGeometry     = init_field(JetInstallationGeometry)

    @property
    def compressors(self):
        return tuple(c for c in self.subcomponents if isinstance(c, Compressor))
    
    @property
    def turbines(self):
        return tuple(c for c in self.subcomponents if isinstance(c, Turbine))
    
    def update_SLS_thrust(self) -> TurbojetEngine:
        design_thrust = self.design_parameters.total_thrust
        
        if design_thrust == 0.0:
            warnings.warn("Attempted to calculate sea-level static thrust without reference design thrust. "
            f"Please set {self.tag}.design_parameters.total_thrust.")
            return self
        
        atmo = USStandard1976()
        T0 = atmo.compute_temperature(0.0)
        a0 = atmo.compute_speed_of_sound(0.0)
        M0 = 0.01

        sls_state = eqx.tree_at(lambda s:(
            s.freestream.altitude,
            s.freestream.gravity,
            s.freestream.mach_number,
            s.freestream.speed_of_sound,
            s.freestream.speed,
            s.freestream.temperature,
            s.freestream.pressure,
            s.freestream.density,
            s.freestream.dynamic_viscosity,
            s.freestream.isentropic_expansion_factor,
            s.freestream.Cp,
            s.freestream.R,
        ), State().expand_rows(1),
        (
            jnp.atleast_2d(0.),
            jnp.atleast_2d(9.81),
            jnp.atleast_2d(M0),
            jnp.atleast_2d(a0),
            jnp.atleast_2d(a0*M0),
            jnp.atleast_2d(T0),
            jnp.atleast_2d(atmo.compute_pressure(0.0)),
            jnp.atleast_2d(atmo.compute_density(0.0)),
            jnp.atleast_2d(atmo.compute_dynamic_viscosity(0.0)),
            jnp.atleast_2d(self.working_fluid.compute_gamma(T0)),
            jnp.atleast_2d(self.working_fluid.compute_Cp(T0)),
            jnp.atleast_2d(self.working_fluid.R_specific),
        ))

        sls_system = Aircraft()
        sls_system = eqx.tree_at(lambda s: s.energy, sls_system, sls_system.energy.add_subcomponent(self))

        sls_state, sls_system, settings = initialize_energy(sls_state, sls_system, Settings())
        sls_state, sls_system, settings = self.transmit(sls_state, sls_system, settings)

        sls_thrust = sls_state.energy.nodes[self.tag].outputs.force.thrust.item(0)

        updated_params = eqx.tree_at(lambda d: d.SLS_thrust, self.design_parameters, sls_thrust)
        updated_self = eqx.tree_at(lambda s: s.design_parameters, self, updated_params)

        return updated_self

    @inputs(
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
    @outputs(
        "state.energy.nodes[Turbojet].outputs.force.thrust",
        "state.energy.nodes[Turbojet].outputs.force.nondimensional_thrust",
        "state.energy.nodes[Turbojet].outputs.force.specific_impulse",
        "state.energy.nodes[Turbojet].outputs.fuel.TSFC",
        "state.energy.nodes[Turbojet].outputs.fuel.flow_rate",
        "state.energy.nodes[Turbojet].outputs.flow.mass_flow_rate",
        "state.energy.nodes[Turbojet].outputs.mechanical.power",
    )
    def transmit(self, state: State, system: System, settings: Settings):
        
        cn_out      = state.energy.nodes[self.tag + ".core_nozzle"].outputs.flow
        comb_out    = state.energy.nodes[self.tag + ".combustor"].outputs.fuel
        
        fs = state.freestream
        
        F, F_sp, I_sp, TSFC, mdot_c, p, ff = func_thrust_and_power(
                gamma=fs.gamma,
                u0=fs.speed,
                a0=fs.speed_of_sound,
                M0=fs.mach_number,
                P0=fs.pressure,
                g=fs.gravity,
                F_ref =self.design_parameters.total_thrust,
                delta_SFC=self.design_parameters.delta_SFC,
                v_fan_nozzle=0.,
                AR_fan_nozzle=0.,
                P_fan_nozzle=0.,
                v_core_nozzle=cn_out.speed,
                AR_core_nozzle=cn_out.area_ratio,
                P_core_nozzle= cn_out.pressure,
                f=comb_out.fuel_air_ratio,
                alpha=0.,
                throttle=state.energy.nodes[self.tag].throttle,
            )
        
        outputs = state.energy.nodes[self.tag].outputs

        outputs = eqx.tree_at(lambda o: o.force.thrust                , outputs, F)
        outputs = eqx.tree_at(lambda o: o.force.non_dimensional_thrust, outputs, F_sp)
        outputs = eqx.tree_at(lambda o: o.force.specific_impulse      , outputs, I_sp)
        
        outputs = eqx.tree_at(lambda o: o.fuel.TSFC,                    outputs, TSFC)
        outputs = eqx.tree_at(lambda o: o.fuel.fuel_flow_rate,          outputs, ff)
        
        outputs = eqx.tree_at(lambda o: o.flow.mass_flow_rate, outputs, mdot_c)
        
        outputs = eqx.tree_at(lambda o: o.mechanical.power, outputs, p)

        updated_state = eqx.tree_at(lambda s: s.energy.nodes[self.tag].outputs, state, outputs)

        return updated_state, system, settings

# ----------------------------------------------------------------------------------------------------------------------
# Turbofan Engine
# ----------------------------------------------------------------------------------------------------------------------    

class Fan(FlowNode):

    tag = "Fan"
    flow_inputs = init_field(["Inlet Nozzle"], static=True)

    @inputs(
        "state.freestream.Cp",
        "state.freestream.gamma",
        "state.energy.nodes[Fan_flow_inputs].outputs.flow.stagnation_temperature",
        "state.energy.nodes[Fan_flow_inputs].outputs.flow.stagnation_pressure",
        "system.energy.nodes[Fan].pressure_ratio",
        "system.energy.nodes[Fan].efficiencies.flow",
    )
    @outputs(
        "state.energy.nodes[Fan].outputs.flow.stagnation_pressure",
        "state.energy.nodes[Fan].outputs.flow.stagnation_temperature",
        "state.energy.nodes[Fan].outputs.flow.stagnation_enthalpy",
        "state.energy.nodes[Fan].outputs.mechanical.work",
    )
    def transmit(self, state: State, system: System, settings: Settings):

        work, P_t_out, T_t_out, h_t_out = func_fan_compressor_performance(
            gamma=state.freestream.gamma,
            Cp=state.freestream.Cp,
            T_t=self.average_inputs(state, "flow", "stagnation_temperature"),
            P_t=self.average_inputs(state, "flow", "stagnation_pressure"),
            PR=self.pressure_ratio,
            n_p=self.efficiencies.flow
        )

        # Set Output State
        outputs = state.energy.nodes[self.tag].outputs
        
        outputs = eqx.tree_at(lambda o: o.mechanical.work, outputs, work)
        
        outputs = eqx.tree_at(lambda o: o.flow.stagnation_pressure   , outputs, P_t_out)
        outputs = eqx.tree_at(lambda o: o.flow.stagnation_temperature, outputs, T_t_out)
        outputs = eqx.tree_at(lambda o: o.flow.stagnation_enthalpy   , outputs, h_t_out)

        updated_state = eqx.tree_at(lambda s: s.energy.nodes[self.tag].outputs, state, outputs)

        return updated_state, system, settings

def _TurbofanSetup(BPR):

    inlet = InletNozzle()
    fan   = Fan()

    core_flow = FlowSplitter(tag="Core Duct", flow_inputs=["Fan"], extraction_fraction=1./(1.+BPR))
    bypass_flow = FlowSplitter(tag="Bypass Duct", flow_inputs=["Fan"], extraction_fraction=BPR/(1.+BPR))
    
    LPC = Compressor(tag="LPC", flow_inputs=["Core Duct"])
    HPC = Compressor(tag="HPC", flow_inputs=["LPC"])

    comb = TurbojetCombustor()

    HPT = Turbine(tag="HPT", mechanical_inputs=["HPC"], flow_inputs=["Combustor"])
    LPT = Turbine(tag="LPT", mechanical_inputs=["LPC", "Fan"], flow_inputs=["HPT"])

    core_nozz = ExpansionNozzle(tag="Core Nozzle", flow_inputs=["LPC"])
    fan_nozz = ExpansionNozzle(tag="Fan Nozzle", flow_inputs=["Bypass Duct"])

    return (inlet, fan, core_flow, bypass_flow, LPC, HPC, comb, HPT, LPT, core_nozz, fan_nozz)
    
class TurbofanEngine(TurbojetEngine):

    tag: str = init_field('Turbofan', static=True)

    bypass_ratio: float = 1.0
    exa: float = 1.0                # Fan Face-to-Exit Distance
    
    def __post_init__(self):
        object.__setattr__(self, "subcomponents", _TurbofanSetup(self.bypass_ratio))
    
    @inputs(
        "state.freestream.gamma",
        "state.freestream.speed",
        "state.freestream.speed_of_sound",
        "state.freestream.mach_number",
        "state.freestream.pressure",
        "state.freestream.gravity",
        "state.energy.nodes[Turbofan].throttle"
        "state.energy.nodes[Turbofan_core_nozzle].outputs.flow.speed",
        "state.energy.nodes[Turbofan_core_nozzle].outputs.flow.area_ratio",
        "state.energy.nodes[Turbofan_core_nozzle].outputs.flow.pressure",
        "state.energy.nodes[Turbofan_fan_nozzle].outputs.flow.speed",
        "state.energy.nodes[Turbofan_fan_nozzle].outputs.flow.area_ratio",
        "state.energy.nodes[Turbofan_fan_nozzle].outputs.flow.pressure",
        "state.energy.nodes[Turbofan_combustor].outputs.fuel.fuel_air_ratio",
        "system.energy.nodes[Turbofan].bypass_ratio"
    )
    @outputs(
        "state.energy.nodes[Turbofan].outputs.force.thrust",
        "state.energy.nodes[Turbofan].outputs.force.nondimensional_thrust",
        "state.energy.nodes[Turbofan].outputs.force.specific_impulse",
        "state.energy.nodes[Turbofan].outputs.fuel.TSFC",
        "state.energy.nodes[Turbofan].outputs.fuel.flow_rate",
        "state.energy.nodes[Turbofan].outputs.flow.mass_flow_rate",
        "state.energy.nodes[Turbofan].outputs.mechanical.power",
    )
    def transmit(self, state: State, system: System, settings: Settings):
        
        cn_out      = state.energy.nodes[self.tag + ".core_nozzle"].outputs.flow
        fn_out      = state.energy.nodes[self.tag + ".fan_nozzle"].outputs.flow
        comb_out    = state.energy.nodes[self.tag + ".combustor"].outputs.fuel
        
        fs = state.freestream
        
        F, F_sp, I_sp, TSFC, mdot_c, p, ff = func_thrust_and_power(
                gamma=fs.gamma,
                u0=fs.speed,
                a0=fs.speed_of_sound,
                M0=fs.mach_number,
                P0=fs.pressure,
                g=fs.gravity,
                F_ref =self.design_parameters.total_thrust,
                delta_SFC=self.delta_SFC,
                v_fan_nozzle=fn_out.speed,
                AR_fan_nozzle=fn_out.area_ratio,
                P_fan_nozzle=fn_out.pressure,
                v_core_nozzle=cn_out.speed,
                AR_core_nozzle=cn_out.area_ratio,
                P_core_nozzle= cn_out.pressure,
                f=comb_out.fuel_air_ratio,
                alpha=self.bypass_ratio,
                throttle=state.energy.nodes[self.tag].throttle,
            )
        
        outputs = state.energy.nodes[self.tag].outputs

        outputs = eqx.tree_at(lambda o: o.force.thrust, outputs, F)
        outputs = eqx.tree_at(lambda o: o.force.nondimensional_thrust, outputs, F_sp)
        outputs = eqx.tree_at(lambda o: o.force.specific_impulse, outputs, I_sp)
        
        outputs = eqx.tree_at(lambda o: o.fuel.TSFC, outputs, TSFC)
        outputs = eqx.tree_at(lambda o: o.fuel.flow_rate, outputs, ff)
        
        outputs = eqx.tree_at(lambda o: o.flow.mass_flow_rate, outputs, mdot_c)
        
        outputs = eqx.tree_at(lambda o: o.mechanical.power, outputs, p)

        updated_state = eqx.tree_at(lambda s: s.energy.nodes[self.tag].outputs, state, outputs)

        return updated_state, system, settings


