# RCAIDE/Library/Components/Energy/Propulsors.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: May 2025, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from RCAIDE.Framework import Aircraft
import dataclasses as dc

# package imports
import equinox as eqx

# RCAIDE imports
from RCAIDE.Framework.Settings import Settings
from RCAIDE.Framework.State import State
from RCAIDE.Framework.System import System
from RCAIDE.utils import init_field
from RCAIDE.Library import Component
from RCAIDE.Library.Propellants import Propellant, JetA
from RCAIDE.Library.Gases import Gas, Air
from RCAIDE.Library.Components.Energy.Nodes import FlowNode


from RCAIDE.Library.Methods.Energy.Transmission.Nozzles import func_compression_nozzle_performance
from RCAIDE.Library.Methods.Energy.Transmission.Fan_Compressors import func_fan_compressor_performance
from RCAIDE.Library.Methods.Energy.Transmission.Turbofans import func_sea_level_static_thrust

# ----------------------------------------------------------------------------------------------------------------------
# Propulsors
# ----------------------------------------------------------------------------------------------------------------------


class DesignParameters(eqx.Module):

    total_thrust:           float = 0.0
    delta_SFC:              float = 0.0

    altitude:               float = 0.0
    mach_number:            float = 0.01
    temperature:            float   = 288.15      # Kelvin
    total_temperature:      float   = 298.15      # Kelvin
    pressure:               float   = 101325.0    # Pascal
    total_pressure:         float   = 101325.0    # Pascal
    
    isa_deviation:          float = 0.0

    SLS_thrust:             float = 0.0

    mass_flow_through_rate: float = 0.0
    fuel_air_ratio:         float = 0.0


class Propulsor(FlowNode):

    design_parameters:  DesignParameters    = init_field(DesignParameters)


# ----------------------------------------------------------------------------------------------------------------------
# Jet Engine
# ----------------------------------------------------------------------------------------------------------------------

class JetInstallationGeometry(eqx.Module):

    xe: float = 1.
    ye: float = 1.
    Ce: float = 2.

class JetEngine(Propulsor):

    tag:                            str     = init_field('Jet', static=True)
    plug_diameter:                  float   = 0.0

    fuel:                           Propellant      = init_field(JetA)
    working_fluid:                  Gas             = init_field(Air)

    installation_geometry:          JetInstallationGeometry     = init_field(JetInstallationGeometry)

    def __post_init__(self):
        object.__setattr__(self, "subcomponents", (FlowNode(tag="Combustor"),))
        

# ----------------------------------------------------------------------------------------------------------------------
# Jet Engine
# ----------------------------------------------------------------------------------------------------------------------

class InletNozzle(FlowNode):
    
    tag: str = "Inlet Nozzle"
    
    def transmit(self, state: State, system: Aircraft, settings: Settings): #type: ignore
        
        fs = state.freestream
        g   = fs.gamma

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

        # Set Output State
        outputs = state.energy.nodes[self.tag].outputs.flow
        
        outputs = eqx.tree_at(lambda o: o.mach_number           , outputs, M_out)
        outputs = eqx.tree_at(lambda o: o.velocity              , outputs, u_out)
        outputs = eqx.tree_at(lambda o: o.stagnation_pressure   , outputs, P_t_out)
        outputs = eqx.tree_at(lambda o: o.stagnation_temperature, outputs, T_t_out)
        outputs = eqx.tree_at(lambda o: o.static_temperature    , outputs, T_out)
        outputs = eqx.tree_at(lambda o: o.stagnation_enthalpy   , outputs, h_t_out)
        outputs = eqx.tree_at(lambda o: o.static_enthalpy       , outputs, h_out)

        updated_state = eqx.tree_at(lambda s: s.energy.nodes[self.tag].outputs.flow, state, outputs)

        return updated_state, system, settings

class Compressor(FlowNode):

    tag: str = "Compressor"

    flow_inputs: list[str] = init_field(list)
    
    def transmit(self, state: State, system: System, settings: Settings):
        fs  = state.freestream

        work, P_t_out, T_t_out, h_t_out = func_fan_compressor_performance(
            gamma=fs.gamma,
            Cp=fs.Cp,
            T_t=self.sum_inputs(state, "stagnation_temperature"),
            P_t=self.sum_inputs(state, "stagnation_pressure"),
            PR=self.pressure_ratio,
            n_p=self.efficiencies.flow
        )

        # Set Output State for current compressor
        outputs = state.energy.nodes[self.tag].outputs
        
        outputs = eqx.tree_at(lambda o: o.mechanical.work             , work)
        
        outputs = eqx.tree_at(lambda o: o.flow.stagnation_pressure    , P_t_out)
        outputs = eqx.tree_at(lambda o: o.flow.stagnation_temperature , T_t_out)
        outputs = eqx.tree_at(lambda o: o.flow.stagnation_enthalpy    , h_t_out)

        updated_state = eqx.tree_at(lambda s: s.energy.nodes[self.tag].outputs, state, outputs)

        return updated_state, system, settings

def _TurbojetSubComponents():
    
    inlet = InletNozzle()
    LPC = Compressor(tag="LPC", flow_inputs=["Inlet Nozzle"])
    HPC = Compressor(tag="HPC", flow_inputs=["LPC"])

    convs = convs.add_subcomponent(FlowNode(tag="Combustor"))
    
    turbs = Component(tag='Turbines')
    turbs = turbs.add_subcomponent(FlowNode(tag='High Pressure Turbine'))
    turbs = turbs.add_subcomponent(FlowNode(tag='Low Pressure Turbine'))
    convs = convs.add_subcomponent(turbs)

    convs = convs.add_subcomponent(OfftakeShaft())
    convs = convs.add_subcomponent(FlowNode(tag='Core Nozzle'))

    return (inlet, LPC, HPC)

class TurbojetEngine(JetEngine):

    tag: str = init_field('Turbojet', static=True)

    converters: Component = init_field(_TurbojetConverters)

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, self.subcomponents, self.subcomponents + _TurbojetSubComponents())
            
    

def _TurbofanConverters():
    convs = _TurbojetConverters()
    convs = dc.replace(convs, tag="Turbofan Converters")
    convs = convs.insert_subcomponent(FlowNode(tag="Fan"), 0)
    convs = convs.insert_subcomponent(FlowNode(tag="Fan Nozzle"), 0)

    return convs

class TurbofanEngine(TurbojetEngine):

    tag: str = init_field('Turbofan', static=True)

    bypass_ratio: float = 1.0
    exa: float = 1.0                # Fan Face-to-Exit Distance

    converters: Component = init_field(_TurbofanConverters)