# RCAIDE/Library/Components/Energy/Propulsors.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: May 2025, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

import dataclasses as dc

# package imports
import equinox as eqx

# RCAIDE imports
from RCAIDE.utils import init_field
from RCAIDE.Library import Component
from RCAIDE.Library.Propellants import Propellant, JetA
from RCAIDE.Library.Gases import Gas, Air
from RCAIDE.Library.Components.Energy.Converters import EnergyConverter, FlowConverter, OfftakeShaft

from RCAIDE.Library.Methods.Energy.Converters.Turbofans import func_sea_level_static_thrust

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


class Propulsor(EnergyConverter):

    converters:         Component           = init_field(lambda: Component(tag='Propulsor Converters'))

    design_parameters:  DesignParameters    = init_field(DesignParameters)

    def compute_thrust(self):
        raise NotImplementedError("Subclasses must implement this method")


# ----------------------------------------------------------------------------------------------------------------------
# Jet Engines
# ----------------------------------------------------------------------------------------------------------------------

class JetInstallationGeometry(eqx.Module):

    xe: float = 1.
    ye: float = 1.
    Ce: float = 2.

def _JetConverters():
    return Component(tag="Jet Converters").add_subcomponent(FlowConverter(tag="Combustor"))

class JetEngine(Propulsor):

    tag:                            str     = init_field('Jet', static=True)
    plug_diameter:                  float   = 0.0

    fuel:                           Propellant      = init_field(JetA)
    working_fluid:                  Gas             = init_field(Air)

    converters:                     Component       = init_field(_JetConverters)

    installation_geometry:          JetInstallationGeometry     = init_field(JetInstallationGeometry)

def _TurbojetConverters():
    convs = Component(tag="Turbojet Converters")
    convs = convs.add_subcomponent(FlowConverter(tag="Inlet Nozzle"))
    
    comps = Component(tag='Compressors')
    comps = comps.add_subcomponent(FlowConverter(tag='Low Pressure Compressor'))
    comps = comps.add_subcomponent(FlowConverter(tag='High Pressure Compressor'))
    convs = convs.add_subcomponent(comps)

    convs = convs.add_subcomponent(FlowConverter(tag="Combustor"))
    
    turbs = Component(tag='Turbines')
    turbs = turbs.add_subcomponent(FlowConverter(tag='High Pressure Turbine'))
    turbs = turbs.add_subcomponent(FlowConverter(tag='Low Pressure Turbine'))
    convs = convs.add_subcomponent(turbs)

    convs = convs.add_subcomponent(OfftakeShaft())
    convs = convs.add_subcomponent(FlowConverter(tag='Core Nozzle'))

    return convs

class TurbojetEngine(JetEngine):

    tag: str = init_field('Turbojet', static=True)

    converters: Component = init_field(_TurbojetConverters)

def _TurbofanConverters():
    convs = _TurbojetConverters()
    convs = dc.replace(convs, tag="Turbofan Converters")
    convs = convs.insert_subcomponent(FlowConverter(tag="Fan"), 0)
    convs = convs.insert_subcomponent(FlowConverter(tag="Fan Nozzle"), 0)

    return convs

class TurbofanEngine(TurbojetEngine):

    tag: str = init_field('Turbofan', static=True)

    bypass_ratio: float = 1.0
    exa: float = 1.0                # Fan Face-to-Exit Distance

    converters: Component = init_field(_TurbofanConverters)

    def __post_init__(self):
        des = self.design_parameters
        if des.total_thrust != 0.0 and des.SLS_thrust == 0.0:
            
            func_sea_level_static_thrust(
                F_ref=des.total_thrust,
                delta_SFC=des.delta_SFC,
                v_fan_nozzle,
                AR_fan_nozzle,
                P_fan_nozzle,
                v_core_nozzle,
                AR_core_nozzle,
                P_core_nozzle,
                f, # Fuel-Air Ratio
                alpha, # Bypass Ratio
            ):
