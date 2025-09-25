# RCAIDE/Library/Components/Energy/Propulsors.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: May 2025, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

import chex
from dataclasses import field

# package imports
import numpy as np

# RCAIDE imports
import RCAIDE.Framework as rcf
import RCAIDE.Library as rcl
from RCAIDE.Library.Components.Energy.Converters import *

# ----------------------------------------------------------------------------------------------------------------------
# Propulsors
# ----------------------------------------------------------------------------------------------------------------------


@chex.dataclass(kw_only=True)
class DesignParameters:

    total_thrust:           float = 0.0

    altitude:               float = 0.0
    mach_number:            float = 0.0
    isa_deviation:          float = 0.0

    SLS_thrust:             float = 0.0

    mass_flow_through_rate: float = 0.0


@chex.dataclass(kw_only=True)
class Propulsor(EnergyConverter):

    converters:                 rcl.Component   = field(default_factory=
                                                        lambda: rcl.Component(name='Propulsor Converters'))

    design_thrust_parameters:   DesignParameters = field(default_factory=DesignParameters)

    def compute_thrust(self):
        raise NotImplementedError("Subclasses must implement this method")


# ----------------------------------------------------------------------------------------------------------------------
# Jet Engines
# ----------------------------------------------------------------------------------------------------------------------


@chex.dataclass(kw_only=True)
class JetInstallationGeometry:

    xe: float = 1.
    ye: float = 1.
    Ce: float = 2.


@chex.dataclass(kw_only=True)
class JetEngine(Propulsor):

    name:                           str             = 'Jet'
    plug_diameter:                  float           = 0.0

    reference_temperature:          float           = 288.15      # Kelvin
    reference_total_temperature:    float           = 298.15      # Kelvin

    reference_pressure:             float           = 101325.0    # Pascal
    reference_total_pressure:       float           = 101325.0    # Pascal

    fuel:                   rcl.Propellants.Propellant  = field(default_factory=rcl.Propellants.JetA)
    working_fluid:          rcl.Gases.Gas               = field(default_factory=rcl.Gases.Air)

    installation_geometry:  JetInstallationGeometry     = field(default_factory=JetInstallationGeometry)

    def __post_init__(self):

        self.converters.add_subcomponent(FlowConverter(name='Combustor'))

        self.design_thrust_parameters.fuel_air_ratio = 0.0


@chex.dataclass(kw_only=True)
class TurbojetEngine(JetEngine):

    name: str = 'Turbojet'

    def __post_init__(self):
        super(TurbojetEngine, self).__post_init__()

        self.converters.add_subcomponent(FlowConverter(name='Inlet Nozzle'))

        self.converters.add_subcomponent(rcl.Component(name='Compressors'))
        self.converters.compressors.add_subcomponent(FlowConverter(name='Low Pressure Compressor'))
        self.converters.compressors.add_subcomponent(FlowConverter(name='High Pressure Compressor'))

        self.converters.add_subcomponent(rcl.Component(name='Turbines'))
        self.converters.turbines.add_subcomponent(FlowConverter(name='High Pressure Turbine'))
        self.converters.turbines.add_subcomponent(FlowConverter(name='Low Pressure Turbine'))

        self.converters.add_subcomponent(OfftakeShaft())

        self.converters.add_subcomponent(FlowConverter(name='Core Nozzle'))


@chex.dataclass(kw_only=True)
class TurbofanEngine(TurbojetEngine):

    name: str = 'Turbofan'

    bypass_ratio = 1.0
    exa: float = 1.0        # Fan Face-to-Exit Distance

    def __post_init__(self):
        super(TurbofanEngine, self).__post_init__()

        self.converters.add_subcomponent(FlowConverter(name='Fan'))

        self.converters.add_subcomponent(FlowConverter(name='Fan Nozzle'))
