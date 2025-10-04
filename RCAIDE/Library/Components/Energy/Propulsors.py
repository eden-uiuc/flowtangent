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


@chex.dataclass(kw_only=True, slots=True)
class DesignParameters:

    total_thrust:           float = 0.0

    altitude:               float = 0.0
    mach_number:            float = 0.0
    isa_deviation:          float = 0.0

    SLS_thrust:             float = 0.0

    mass_flow_through_rate: float = 0.0


@chex.dataclass(kw_only=True, slots=True)
class Propulsor(EnergyConverter):

    converters:                 rcl.Component   = field(default_factory=
                                                        lambda: rcl.Component(tag='Propulsor Converters'))

    design_thrust_parameters:   DesignParameters = field(default_factory=DesignParameters)

    def compute_thrust(self):
        raise NotImplementedError("Subclasses must implement this method")


# ----------------------------------------------------------------------------------------------------------------------
# Jet Engines
# ----------------------------------------------------------------------------------------------------------------------


@chex.dataclass(kw_only=True, slots=True)
class JetInstallationGeometry:

    xe: float = 1.
    ye: float = 1.
    Ce: float = 2.


@chex.dataclass(kw_only=True, slots=True)
class JetEngine(Propulsor):

    tag:                           str             = 'Jet'
    plug_diameter:                  float           = 0.0

    reference_temperature:          float           = 288.15      # Kelvin
    reference_total_temperature:    float           = 298.15      # Kelvin

    reference_pressure:             float           = 101325.0    # Pascal
    reference_total_pressure:       float           = 101325.0    # Pascal

    fuel:                   rcl.Propellants.Propellant  = field(default_factory=rcl.Propellants.JetA)
    working_fluid:          rcl.Gases.Gas               = field(default_factory=rcl.Gases.Air)

    installation_geometry:  JetInstallationGeometry     = field(default_factory=JetInstallationGeometry)

    def __post_init__(self):

        self.converters.add_subcomponent(FlowConverter(tag='Combustor'))

        self.design_thrust_parameters.fuel_air_ratio = 0.0


@chex.dataclass(kw_only=True, slots=True)
class TurbojetEngine(JetEngine):

    tag: str = 'Turbojet'

    def __post_init__(self):
        super(TurbojetEngine, self).__post_init__()

        self.converters.add_subcomponent(FlowConverter(tag='Inlet Nozzle'))

        self.converters.add_subcomponent(rcl.Component(tag='Compressors'))
        self.converters.compressors.add_subcomponent(FlowConverter(tag='Low Pressure Compressor'))
        self.converters.compressors.add_subcomponent(FlowConverter(tag='High Pressure Compressor'))

        self.converters.add_subcomponent(rcl.Component(tag='Turbines'))
        self.converters.turbines.add_subcomponent(FlowConverter(tag='High Pressure Turbine'))
        self.converters.turbines.add_subcomponent(FlowConverter(tag='Low Pressure Turbine'))

        self.converters.add_subcomponent(OfftakeShaft())

        self.converters.add_subcomponent(FlowConverter(tag='Core Nozzle'))


@chex.dataclass(kw_only=True, slots=True)
class TurbofanEngine(TurbojetEngine):

    tag: str = 'Turbofan'

    bypass_ratio = 1.0
    exa: float = 1.0        # Fan Face-to-Exit Distance

    def __post_init__(self):
        super(TurbofanEngine, self).__post_init__()

        self.converters.add_subcomponent(FlowConverter(tag='Fan'))

        self.converters.add_subcomponent(FlowConverter(tag='Fan Nozzle'))
