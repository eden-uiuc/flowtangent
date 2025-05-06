# RCAIDE/Library/Components/Energy/Converter.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: Apr 2025, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from dataclasses import dataclass, field, make_dataclass

# package imports
import numpy as np

# RCAIDE imports
import RCAIDE.Library as rcl

# ----------------------------------------------------------------------------------------------------------------------
#  Energy Converter
# ----------------------------------------------------------------------------------------------------------------------


@dataclass(kw_only=True)
class EnergyConverter(rcl.Component):

    efficiency:             float = 1.0


# ----------------------------------------------------------------------------------------------------------------------
# Propulsor Subcomponents
# ----------------------------------------------------------------------------------------------------------------------

@dataclass(kw_only=True)
class FlowConverter(EnergyConverter):

    mechanical_efficiency:      float = 1.0
    polytropic_efficiency:      float = 1.0

    pressure_ratio:             float = 1.0
    area_ratio:                 float = 1.0

    design_intake_temperature:  float = 298.15    # Kelvin

    rotation_speed:             float = 0.0
    noise_speed:                float = 0.0


@dataclass(kw_only=True)
class OfftakeShaft(EnergyConverter):

    power_draw:             float = 0.0
    reference_temperature:  float = 298.15      # Kelvin
    reference_pressure:     float = 101325.0    # Pascal

# ----------------------------------------------------------------------------------------------------------------------
# Propulsors
# ----------------------------------------------------------------------------------------------------------------------


@dataclass(kw_only=True)
class DesignParameters:

    total_thrust:           float = 0.0

    altitude:               float = 0.0
    mach_number:            float = 0.0
    isa_deviation:          float = 0.0

    SLS_thrust:             float = 0.0

    mass_flow_through_rate: float = 0.0


@dataclass(kw_only=True)
class Propulsor(EnergyConverter):

    converters:                 dataclass               = field(default_factory=
                                                                lambda: make_dataclass(cls_name='PropulsorConverters',
                                                                    fields=[]))

    design_thrust_parameters:   DesignParameters = field(default_factory=DesignParameters)

    def compute_thrust(self):
        raise NotImplementedError("Subclasses must implement this method")

# ----------------------------------------------------------------------------------------------------------------------
# Jet Engines
# ----------------------------------------------------------------------------------------------------------------------


@dataclass(kw_only=True)
class JetInstallationGeometry:

    xe: float = 1.
    ye: float = 1.
    Ce: float = 2.


@dataclass(kw_only=True)
class JetEngine(Propulsor):

    name:                   str             = 'Jet'
    plug_diameter:          float           = 0.0

    reference_temperature:  float           = 288.15      # Kelvin
    reference_pressure:     float           = 101325.0    # Pascal

    fuel:                   rcl.Propellants.Propellant              = field(default_factory=rcl.Propellants.JetA)
    working_fluid:          rcl.Gases.Gas                     = field(default_factory=rcl.Gases.Air)

    installation_geometry:  JetInstallationGeometry     = field(default_factory=JetInstallationGeometry)

    def __post_init__(self):

        self.converters.combustor = FlowConverter(name='Combustor')

        self.design_thrust_parameters.fuel_air_ratio = 0.0


@dataclass(kw_only=True)
class TurbojetEngine(JetEngine):

    name: str = 'Turbojet'

    def __post_init__(self):
        super().__post_init__()

        self.converters.inlet_nozzle = FlowConverter(name='Inlet Nozzle')

        self.converters.compressors = [FlowConverter(name='Low Pressure Compressor'),
                                       FlowConverter(name='High Pressure Compressor')]

        self.converters.turbines = [FlowConverter(name='High Pressure Turbine'),
                                    FlowConverter(name='Low Pressure Turbine')]

        self.converters.offtake_shaft = OfftakeShaft()

        self.converters.core_nozzle = FlowConverter(name='Core Outlet Nozzle')


@dataclass(kw_only=True)
class TurbofanEngine(TurbojetEngine):

    name: str = 'Turbofan'

    bypass_ratio = 1.0
    exa: float = 1.0        # Fan Face-to-Exit Distance

    def __post_init__(self):
        super().__post_init__()

        self.converters.fan = FlowConverter(name='Fan')

        self.converters.fan_nozzle = FlowConverter(name='Fan Outlet Nozzle')
