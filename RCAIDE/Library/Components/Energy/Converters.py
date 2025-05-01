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

    efficiency:             float = field(default=1.0)


# ----------------------------------------------------------------------------------------------------------------------
# Propulsor Subcomponents
# ----------------------------------------------------------------------------------------------------------------------

@dataclass(kw_only=True)
class FlowConverter(EnergyConverter):

    mechanical_efficiency:      float = field(default=1.0)
    polytropic_efficiency:      float = field(default=1.0)

    pressure_ratio:             float = field(default=1.0)

    design_intake_temperature:  float = field(default=298.15)    # Kelvin


@dataclass(kw_only=True)
class OfftakeShaft(EnergyConverter):

    power_draw:             float = field(default=0.0)
    reference_temperature:  float = field(default=298.15)    # Kelvin
    reference_pressure:     float = field(default=101325.0)  # Pascal

# ----------------------------------------------------------------------------------------------------------------------
# Propulsors
# ----------------------------------------------------------------------------------------------------------------------


@dataclass(kw_only=True)
class Propulsor(EnergyConverter):

    converters: dataclass = field(default_factory=lambda: make_dataclass('PropulsorConverters', []))

    def compute_thrust(self):
        raise NotImplementedError("Subclasses must implement this method")

# ----------------------------------------------------------------------------------------------------------------------
# Jet Engines
# ----------------------------------------------------------------------------------------------------------------------


@dataclass(kw_only=True)
class JetEngine(Propulsor):

    name: str = 'Jet'

    fuel:                   rcl.Propellant  = field(default_factory=lambda: rcl.Components.Energy.Fuel())
    design_fuel_air_ratio:  float           = 0.0

    working_fluid:          rcl.Air         = field(default_factory=lambda: rcl.Air())

    engine_length:          float           = 0.0

    def __post_init__(self):

        self.converters.combustor = FlowConverter(name='Combustor')


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

    def __post_init__(self):
        super().__post_init__()

        self.converters.fan = FlowConverter(name='Fan')

        self.converters.fan_nozzle = FlowConverter(name='Fan Outlet Nozzle')