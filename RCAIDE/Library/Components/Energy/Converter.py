# RCAIDE/Library/Components/Energy/Converter.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: Apr 2025, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from dataclasses import dataclass, field

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


@dataclass(kw_only=True)
class FlowConverter(EnergyConverter):

    mechanical_efficiency:  float = field(default=1.0)
    polytropic_efficiency:  float = field(default=1.0)

    pressure_ratio:         float = field(default=1.0)


@dataclass(kw_only=True)
class Turbine(FlowConverter):

    intake_temperature:  float = field(default=298.15)    # Kelvin

@dataclass(kw_only=True)
class OfftakeShaft(EnergyConverter):

    power_draw:             float = field(default=0.0)
    reference_temperature:  float = field(default=298.15)    # Kelvin
    reference_pressure:     float = field(default=101325.0)  # Pascal

@dataclass(kw_only=True)
class JetCombustor(EnergyConverter):
