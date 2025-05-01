# RCAIDE/Library/Components/Energy/Stores.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: May, 2025, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
# IMPORT 
# ----------------------------------------------------------------------------------------------------------------------

from dataclasses import dataclass, field, make_dataclass

import RCAIDE.Library as rcl

# ----------------------------------------------------------------------------------------------------------------------
# Fuel Tank
# ----------------------------------------------------------------------------------------------------------------------


@dataclass(kw_only=True)
class FuelTank(rcl.Component):

    name = 'Fuel Tank'

    fuel_selector_ratio: float = 1.0
    secondary_fuel_flow: float = 0.0

    def __post_init__(self):

        self.mass_properties.full_fuel_mass     = 0.0
        self.mass_properties.full_fuel_volume   = 0.0

# ----------------------------------------------------------------------------------------------------------------------
# Battery
# ----------------------------------------------------------------------------------------------------------------------


@dataclass(kw_only=True)
class BatteryRagoneParameters:

    const_1: float = 0.0
    const_2: float = 0.0
    lower_bound: float = 0.0
    i: float = 0.0


@dataclass(kw_only=True)
class Battery(rcl.Component):

    name = 'Battery'

    max_energy:     float = 0.0
    max_power:      float = 0.0
    max_voltage:    float = 0.0

    energy_density: float = 0.0
    resistance:     float = 0.0

    ragone: BatteryRagoneParameters = field(default_factory=BatteryRagoneParameters)