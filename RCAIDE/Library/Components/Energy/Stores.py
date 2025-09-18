# RCAIDE/Library/Components/Energy/Stores.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: May, 2025, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
# IMPORT 
# ----------------------------------------------------------------------------------------------------------------------

import chex
from dataclasses import field, make_dataclass

import RCAIDE.Library as rcl

# ----------------------------------------------------------------------------------------------------------------------
# Energy Store
# ----------------------------------------------------------------------------------------------------------------------


@chex.dataclass(kw_only=True)
class EnergyStore(rcl.Component):

    name = 'Energy Store'

    max_energy: float = 0.0

    specific_energy: float = 0.0
    specific_volume: float = 0.0

    def __post_init__(self):
        if self.mass_properties.total and not self.specific_energy:
            self.specific_energy = self.max_energy / self.mass_properties.total
        if self.mass_properties.volume and not self.specific_volume:
            self.specific_volume = self.max_energy / self.mass_properties.volume

# ----------------------------------------------------------------------------------------------------------------------
# Fuel Tank
# ----------------------------------------------------------------------------------------------------------------------


@chex.dataclass(kw_only=True)
class FuelTank(EnergyStore):

    name = 'Fuel Tank'

    fuel_selector_ratio: float = 1.0
    secondary_fuel_flow: float = 0.0

    def __post_init__(self):

        self.mass_properties.full_fuel_mass     = 0.0
        self.mass_properties.full_fuel_volume   = 0.0

# ----------------------------------------------------------------------------------------------------------------------
# Battery
# ----------------------------------------------------------------------------------------------------------------------


@chex.dataclass(kw_only=True)
class BatteryRagoneParameters:

    const_1: float = 0.0
    const_2: float = 0.0
    lower_bound: float = 0.0
    i: float = 0.0


@chex.dataclass(kw_only=True)
class Battery(EnergyStore):

    name = 'Battery'

    max_energy:     float = 0.0
    max_power:      float = 0.0
    max_voltage:    float = 0.0

    resistance:     float = 0.0

    ragone: BatteryRagoneParameters = field(default_factory=BatteryRagoneParameters)