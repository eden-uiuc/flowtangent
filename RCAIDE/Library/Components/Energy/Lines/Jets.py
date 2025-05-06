# RCAIDE/Library/Components/Energy/Networks/Jets.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: Apr 2025, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from dataclasses import dataclass

# RCAIDE imports
import RCAIDE.Framework as rcf
import RCAIDE.Library as rcl
from RCAIDE.Library.Components.Energy import EnergyLine

# ----------------------------------------------------------------------------------------------------------------------
#  Jets
# ----------------------------------------------------------------------------------------------------------------------


@dataclass(kw_only=True)
class JetEnergyLine(EnergyLine):

    name = 'Jet Network'

    def __post_init__(self):
        self.converters.engine_1 = rcl.Components.Energy.Converters.JetEngine()

        self.stores.fuel_tank = rcl.Components.Energy.Stores.FuelTank()


@dataclass(kw_only=True)
class TurbojetEnergyLine(EnergyLine):

    name = 'Turbojet Network'

    def __post_init__(self):
        self.converters.engine_1 = rcl.Components.Energy.Converters.TurbojetEngine()

        self.stores.fuel_tank = rcl.Components.Energy.Stores.FuelTank()


@dataclass(kw_only=True)
class TurbofanEnergyLine(EnergyLine):

    name = 'Turbofan Network'

    def __post_init__(self):
        self.converters.engine_1 = rcl.Components.Energy.Converters.TurbofanEngine()

        self.stores.fuel_tank = rcl.Components.Energy.Stores.FuelTank()

    @staticmethod
    def calculate_performance(state: "rcf.State",
                              system: "rcf.System",
                              settings: "rcf.Settings"
                              ):

        state, system, settings = rcl.Methods.Propulsors.Turbofan.thrust_and_power(state, system, settings)

        return state, system, settings

