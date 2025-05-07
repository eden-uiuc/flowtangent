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
from RCAIDE.Library.Components.Energy.Networks import EnergyLine

# ----------------------------------------------------------------------------------------------------------------------
#  Jets
# ----------------------------------------------------------------------------------------------------------------------


@dataclass(kw_only=True)
class JetEnergyLine(EnergyLine):

    name = 'Jet Network'


@dataclass(kw_only=True)
class TurbojetEnergyLine(EnergyLine):

    name = 'Turbojet Network'


@dataclass(kw_only=True)
class TurbofanEnergyLine(EnergyLine):

    name = 'Turbofan Network'

    @staticmethod
    def calculate_performance(state: "rcf.State",
                              system: "rcf.System",
                              settings: "rcf.Settings"
                              ):

        state, system, settings = rcl.Methods.Energy.Converters.Turbofans.thrust_and_power(state, system, settings)

        return state, system, settings

