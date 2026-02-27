# RCAIDE/Library/Components/Energy/Networks/Jets.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: Apr 2025, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from __future__ import annotations

# RCAIDE imports
from RCAIDE.Library.Components.Energy.Networks import EnergyLine
from RCAIDE.Library.Methods.Energy.Converters.Turbofans import thrust_and_power

from RCAIDE.Framework import State, Aircraft, Settings

# ----------------------------------------------------------------------------------------------------------------------
#  Jets
# ----------------------------------------------------------------------------------------------------------------------


class TurbofanEnergyLine(EnergyLine):

    name = 'Turbofan Energy Line'

    @staticmethod
    def calculate_performance(state: State,     #type: ignore
                              system: Aircraft,
                              settings: Settings
                              ): 

        state, system, settings = thrust_and_power(state, system, settings)

        return state, system, settings

