# RCAIDE/Library/Components/Energy/Networks/Jets.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: Apr 2025, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from __future__ import annotations

# package import
import equinox as eqx

# RCAIDE imports
from RCAIDE.utils import init_field
from RCAIDE.Library.Components.Energy.Networks import EnergyLine
from RCAIDE.Library.Methods.Energy.Transmission.Turbofans import thrust_and_power


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from RCAIDE.Framework import State, Aircraft, Settings

# ----------------------------------------------------------------------------------------------------------------------
#  Jets
# ----------------------------------------------------------------------------------------------------------------------


class TurbofanEnergyLine(EnergyLine):

    tag: str = init_field('Turbofan Energy Line', static=True)

    @staticmethod
    def calculate_performance(state: State,     #type: ignore
                              system: Aircraft,
                              settings: Settings
                              ): 

        state, system, settings = thrust_and_power(state, system, settings)

        return state, system, settings

