# RCAIDE/Library/Components/Energy/Stores.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: May, 2025, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
# IMPORT 
# ----------------------------------------------------------------------------------------------------------------------

from __future__ import annotations
from typing import TYPE_CHECKING
import jax
import jax.numpy as jnp
import equinox as eqx

# --- Framework Imports (Strictly for Type Hinting to avoid Circular Imports) ---
if TYPE_CHECKING:
    from RCAIDE.Framework.State import State
    from RCAIDE.Framework.System import System
    from RCAIDE.Framework.Settings import Settings
# RCAIDE imports
from RCAIDE.utils import init_field
from RCAIDE.Library import Component, MassProperties

from .Nodes import EnergyNode

# ----------------------------------------------------------------------------------------------------------------------
# Energy Store
# ----------------------------------------------------------------------------------------------------------------------

class EnergyStore(EnergyNode):

    tag: str = init_field('Energy Store', static=True)

    max_energy: float = 0.0

    specific_energy: float = 0.0
    specific_volume: float = 0.0   


# ----------------------------------------------------------------------------------------------------------------------
# Fuel Tank
# ----------------------------------------------------------------------------------------------------------------------

class FuelTank(EnergyStore):

    tag = init_field('Fuel Tank', static=True)

    selector_ratio:         float = 1.0
    secondary_fuel_flow:    float = 0.0

    def transmit(
            self,
            state: State,
            system: System,
            settings: Settings,
    ):  
        return state, system, settings

# ----------------------------------------------------------------------------------------------------------------------
# Battery
# ----------------------------------------------------------------------------------------------------------------------

class BatteryRagoneParameters(eqx.Module):

    const_1: float = 0.0
    const_2: float = 0.0
    lower_bound: float = 0.0
    i: float = 0.0


class Battery(EnergyStore):

    tag: str = init_field('Battery', static=True)

    max_energy:     float = 0.0
    max_power:      float = 0.0
    max_voltage:    float = 0.0

    resistance:     float = 0.0

    ragone: BatteryRagoneParameters = init_field(BatteryRagoneParameters)