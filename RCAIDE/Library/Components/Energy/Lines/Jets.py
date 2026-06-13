# RCAIDE/Library/Components/Energy/Networks/Jets.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: Apr 2025, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from RCAIDE.Framework import State, System, Settings

# package import
import jax.numpy as jnp
import equinox as eqx

# RCAIDE imports
from RCAIDE.utils import init_field, inputs, outputs

from RCAIDE.Library.Components.Energy.Stores import FuelTank
from RCAIDE.Library.Components.Energy.Networks import EnergyLine
from RCAIDE.Library.Components.Energy.Propulsors import TurbojetEngine, TurbofanEngine



from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from RCAIDE.Framework import State, Aircraft, Settings

# ----------------------------------------------------------------------------------------------------------------------
#  Jets
# ----------------------------------------------------------------------------------------------------------------------

def _TurbojetLineSetup():
    E1 = TurbofanEngine(tag="Engine 1")
    E2 = TurbofanEngine(tag="Engine 2")

    tank = FuelTank()

    return (E1, E2, tank)


class TurbojetEnergyLine(EnergyLine):

    tag:       str = init_field('Turbojet Energy Line', static=True)

    fuel_inputs = ["Engine 1", "Engine 2"]
    
    tank_draw_ratios: tuple[float, ...] = init_field((1.0,))
    
    subcomponents = init_field(_TurbojetLineSetup())
    
    _bookkeeping = {
        "engines": TurbojetEngine,
        "stores": FuelTank,
        "fuel_tanks": FuelTank,
    }

    def __post_init__(self):
        if len(self.tank_draw_ratios) != len(self.fuel_tanks):
            # If draw ratios not specified, balance fuel draw by tank mass
            object.__setattr__(self, "tank_draw_ratios", tuple(t.mass_properties.total for t in self.fuel_tanks))

    @inputs(
            "state.energy.nodes[Line_fuel_tanks].mass",
            "state.energy.nodes[Line_fuel_inputs].outputs.fuel.flow_rate",
            "system.energy.nodes[Line_fuel_tanks].selector_ratio",
            "system.energy.nodes[Line_fuel_tanks].mass_properties.total",
            "system.energy.nodes[Line].tank_draw_ratios"
    )
    @outputs(
        "state.energy.nodes[Line_fuel_tanks].outputs.fuel.flow_rate"
    )
    def transmit(self, state: State, system: System, settings: Settings):
        total_fuel_burn = self.sum_inputs(state, "fuel", "flow_rate")
        
        total_fuel_mass = jnp.sum(jnp.asarray([t.mass_properties.total for t in self.fuel_tanks]))
        current_fuel_mass = jnp.sum(jnp.asarray([state.energy.nodes[t.tag].mass for t in self.fuel_tanks]))
        fuel_fraction = current_fuel_mass / total_fuel_mass

        selector_ratios = {t: t.selector_ratio for t in self.fuel_tanks}
        active_tanks = tuple(t for t, r in selector_ratios.items() if r >= fuel_fraction)
        
        active_draws = jnp.asarray([self.tank_draw_ratios[self.fuel_tanks.subcomponents.index(t)] for t in active_tanks])
        balanced_draws = active_draws/jnp.sum(active_draws)
        
        updated_state = eqx.tree_at(
            lambda s: tuple(s.energy.nodes[t.tag].outputs.fuel.flow_rate for t in self.fuel_tanks),
            state,
            tuple(-balanced_draws[i] * total_fuel_burn for i in range(len(balanced_draws)))
        )

        return updated_state, system, settings

