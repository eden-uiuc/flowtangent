from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from RCAIDE.Framework import Settings, State, System

import equinox as eqx
import jax.numpy as jnp

from RCAIDE.utils import init_field, inputs, outputs

from .nodes import EnergyInput, EnergyNode, EnergySplitter, EnergyStore, FuelTank
from .propulsors import Propulsor, TurbojetEngine

# ----------------------------------------------------------------------------------------------------------------------
#  Energy Line
# ----------------------------------------------------------------------------------------------------------------------

class EnergyLine(EnergyNode):
    _bookkeeping: dict = init_field(
        lambda: {
            "propulsors": Propulsor,
            "splitters": EnergySplitter,
            "stores": EnergyStore,
        },
        static=True,
    )

# ----------------------------------------------------------------------------------------------------------------------
#  Jets
# ----------------------------------------------------------------------------------------------------------------------
def _TurbojetLineSetup():
    E1 = TurbojetEngine(tag="Engine 1")

    E2 = TurbojetEngine(tag="Engine 2")

    tank = FuelTank()

    return (E1, E2, tank)


class TurbojetEnergyLine(EnergyLine):
    tag: str = init_field("Turbojet Energy Line", static=True)

    fuel_inputs: tuple[str, ...] = init_field((
        EnergyInput("fuel", "self.engine_1"),
        EnergyInput("fuel", "self.engine_2")),
    static=True)

    tank_draw_ratios: tuple[float, ...] = init_field((1.0,))

    subcomponents: tuple = init_field(_TurbojetLineSetup())

    _bookkeeping: dict = init_field(
        lambda: {"engines": TurbojetEngine, "stores": FuelTank, "fuel_tanks": FuelTank},
        static=True,
    )

    @inputs(
        "state.energy.nodes[Line_fuel_tanks].mass",
        "state.energy.nodes[Line_fuel_inputs].outputs.fuel.flow_rate",
        "system.energy.nodes[Line_fuel_tanks].selector_ratio",
        "system.energy.nodes[Line_fuel_tanks].mass_properties.total",
        "system.energy.nodes[Line].tank_draw_ratios",
    )
    @outputs("state.energy.nodes[Line_fuel_tanks].outputs.fuel.flow_rate")
    def transmit(self, state: State, system: System, settings: Settings):

        # Manage Fuel --------------------------------------------------------------------------------------------------
        total_fuel_burn = self.sum_inputs(state, "fuel", "flow_rate")

        #  Compute fuel fraction
        total_fuel_mass = jnp.sum(jnp.asarray([t.mass_properties.total for t in self.fuel_tanks]))
        current_fuel_mass = jnp.sum(jnp.asarray([state.energy.nodes[t.network_ID].mass for t in self.fuel_tanks]))
        fuel_fraction = current_fuel_mass / total_fuel_mass

        # Extract configuration as pure JAX arrays
        selector_ratios = jnp.asarray([t.selector_ratio for t in self.fuel_tanks])
        baseline_draws = jnp.asarray([self.tank_draw_ratios[i] for i in range(len(self.fuel_tanks))])

        # Create the active mask (1.0 if active, 0.0 if inactive)
        active_mask = jnp.where(selector_ratios[None, :] >= fuel_fraction, 1.0, 0.0)

        # Mask the baseline draws
        masked_draws = baseline_draws * active_mask

        # Normalize the draws (with a safeguard against division-by-zero if all tanks are inactive)
        sum_draws = jnp.sum(masked_draws)
        safe_sum = jnp.where(sum_draws == 0.0, 1.0, sum_draws)
        balanced_draws = masked_draws / safe_sum

        # Distribute the burn across ALL tanks (inactive ones get multiplied by 0.0)
        tank_burns = tuple(-balanced_draws[i] * total_fuel_burn for i in range(len(self.fuel_tanks)))

        # Apply updates sequentially
        updated_state = eqx.tree_at(
            lambda s: tuple(s.energy.nodes[t.network_ID].outputs.fuel.flow_rate for t in self.fuel_tanks),
            state,
            tank_burns,
        )

        updated_state = eqx.tree_at(
            lambda s: s.mass.rate_of_change, updated_state, updated_state.mass.rate_of_change - total_fuel_burn
        )

        # # Manage Electrical Power --------------------------------------------------------------------------------------

        # for idx, offtake in enumerate(self.offtakes):  # type: ignore
        #     offtake: OfftakeShaft
        #     engine_ID = self.engines[idx].network_ID  # type: ignore

        #     updated_state = eqx.tree_at(
        #         lambda s: (
        #             s.energy.nodes[offtake.network_ID].outputs.electical.power,
        #             s.energy.nodes[offtake.network_ID].outputs.mechanical.work,
        #         ),
        #         updated_state,
        #         (offtake.power_draw, offtake.power_draw / state.energy.nodes[engine_ID].outputs.flow.mass_flow_rate),
        #     )

        return updated_state, system, settings
