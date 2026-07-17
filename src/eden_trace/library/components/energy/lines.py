from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eden_trace.framework import Settings, State, System

import equinox as eqx
import jax.numpy as jnp

from eden_trace.utils import init_field, outputs, register
from eden_trace.utils import inputs as func_inputs

from .nodes import GraphInput, GraphNode, GraphSplitter, EnergyStore, FuelTank
from .jets import TurbojetEngine, TurbofanEngine

# ----------------------------------------------------------------------------------------------------------------------
#  Energy Line
# ----------------------------------------------------------------------------------------------------------------------

@register
class EnergyLine(GraphNode):
    tag: str = init_field("Line", static=True)
    _bookkeeping: dict = init_field(
        lambda: {
            "splitters": GraphSplitter,
            "stores": EnergyStore,
        },
        static=True,
    )


# ----------------------------------------------------------------------------------------------------------------------
#  Jets
# ----------------------------------------------------------------------------------------------------------------------

# Turbojet ---------------------------------------------------------------------

def _TurbojetLineSetup():
    return TurbojetEngine(), FuelTank()

@register
class TurbojetLine(EnergyLine):

    subcomponents: tuple = init_field(_TurbojetLineSetup)
    
    inputs: tuple | GraphInput = init_field(
        (
            GraphInput("fuel", "self.engine"),
            GraphInput("force", "self.engine"),
            GraphInput("residual", "self.engine")
        ),
        static=True,
    )

    tank_draw_ratios: tuple[float, ...] = init_field((1.0,))

    _bookkeeping: dict = init_field(
        lambda: {
            "engines": TurbojetEngine,
            "stores": FuelTank,
            "fuel_tanks": FuelTank
        },
        static=True,
    )

    @func_inputs(
        "state.energy.nodes[Line_fuel_tanks].mass",
        "state.energy.nodes[Line_fuel_inputs].outputs.fuel.flow_rate",
        "system.energy.nodes[Line_fuel_tanks].selector_ratio",
        "system.energy.nodes[Line_fuel_tanks].mass_properties.total",
        "system.energy.nodes[Line].tank_draw_ratios",
    )
    @outputs("state.energy.nodes[Line_fuel_tanks].outputs.fuel.flow_rate")
    def transmit(self, state: State, system: System, settings: Settings):

        # Fuel Burn ------------------------------------------------------------
        total_fuel_burn = self.apply_domain_op(jnp.sum, state, "fuel", "flow_rate")

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

        # Total Thrust ---------------------------------------------------------

        updated_state = eqx.tree_at(
            lambda s: s.energy.nodes[self.network_ID].outputs.force.thrust,
            updated_state,
            self.apply_domain_op(jnp.sum, updated_state, "force", "thrust")
        )

        # Mass & Work Imbalance ------------------------------------------------

        updated_state = eqx.tree_at(
            lambda s: (
                s.energy.nodes[self.network_ID].outputs.residual.mass_flow_rate,
                s.energy.nodes[self.network_ID].outputs.residual.power,
            ),
            updated_state, (
                self.apply_domain_op(jnp.sum, updated_state, "residual", "mass_flow_rate"),
                self.apply_domain_op(jnp.sum, updated_state, "residual", "power"),
            )
        )

        return updated_state, system, settings


# Turbofan ---------------------------------------------------------------------

def _TurbofanLineSetup():
    return TurbofanEngine(), FuelTank()

def TurbofanLine(**kwargs):
    return TurbojetLine(
        subcomponents=_TurbofanLineSetup(),
        **kwargs
    )