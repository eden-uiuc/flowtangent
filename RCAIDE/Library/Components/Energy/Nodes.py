# RCAIDE/Library/Components/Energy/Nodes.py
# (c) Copyright 2026 Aerospace Research Community LLC
#
# Created: Jun 2026, J. Smart
# Modified: Jun 2026, J. Smart

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from __future__ import annotations
from typing import TYPE_CHECKING, Literal

import jax.numpy as jnp
import equinox as eqx

# --- Framework Imports (Strictly for Type Hinting to avoid Circular Imports) ---
if TYPE_CHECKING:
    from RCAIDE.Framework.State import State
    from RCAIDE.Framework.Systems import System
    from RCAIDE.Framework.Settings import Settings

from RCAIDE.utils import init_field
from RCAIDE.Library import Component, Units
from RCAIDE.Library.Gases import IdealGas, Air

# ----------------------------------------------------------------------------------------------------------------------
#  Energy Nodes
# ----------------------------------------------------------------------------------------------------------------------

class EnergyEfficiencies(eqx.Module):

    total: float = 1.0

    mechanical: float = 1.0
    electrical: float = 1.0
    fuel:       float = 1.0
    flow:       float = 1.0
    force:      float = 1.0

EnergyDomain = Literal[
    "flow",
    "mechanical",
    "electrical",
    "fuel",
    "force"
]

class EnergyInput(eqx.Module):

    domain: EnergyDomain
    network_ID: str

class EnergyNode(Component):

    network_ID: str = init_field("energy_node", static=True)

    efficiencies:   EnergyEfficiencies  = init_field(EnergyEfficiencies)

    inputs: tuple[EnergyInput, ...] = init_field(tuple, static=True)

    def _get_inputs_by_domain(self, domain: EnergyDomain):
        return tuple(i for i in self.inputs if i.domain == domain)

    def __getattr__(self, name):
        if name.endswith("_inputs"):
            domain = name.replace("_inputs", "")
            return tuple(i.network_ID for i in self._get_inputs_by_domain(domain))
        else:
            return super().__getattribute__(name)

    @eqx.filter_jit
    def _get_all_inputs(self, state, input_type: EnergyDomain, input_field: str):
        output_conditions = [getattr(state.energy.nodes[i].outputs, input_type) for i in self._get_inputs_by_domain(input_type)]
        return jnp.concatenate([getattr(out, input_field) for out in output_conditions], axis=-1)

    @eqx.filter_jit
    def sum_inputs(self, state, input_type: EnergyDomain, input_field: str):
        all_inputs = self._get_all_inputs(state, input_type, input_field)
        return jnp.atleast_2d(jnp.sum(all_inputs, axis=-1)).T

    @eqx.filter_jit
    def average_inputs(self, state, input_type: EnergyDomain, input_field: str):
        all_inputs = self._get_all_inputs(state, input_type, input_field)
        return jnp.atleast_2d(jnp.mean(all_inputs, axis=-1)).T

    def transmit(self, state: State, system: System, settings: Settings):
        raise NotImplementedError("No transmission method implemented. " \
        "Subclasses of EnergyNode must implement their individual transmission methods.")


class EnergySplitter(EnergyNode):

    extraction_fraction: float = 1.0

    _splitter_type: str = init_field("flow", static=True)
    split_values: tuple[str] = init_field(("mass_flow_rate",), static=True)

    def __post_init__(self):
        assert len(self.inputs) == 1 , f"Energy splitters can only have one input. Found: {self.inputs}"
        for splitter in ["flow", "mechanical", "electrical", "fuel", "force"]:
            if len(getattr(self, splitter+"_inputs")) > 0:
                self._splitter_type = splitter

    def transmit(self, state: State, system: System, settings: Settings):

        total_input = getattr(state.energy.nodes[self.inputs[0]].outputs, self._splitter_type)

        extracted_input = eqx.tree_at(
            lambda t:tuple(getattr(t, s) for s in self.split_values),
            total_input,
            tuple(getattr(total_input, s) * self.extraction_fraction for s in self.split_values)
        )

        updated_state = eqx.tree_at(
            lambda s: getattr(s.energy.nodes[self.network_ID].outputs, self._splitter_type),
            state,
            extracted_input
        )

        return updated_state, system, settings

# ----------------------------------------------------------------------------------------------------------------------
#  Flow Nodes
# ----------------------------------------------------------------------------------------------------------------------

class FlowNode(EnergyNode):

    pressure_ratio:             float = 1.0
    pressure_recovery:          float = 1.0
    area_ratio:                 float = 1.0

    design_intake_temperature:  float = 298.15    # Kelvin

    rotation_speed:             float = 0.0
    noise_speed:                float = 0.0

    working_fluid:              IdealGas = init_field(Air)

# ----------------------------------------------------------------------------------------------------------------------
#  Mechanical Nodes
# ----------------------------------------------------------------------------------------------------------------------

class OfftakeShaft(EnergyNode):

    tag: str = init_field('Offtake Shaft', static=True)

    power_draw: float = 1.0 * Units.W

    reference_temperature: float = 288.15 * Units.K
    reference_pressure: float = 101325. * Units.Pa

    def transmit(self, state: State, system: System, settings: Settings):
        return state, system, settings

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

    tag: str = init_field('Fuel Tank', static=True)

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
