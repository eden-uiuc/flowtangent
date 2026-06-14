# RCAIDE/Library/Components/Energy/Nodes.py
# (c) Copyright 2026 Aerospace Research Community LLC
#
# Created: Jun 2026, J. Smart
# Modified: Jun 2026, J. Smart

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from __future__ import annotations
from typing import TYPE_CHECKING, Literal, Optional
import jax
import jax.numpy as jnp
import equinox as eqx

# --- Framework Imports (Strictly for Type Hinting to avoid Circular Imports) ---
if TYPE_CHECKING:
    from RCAIDE.Framework.State import State
    from RCAIDE.Framework.Systems import System
    from RCAIDE.Framework.Settings import Settings

from RCAIDE.utils import init_field
from RCAIDE.Library import Component

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

class EnergyNode(Component):
    
    network_ID: str = init_field("energy_node", static=True)
    
    efficiencies:   EnergyEfficiencies  = init_field(EnergyEfficiencies)

    mechanical_inputs: tuple[str, ...] = init_field(tuple, static=True)
    electrical_inputs: tuple[str, ...] = init_field(tuple, static=True)
    fuel_inputs:       tuple[str, ...] = init_field(tuple, static=True)
    flow_inputs:       tuple[str, ...] = init_field(tuple, static=True)
    force_inputs:      tuple[str, ...] = init_field(tuple, static=True)
    
    @property
    def inputs(self):
        return (self.mechanical_inputs + self.electrical_inputs + self.fuel_inputs
                + self.flow_inputs + self.force_inputs) #type: ignore
    
    @eqx.filter_jit
    def _get_all_inputs(self, state, input_type: str, input_field: str):
        output_conditions = [getattr(state.energy.nodes[i].outputs, input_type) for i in getattr(self, f"{input_type}_inputs")]
        return jnp.concatenate([getattr(out, input_field) for out in output_conditions], axis=-1)

    @eqx.filter_jit
    def sum_inputs(self, state, input_type: str, input_field: str):
        all_inputs = self._get_all_inputs(state, input_type, input_field)
        return jnp.atleast_2d(jnp.sum(all_inputs, axis=-1))

    @eqx.filter_jit
    def average_inputs(self, state, input_type: str, input_field: str):
        all_inputs = self._get_all_inputs(state, input_type, input_field)
        return jnp.atleast_2d(jnp.mean(all_inputs, axis=-1))

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