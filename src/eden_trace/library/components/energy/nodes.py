# Trace/Library/Components/Energy/Nodes.py
# (c) Copyright 2026 Aerospace Research Community LLC
#
# Created: Jun 2026, J. Smart
# Modified: Jun 2026, J. Smart

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import equinox as eqx
import jax.numpy as jnp

# --- Framework Imports (Strictly for Type Hinting to avoid Circular Imports) ---
if TYPE_CHECKING:
    from src.eden_trace.framework.settings import Settings
    from src.eden_trace.framework.state import State
    from src.eden_trace.framework.systems import System

from src.eden_trace.utils import init_field, register

from src.eden_trace.library import Component
from src.eden_trace.library.gases import Air, IdealGas

# ----------------------------------------------------------------------------------------------------------------------
#  Energy Nodes
# ----------------------------------------------------------------------------------------------------------------------

@register
class EnergyEfficiencies(eqx.Module):
    total: float = 1.0

    mechanical: float = 1.0
    electrical: float = 1.0
    fuel: float = 1.0
    flow: float = 1.0
    force: float = 1.0


EnergyDomain = Literal["flow", "mechanical", "electrical", "fuel", "force", "residual"]

@register
class EnergyInput(eqx.Module):
    domain: EnergyDomain = init_field("flow", static=True)
    network_ID: str = init_field("network", static=True)

@register
class EnergyNode(Component):
    network_ID: str = init_field("energy_node", static=True)

    efficiencies: EnergyEfficiencies = init_field(EnergyEfficiencies)

    inputs: tuple[EnergyInput, ...] = init_field(tuple, static=True)

    

    def __getattr__(self, item: str):
        if item.endswith("_inputs"):
            domain = item.replace("_inputs", "")
            return tuple(i.network_ID for i in self._get_inputs_by_domain(domain))
        else:
            return super(EnergyNode, self).__getattr__(item)
    
    @eqx.filter_jit
    def get_input(self, state, input: EnergyInput, input_field: str):
        return getattr(getattr(state.energy.nodes[input.network_ID].outputs, input.domain), input_field)

    @eqx.filter_jit
    def _get_inputs_by_domain(self, domain: EnergyDomain):
        return tuple(i for i in self.inputs if i.domain == domain)

    @eqx.filter_jit
    def get_domain_inputs(self, state, input_type: EnergyDomain, input_field: str):
        output_conditions = [
            getattr(state.energy.nodes[i.network_ID].outputs, input_type)
            for i in self._get_inputs_by_domain(input_type)
        ]
        input_values = [jnp.asarray(getattr(out, input_field)) for out in output_conditions]
        return jnp.concatenate([jnp.atleast_2d(v) for v in input_values if v.size > 0], axis=-1)

    @eqx.filter_jit
    def sum_domain_inputs(self, state, input_type: EnergyDomain, input_field: str):
        all_inputs = self.get_domain_inputs(state, input_type, input_field)
        return jnp.atleast_2d(jnp.sum(all_inputs, axis=-1)).T
        

    @eqx.filter_jit
    def diff_domain_inputs(self, state, input_type: EnergyDomain, input_field:str):
        all_inputs = self.get_domain_inputs(state, input_type, input_field)
        return jnp.atleast_2d(jnp.diff(all_inputs, axis=-1)).T
    
    @eqx.filter_jit
    def average_domain_inputs(self, state, input_type: EnergyDomain, input_field: str):
        all_inputs = self.get_domain_inputs(state, input_type, input_field)
        return jnp.atleast_2d(jnp.mean(all_inputs, axis=-1)).T

    def transmit(self, state: State, system: System, settings: Settings):
        raise NotImplementedError(
            "No transmission method implemented. "
            "Subclasses of EnergyNode must implement their individual transmission methods."
        )

@register
class EnergySplitter(EnergyNode):
    extraction_fraction: float = 1.0

    _splitter_type: str = init_field("flow", static=True)
    split_values: tuple[str] = init_field(("mass_flow_rate",), static=True)

    def __post_init__(self):
        assert len(self.inputs) == 1, f"Energy splitters can only have one input. Found: {self.inputs}"
        for splitter in ["flow", "mechanical", "electrical", "fuel", "force"]:
            if len(getattr(self, splitter + "_inputs")) > 0:
                self._splitter_type = splitter

    def transmit(self, state: State, system: System, settings: Settings):

        total_input = getattr(state.energy.nodes[self.inputs[0]].outputs, self._splitter_type)

        extracted_input = eqx.tree_at(
            lambda t: tuple(getattr(t, s) for s in self.split_values),
            total_input,
            tuple(getattr(total_input, s) * self.extraction_fraction for s in self.split_values),
        )

        updated_state = eqx.tree_at(
            lambda s: getattr(s.energy.nodes[self.network_ID].outputs, self._splitter_type), state, extracted_input
        )

        return updated_state, system, settings


# ----------------------------------------------------------------------------------------------------------------------
#  Flow Nodes
# ----------------------------------------------------------------------------------------------------------------------
@register
class FlowDesign(eqx.Module):
    pressure_ratio: float = 1.0
    pressure_recovery: float = 1.0
    
    intake_temperature: float = 298.15
    output_temperature: float = 298.15
    
    A_ratio: float = 1.0
    A_intake: float = 1.0
    A_throat: float = 1.0
    A_exit: float = 1.0

    exit_mach_number: float = 0.5
    
    rotation_speed: float = 0.0
    noise_speed: float = 0.0

@register
class FlowNode(EnergyNode):
    
    design_parameters: FlowDesign = init_field(FlowDesign)
    working_fluid: IdealGas = init_field(Air)


# ----------------------------------------------------------------------------------------------------------------------
# Energy Store
# ----------------------------------------------------------------------------------------------------------------------

@register
class EnergyStore(EnergyNode):
    tag: str = init_field("Energy Store", static=True)

    max_energy: float = 0.0

    specific_energy: float = 0.0
    specific_volume: float = 0.0


# ----------------------------------------------------------------------------------------------------------------------
# Fuel Tank
# ----------------------------------------------------------------------------------------------------------------------

@register
class FuelTank(EnergyStore):
    tag: str = init_field("Fuel Tank", static=True)

    selector_ratio: float = 1.0
    secondary_fuel_flow: float = 0.0

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

@register
class BatteryRagoneParameters(eqx.Module):
    const_1: float = 0.0
    const_2: float = 0.0
    lower_bound: float = 0.0
    i: float = 0.0

@register
class Battery(EnergyStore):
    tag: str = init_field("Battery", static=True)

    max_energy: float = 0.0
    max_power: float = 0.0
    max_voltage: float = 0.0

    resistance: float = 0.0

    ragone: BatteryRagoneParameters = init_field(BatteryRagoneParameters)
