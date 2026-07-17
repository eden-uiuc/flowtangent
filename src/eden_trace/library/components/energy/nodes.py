# Trace/Library/Components/Energy/Nodes.py
# (c) Copyright 2026 Aerospace Research Community LLC
#
# Created: Jun 2026, J. Smart
# Modified: Jun 2026, J. Smart

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Callable, Iterable, get_args, cast

import equinox as eqx
import jax.numpy as jnp

# --- Framework Imports (Strictly for Type Hinting to avoid Circular Imports) ---
if TYPE_CHECKING:
    from eden_trace.framework.settings import Settings
    from eden_trace.framework.state import State
    from eden_trace.framework.systems import System

from eden_trace.utils import init_field, register

from eden_trace.library import Component
from eden_trace.library.gases import Air, IdealGas

# ----------------------------------------------------------------------------------------------------------------------
#  Graph Nodes
# ----------------------------------------------------------------------------------------------------------------------

# Inputs & Nodes ---------------------------------------------------------------

@register
class Efficiencies(eqx.Module):
    total: float = 1.0

    mechanical: float = 1.0
    electrical: float = 1.0
    fuel: float = 1.0
    flow: float = 1.0
    force: float = 1.0


GraphDomain = Literal["flow", "mechanical", "electrical", "fuel", "force", "residual"]

@register
class GraphInput(eqx.Module):
    domain: GraphDomain = init_field("flow", static=True)
    network_ID: str = init_field("network", static=True)

    # Define iter to make castable to tuple as (self,)
    def __iter__(self):
        yield self

@register
class GraphNode(Component):
    network_ID: str = init_field("energy_node", static=True)

    efficiencies: Efficiencies = init_field(Efficiencies)

    inputs: tuple[GraphInput, ...] | GraphInput = init_field(tuple, static=True)

    def __post_init__(self):
        if isinstance(self.inputs, GraphInput):
            object.__setattr__(self, "inputs", (self.inputs,))
    

    def __getattr__(self, item: str):
        if item.endswith("_inputs"):
            domain = item.replace("_inputs", "")
            return self._get_inputs_by_domain(domain)
        else:
            return super(GraphNode, self).__getattr__(item)
    
    @eqx.filter_jit
    def _get_inputs_by_domain(self, domain: GraphDomain | str):
        return tuple(filter(lambda i: i.domain == domain, cast(tuple, self.inputs)))

    @eqx.filter_jit
    def get_input_state(self, state: State, input: GraphInput, input_field: str):
        return getattr(getattr(state.energy.nodes[input.network_ID].outputs, input.domain), input_field)

    @eqx.filter_jit
    def get_input_states(self, state: State, inputs: Iterable[GraphInput]):
        return [
            getattr(state.energy.nodes[i.network_ID].outputs, i.domain)
            for i in inputs
        ]
    
    @eqx.filter_jit
    def _get_input_array(self, state: State, inputs: Iterable[GraphInput], input_field: str):
        input_conditions = self.get_input_states(state, inputs)
        input_values = [jnp.asarray(getattr(inp, input_field)) for inp in input_conditions]
        return jnp.concatenate([jnp.atleast_2d(v) for v in input_values if v.size > 0], axis=-1)
    
    # Input Operations
    @eqx.filter_jit
    def apply_input_op(self, arr_func:Callable, state:State, inputs: Iterable[GraphInput], input_field:str):
        input_arr = self._get_input_array(state, inputs, input_field)
        return jnp.atleast_2d(arr_func(input_arr, axis=-1)).T
    
    @eqx.filter_jit
    def apply_domain_op(self, arr_func:Callable, state:State, domain: GraphDomain, input_field:str):
        inputs = self._get_inputs_by_domain(domain)
        input_arr = self._get_input_array(state, inputs, input_field)
        return jnp.atleast_2d(arr_func(input_arr, axis=-1)).T
    


    def transmit(self, state: State, system: System, settings: Settings):
        raise NotImplementedError(
            "No transmission method implemented. "
            "Subclasses of EnergyNode must implement their individual transmission methods."
        )

# Splitters --------------------------------------------------------------------

@register
class GraphSplitter(GraphNode):
    fraction: float | Callable[[State], float|jnp.ndarray] = init_field(1.0, as_value=True, static=True)

    values: tuple[str] = init_field(("mass_flow_rate",), static=True)
    _domain: str = init_field("flow", static=True)

    def __post_init__(self):
        super(GraphSplitter, self).__post_init__()
        assert(isinstance(self.inputs, tuple))
        assert len(self.inputs) == 1, f"Graph splitters can only have one input. Found: {self.inputs}"
        for splitter in get_args(GraphDomain):
            if len(getattr(self, splitter + "_inputs")) > 0:
                self._domain = splitter

    def transmit(self, state: State, system: System, settings: Settings):

        total_input = getattr(state.energy.nodes[tuple(self.inputs)[0].network_ID].outputs, self._domain)

        if callable(self.fraction):
            frac = self.fraction(state)
        else:
            frac = self.fraction

        extracted_input = eqx.tree_at(
            lambda t: tuple(getattr(t, s) for s in self.values),
            total_input,
            tuple(getattr(total_input, s) * frac for s in self.values),
        )

        updated_state = eqx.tree_at(
            lambda s: getattr(s.energy.nodes[self.network_ID].outputs, self._domain), state, extracted_input
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
class FlowNode[DesignType: FlowDesign](GraphNode):
    
    design_parameters: DesignType = init_field(FlowDesign)
    working_fluid: IdealGas = init_field(Air)

# ----------------------------------------------------------------------------------------------------------------------
# Energy Store
# ----------------------------------------------------------------------------------------------------------------------

@register
class EnergyStore(GraphNode):
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

if __name__ == "__main__":

    doms = get_args(GraphDomain)
    for d in doms:
        print(d)