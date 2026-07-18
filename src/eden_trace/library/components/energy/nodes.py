# Trace/Library/Components/Energy/Nodes.py
# (c) Copyright 2026 Aerospace Research Community LLC
#
# Created: Jun 2026, J. Smart
# Modified: Jun 2026, J. Smart

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Callable, Iterable, Self, get_args, cast
# --- Framework Imports (Strictly for Type Hinting to avoid Circular Imports) ---
if TYPE_CHECKING:
    pass

import warnings
import equinox as eqx
import jax.numpy as jnp

from dataclasses import dataclass, replace
from functools import reduce
from collections import defaultdict

from eden_trace.framework.settings import Settings
from eden_trace.framework.state import State
from eden_trace.framework.systems import System
from eden_trace.utils import init_field, register

from eden_trace.library import Component
from eden_trace.library.gases import Air, IdealGas, MixedGas, GasComposition

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
    primary: bool = init_field(False, static=True)

    # Define iter to make castable to tuple as (self,)
    def __iter__(self):
        yield self

    def get_value(self, state:State, value: str):
        return reduce(getattr, (state.energy.nodes[self.network_ID].outputs, self.domain, value))

@register
class GraphNode(Component):
    network_ID: str = init_field("energy_node", static=True)

    efficiencies: Efficiencies = init_field(Efficiencies)

    inputs: tuple[GraphInput, ...] | GraphInput = init_field(tuple, static=True)

    def __post_init__(self):
        if isinstance(self.inputs, GraphInput):
            object.__setattr__(self, "inputs", (self.inputs,))
        
        for domain in get_args(GraphDomain):
            assert(isinstance(self.inputs, tuple))
            domain_inputs = self._get_inputs_by_domain(domain)
            if len(domain_inputs) == 1:
                p_input = domain_inputs[0]
                p_idx = self.inputs.index(p_input)
                self.inputs = (self.inputs[:p_idx] + replace(p_input, primary=True) + self.inputs[p_idx + 1:])

    def __getattr__(self, item: str):
        if item.endswith("_inputs"):
            domain = item.replace("_inputs", "")
            return self._get_inputs_by_domain(domain)
        else:
            return super(GraphNode, self).__getattr__(item)
    
    @property
    @eqx.filter_jit
    def input_node_IDs(self):
        return tuple(set([i.network_ID for i in self.inputs]))
    
    @property
    @eqx.filter_jit
    def input_domains(self):
        return tuple(set([i.domain for i in self.inputs]))
    
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
    def get_domain_primary(self, domain: GraphDomain):
        domain_inputs = self._get_inputs_by_domain(domain)
        domain_primary = next(filter(lambda i: i.primary, domain_inputs))
        return domain_primary
    
    @eqx.filter_jit
    def get_primary_input_state(self, state, domain: GraphDomain, input_field):
        p_input = self.get_domain_primary(domain)
        return self.get_input_state(state, p_input, input_field)
    
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

# Interpolators & Splitters ----------------------------------------------------

@register
class Interpolator(GraphNode):

    values: str | tuple[str] = init_field(tuple, static=True)
    fractions: float | Callable | tuple[float | Callable] = init_field(tuple, static=True)

    def __post_init__(self):
        super().__post_init__()
        assert(isinstance(self.inputs, tuple))
        input_nodes = self.input_node_IDs
        if len(input_nodes) != 1:
            warnings.warn(f"Interpolaters can only have one input node. Found {len(input_nodes)}: {input_nodes}.",
                          RuntimeWarning)
        
        if isinstance(self.values, str):
            object.__setattr__(self, "values", (self.values,))
        if isinstance(self.fractions, float) or isinstance(self.fractions, Callable):
            object.__setattr__(self, "fractions", (self.fractions,))

    def transmit(self, state: State, system: System, settings: Settings):
        

@register
@dataclass(frozen=True)
class GraphSplit:
    input: GraphInput
    values: str | tuple[str]
    fraction: float | tuple[float]

@register
class Splitter(GraphNode):

    values: str | tuple[str] = init_field(tuple, static=True)
    fractions: float | Callable | tuple[float | Callable] = init_field(tuple, static=True)

    def __post_init__(self):
        
        # Set inputs
        object.__setattr__(self, "inputs", (s.input for s in self.splits))
        super(Splitter, self).__post_init__()
        assert(isinstance(self.inputs, tuple))
        input_nodes = self.input_node_IDs
        if len(input_nodes) != 1:
            warnings.warn(f"Splitters can only have one input node. Found {len(input_nodes)}: {input_nodes}.",
                          RuntimeWarning)
        
        if isinstance(self.values, str):
            object.__setattr__(self, "values", (self.values,))
        if isinstance(self.fractions, float) or isinstance(self.fractions, Callable):
            object.__setattr__(self, "fractions", (self.fractions,))

    def transmit(self, state: State, system: System, settings: Settings):

        assert isinstance(self.fractions, tuple)
        assert isinstance(self.values, tuple)
        updated_state = state

        inp = cast(tuple, self.inputs)[0]
        ID = inp.network_ID
        domain = inp.domain
        
        for v_idx, value in enumerate(self.values):

            total_input = getattr(getattr(state.energy.nodes[ID].outputs, domain), value)

            if callable(self.fractions[v_idx]):
                frac = self.fractions[v_idx](state)  # type: ignore
            else:
                frac = self.value_fractions

            split_input = eqx.tree_at(
                lambda t: getattr(t, value),
                total_input,
                getattr(total_input, value) * frac,
            )

            updated_state = eqx.tree_at(
                lambda s: getattr(s.energy.nodes[self.network_ID].outputs, domain), updated_state, split_input
            )

        return updated_state, system, settings
    

@register
class Interpolator(Splitter):

    values: str | tuple[str] = init_field(tuple, static=True)
    fractions: float | Callable | tuple[float | Callable] = init_field(tuple, static=True)

    def transmit(self, state: State, system: System, settings: Settings):




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
class BleedFlow(GraphNode):

    tag: str = init_field("Bleed Flow", static=True)
    fraction_dict: dict[str, float | Callable] = init_field(defaultdict(lambda: 0.0), static=True)

    parent_ID: str = init_field('', static=True)
    grandparent_ID: str = init_field(tuple, static=True)

    def transmit(self, state: State, system: System, settings: Settings):
        
        updated_state = state
        
        for attr in self.fraction_dict:
            if callable(self.fraction_dict[attr]):
                frac = self.fraction_dict[attr](state)  # type: ignore
            else:
                frac = self.fraction_dict[attr]

            in_value = getattr(state.energy.nodes[self.grandparent_ID].outputs.flow, attr)
            out_value = getattr(state.energy.nodes[self.parent_ID].outputs.flow, attr)
            
            if attr == "mass_flow_rate":
                bleed_value = in_value * frac
            else:
                bleed_value = in_value + (out_value - in_value) * frac
            
            updated_state = eqx.tree_at(
                lambda s: getattr(s.energy.nodes[self.network_ID].outputs.flow, attr),
                updated_state,
                bleed_value
            )
        
        return updated_state, system, settings


@register
class FlowNode[DesignType: FlowDesign](GraphNode):
    
    design_parameters: DesignType = init_field(FlowDesign)
    working_fluid: IdealGas = init_field(Air)

    output_bleeds: tuple[BleedFlow,...] = init_field(tuple, static=True)

    def __post_init__(self):
        super(FlowNode, self).__post_init__()

        object.__setattr__(self, "subcomponents", self.subcomponents + self.output_bleeds)
    
    def mix_inputs(self, state: State):
        
        W_fracs = jnp.array([i.get_value("mass_flow_rate") for i in self.flow_inputs])
        T_fracs = jnp.array([i.get_value("stagnation_temperature") for i in self.flow_inputs])
        h_fracs = jnp.array([i.get_value("enthalpy") for i in self.flow_inputs])
        
        W_mix = self.apply_domain_op(jnp.sum, state, "flow", "mass_flow_rate")
        h_mix = jnp.dot(W_fracs, h_fracs) / W_mix

        mixed_fluid = MixedGas(
            tag = f"{self.tag} input flow",
            composition=GasComposition(
                    elements=tuple(i.get_value("fluid") for i in self.flow_inputs),
                    mass_fractions=W_fracs,
                )
            )
        
        T_t_guess = jnp.dot(W_fracs, T_fracs) / W_mix
        T_t = mixed_fluid.invert_enthalpy(h_mix, T_t_guess)
        P_t = self.get_primary_input_state(state, "flow", "stagnation_pressure")

        p_FAR = self.get_primary_input_state(state, "fuel", "fuel_air_ratio")
        p_W = self.get_primary_input_state(state, "flow", "mass_flow_rate")
        FAR = p_FAR * p_W / W_mix

        return mixed_fluid, T_t, P_t, W_mix, FAR
            

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