# Trace/Library/Components/Energy/Nodes.py
# (c) Copyright 2026 Aerospace Research Community LLC
#
# Created: Jun 2026, J. Smart
# Modified: Jun 2026, J. Smart

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Callable, Iterable, get_args, cast
# --- Framework Imports (Strictly for Type Hinting to avoid Circular Imports) ---
if TYPE_CHECKING:
    from eden_trace.framework import State, System, Settings

import warnings
import equinox as eqx
import jax
import jax.numpy as jnp

from dataclasses import replace
from functools import reduce

from eden_trace.utils import init_field, register

from eden_trace.library import Component
from eden_trace.library.gases import Air, IdealGas, MixedGas, MixedGasTemplate, flatten_elements

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
    _assigned: bool = init_field(False)

    # Define iter to make castable to tuple as (self,)
    def __iter__(self):
        yield self

    def __repr__(self) -> str:
        if self.primary:
            p_str = "Primary "
        else:
            p_str = ""
        return p_str + f"{self.domain.title()} Input: {self.network_ID}"

    def get_value(self, state:State, value: str):
        return reduce(getattr, (state.energy.nodes[self.network_ID].outputs, self.domain, value))

@register
class GraphNode(Component):
    network_ID: str = init_field("energy_node", static=True)

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
                self.inputs = (self.inputs[:p_idx] + (replace(p_input, primary=True),) + self.inputs[p_idx + 1:])

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
        return state, system, settings

# Splitters --------------------------------------------------------------------

@register
class Splitter(GraphNode):

    values: str | tuple[str] = init_field(tuple, static=True)
    fractions: float | Callable | tuple[float | Callable] = init_field(tuple, static=True)

    def __post_init__(self):
        
        # Set inputs
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

            domain_input = getattr(state.energy.nodes[ID].outputs, domain)
            total_input = getattr(domain_input, value)

            if callable(self.fractions[v_idx]):
                frac = self.fractions[v_idx](state)  # type: ignore
            else:
                frac = self.value_fractions

            split_input = eqx.tree_at(
                lambda t: getattr(t, value),
                domain_input,
                jnp.atleast_2d(total_input * frac),
            )

            updated_state = eqx.tree_at(
                lambda s: getattr(s.energy.nodes[self.network_ID].outputs, domain), updated_state, split_input
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

    exit_mach_number: float = 1e-6
    
    rotation_speed: float = 0.0
    noise_speed: float = 0.0

    eff: Efficiencies = init_field(Efficiencies, static=True)

@register
class BleedFlow(GraphNode):

    tag: str = init_field("Bleed Flow", static=True)
    fractions_dict: dict[str, float | Callable] = init_field(dict)

    parent_ID: str = init_field('', static=True)
    grandparent_ID: str = init_field(tuple, static=True)

    def transmit(self, state: State, system: System, settings: Settings):
        
        updated_state = eqx.tree_at(
            lambda s: s.energy.nodes[self.network_ID].outputs.flow,
            state,
            state.energy.nodes[self.grandparent_ID].outputs.flow,
        )
        
        for attr in self.fractions_dict:
            if callable(self.fractions_dict[attr]):
                frac = self.fractions_dict[attr](state)  # type: ignore
            else:
                frac = self.fractions_dict[attr]

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

            if attr == "stagnation_enthalpy":
                fluid: IdealGas = state.energy.nodes[self.parent_ID].outputs.flow.fluid
                T_t = fluid.invert_enthalpy(bleed_value)
                updated_state = eqx.tree_at(
                    lambda s: s.energy.nodes[self.network_ID].outputs.flow.stagnation_temperature,
                    updated_state,
                    T_t
                )   
        
        return updated_state, system, settings


@register
class FlowNode[DesignType: FlowDesign](GraphNode):
    
    design_parameters: DesignType = init_field(FlowDesign)
    working_fluid: IdealGas = init_field(Air)
    add_mixer: bool = init_field(False)

    output_bleeds: tuple[BleedFlow,...] = init_field(tuple)

    def __post_init__(self):
        super(FlowNode, self).__post_init__()

        if len(self.output_bleeds) > 0:
            add_mixer = not hasattr(self, "mixer")
            self_bleeds = tuple(replace(b, inputs=GraphInput("flow", "parent")) for b in self.output_bleeds)
            object.__setattr__(self, "subcomponents", self.subcomponents + self_bleeds)
            object.__setattr__(self, "output_bleeds", tuple())
        else:
            add_mixer = self.add_mixer and not hasattr(self, "mixer")

        if add_mixer:
            parent_inputs = tuple(replace(i, network_ID="parent."+i.network_ID) for i in self.flow_inputs)
            mixer = FlowNode(tag=f"Mixer", inputs=parent_inputs, add_mixer=False)
            
            other_inputs = tuple(i for i in self.inputs if i not in self.flow_inputs)
            object.__setattr__(self, "inputs", other_inputs + (GraphInput(domain="flow", network_ID="self.mixer", primary=True),))
            object.__setattr__(self, "subcomponents", self.subcomponents + (mixer,))
    
    def mix_inputs(self, state: State) -> tuple[MixedGas, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        
        # Get incoming flow values
        W_fracs = jnp.concatenate([i.get_value(state, "mass_flow_rate") for i in self.flow_inputs], axis=-1)
        T_t_fracs = jnp.concatenate([i.get_value(state, "stagnation_temperature") for i in self.flow_inputs], axis=-1)
        h_t_fracs = jnp.concatenate([i.get_value(state, "stagnation_enthalpy") for i in self.flow_inputs], axis=-1)
        
        # Calculate mixed baseline
        W_mix = self.apply_domain_op(jnp.sum, state, "flow", "mass_flow_rate")
        h_t_mix = jnp.dot(W_fracs, h_t_fracs.T) / W_mix

        # Mix flows into new fluid using cached MixedGasTemplate to avoid recompilation
        elements, fractions = flatten_elements(tuple(i.get_value(state, "fluid") for i in self.flow_inputs), W_fracs/W_mix)
        template_fluid = MixedGasTemplate(tag=f"{self.tag} Input Fluid", elements=elements)
        
        mixed_fluid = eqx.tree_at(
            lambda t: t.composition.mass_fractions,
            template_fluid,
            fractions)
        
        # Invert temperature from enthalpy
        T_t_guess = jnp.dot(W_fracs, T_t_fracs.T) / W_mix
        T_t = mixed_fluid.invert_enthalpy(h_t_mix, T_t_guess)
        P_t = self.get_primary_input_state(state, "flow", "stagnation_pressure")

        # Check for fuel in mixture and dilute
        try:
            p_FAR = self.get_primary_input_state(state, "flow", "fuel_air_ratio")
            p_W = self.get_primary_input_state(state, "flow", "mass_flow_rate")
            FAR = p_FAR * p_W / W_mix
        except:
            FAR = jnp.atleast_2d(0.0)
        
        M = self.get_primary_input_state(state, "flow", "mach_number")

        return mixed_fluid, T_t, P_t, W_mix, FAR, M
    
    def bleed_MFR_frac(self, state:State):
        if len(self.output_bleeds) > 0:
            bleed_fracs = [b.fractions_dict.get("mass_flow_rate", 0.0) for b in self.output_bleeds]
            actual_fracs = jnp.array([f(state) if callable(f) else f for f in bleed_fracs])
            return jnp.atleast_2d(jnp.sum(actual_fracs))
        else:
            return jnp.atleast_2d(0.0)
    
    @staticmethod
    def kinematics(gas: IdealGas, T_t_out, P_t_out, M_out, mdot):

        # Unpack boundary stagnation properties
        R       = gas.R_specific
        gamma   = gas.compute_gamma(T_t_out)
        
        # Compute exit static properties
        T_out = T_t_out / (1.0 + ((gamma - 1.0) / 2.0) * M_out ** 2)
        P_out = P_t_out * (T_out / T_t_out) ** (gamma / (gamma - 1.0))

        # Compute exit kinematic properties
        h_out = gas.compute_enthalpy(T_out)
        h_t_out = gas.compute_enthalpy(T_t_out)
        u_out = jnp.sqrt(jnp.maximum(2.0 * (h_t_out - h_out), 1e-10))

        rho_out = P_out / (R * T_out)
        
        A_out = mdot / (rho_out * u_out)

        return A_out, u_out, P_out, T_out, h_t_out, h_out
    
    @staticmethod
    def stagnation(
        gas: IdealGas,
        T_t: jnp.ndarray | float,
        P_t: jnp.ndarray | float,
        PR: jnp.ndarray | float,
        n_isn: jnp.ndarray | float,
        # Ignored for subsonic flows
        M: jnp.ndarray | float = 0.0,
        P_rec: jnp.ndarray | float = 1.0
    ):
        gamma_in = gas.compute_gamma(T_t)
        gamma_avg = gamma_in
        T_t_out_ideal = T_t * (PR ** ((gamma_in - 1.0) / gamma_in))
        P_t_out_ideal = P_t * PR * P_rec

        # Normal Shock Recovery
        safe_M = jnp.maximum(M, 1.0)
        ns_P_t = (
            PR * P_t
            * ((((gamma_in + 1.0) * (safe_M**2.0)) / ((gamma_in - 1.0) * safe_M**2.0 + 2.0)) ** (gamma_in / (gamma_in - 1.0)))
            * ((gamma_in + 1.0) / (2.0 * gamma_in * safe_M**2.0 - (gamma_in - 1.0))) ** (1.0 / (gamma_in - 1.0))
        )

        P_t_out = jnp.where(M > 1.0, ns_P_t, P_t_out_ideal)
        PR_actual = P_t_out / P_t

        def step(T_t_out_ideal, _):
            gamma_out = gas.compute_gamma(T_t_out_ideal)
            gamma_avg = 0.5 * (gamma_in + gamma_out)
            T_t_out_ideal = T_t * (PR_actual ** ((gamma_avg - 1.0) / gamma_avg))
            return T_t_out_ideal, None
        
        T_t_out_ideal, _ = jax.lax.scan(step, T_t_out_ideal, jnp.arange(5))
        
        # Compressor passes 1 / n_isn, so T_t_out is higher, Turbine passes n_isn, so T_t_out is lower
        T_t_out = T_t + (T_t_out_ideal - T_t) * n_isn

        return T_t_out, P_t_out
    
    @staticmethod
    def statics(
        gas: IdealGas,
        T_t: float | jnp.ndarray,
        P_t: float | jnp.ndarray,
        mdot: float | jnp.ndarray,
        area: float | jnp.ndarray
    ):
        gamma   = jnp.atleast_2d(gas.compute_gamma(T_t))
        R       = jnp.atleast_2d(gas.R_specific)

        # Non-dimensional mass flow
        Q = (mdot * jnp.sqrt(R * T_t)) / (P_t * area * jnp.sqrt(gamma))

        # Newton loop to find subsonic Mach number
        def step(M, _):
            term = 1.0 + (gamma - 1.0) / 2.0 * M**2
            power = - (gamma + 1.0) / (2.0 * (gamma - 1.0))
            
            f = M * (term ** power) - Q
            
            # Derivative df/dM
            df_dM = (term ** power) + M * power * (term ** (power - 1.0)) * (gamma - 1.0) * M
            
            M = jnp.clip(M - f / df_dM, 1e-6, 0.99)

            return M, None
        
        M, _ = jax.lax.scan(step, 0.5 * jnp.ones_like(gamma), jnp.arange(5))

        T = jnp.atleast_2d(T_t / (1.0 + (gamma - 1.0) / 2.0 * M**2))
        P = jnp.atleast_2d(P_t / (1.0 + (gamma - 1.0) / 2.0 * M**2) ** (gamma / (gamma - 1.0)))
        
        h_t = jnp.atleast_2d(gas.compute_enthalpy(T_t))
        h   = jnp.atleast_2d(gas.compute_enthalpy(T))
        u   = jnp.atleast_2d(jnp.sqrt(2.0 * (h_t - h)))

        return T, P, h_t, h, u, M
    
    def transmit(self, state: State, system: System, settings: Settings):
        """
        Duct-like transmission when not overridden by child class
        """

        updated_state = state
        updated_system = system

        gas, T_t, P_t, W_in, FAR, M = self.mix_inputs(state)
        W_out = W_in * (1.0 - self.bleed_MFR_frac(state))
        
        PR    = jnp.atleast_2d(self.design_parameters.pressure_ratio)
        P_rec = jnp.atleast_2d(self.design_parameters.pressure_recovery)
        n_isn = jnp.atleast_2d(self.design_parameters.eff.flow)

        T_t_out, P_t_out = self.stagnation(gas, T_t, P_t, PR, n_isn, M, P_rec)

        if settings.analysis.energy.design_mode:

            M_out = jnp.atleast_2d(self.design_parameters.exit_mach_number)

            A_out, u_out, P_out, T_out, h_t_out, h_out = self.kinematics(
                gas=gas,
                T_t_out=T_t_out,
                P_t_out=P_t_out,
                M_out=M_out,
                mdot=W_out
            )

            updated_design_parameters = eqx.tree_at(
                lambda d: d.A_exit,
                self.design_parameters,
                A_out.squeeze()
            )

            updated_system = eqx.tree_at(
                lambda s: s.energy.nodes[self.network_ID].design_parameters,
                updated_system,
                updated_design_parameters
            )
        
        else:
            A_out = jnp.atleast_2d(self.design_parameters.A_exit)
            T_out, P_out, h_t_out, h_out, u_out, M_out = self.statics(gas, T_t_out, P_t_out, W_out, A_out)

        outputs = state.energy.nodes[self.network_ID].outputs.flow

        outputs = eqx.tree_at(lambda o: o.mass_flow_rate, outputs,          jnp.atleast_2d(W_out))
        outputs = eqx.tree_at(lambda o: o.mach_number, outputs,             jnp.atleast_2d(M_out))
        outputs = eqx.tree_at(lambda o: o.speed, outputs,                   jnp.atleast_2d(u_out))
        outputs = eqx.tree_at(lambda o: o.stagnation_pressure, outputs,     jnp.atleast_2d(P_t_out))
        outputs = eqx.tree_at(lambda o: o.stagnation_temperature, outputs,  jnp.atleast_2d(T_t_out))
        outputs = eqx.tree_at(lambda o: o.temperature, outputs,             jnp.atleast_2d(T_out))
        outputs = eqx.tree_at(lambda o: o.pressure, outputs,                jnp.atleast_2d(P_out))
        outputs = eqx.tree_at(lambda o: o.stagnation_enthalpy, outputs,     jnp.atleast_2d(h_t_out))
        outputs = eqx.tree_at(lambda o: o.enthalpy, outputs,                jnp.atleast_2d(h_out))
        outputs = eqx.tree_at(lambda o: o.area, outputs,                    jnp.atleast_2d(A_out))
        outputs = eqx.tree_at(lambda o: o.fuel_air_ratio, outputs,          jnp.atleast_2d(FAR))


        updated_state = eqx.tree_at(lambda s:
            s.energy.nodes[self.network_ID].outputs.flow,
            updated_state,
            outputs,
        )

        return updated_state, updated_system, settings
        
            

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
class RagoneParameters(eqx.Module):
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

    ragone: RagoneParameters = init_field(RagoneParameters)

if __name__ == "__main__":

    doms = get_args(GraphDomain)
    for d in doms:
        print(d)