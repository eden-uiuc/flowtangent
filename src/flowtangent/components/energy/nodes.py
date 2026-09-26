# flowtangent/Library/Components/Energy/Nodes.py
# (c) Copyright 2026 Aerospace Research Community LLC
#
# Created: Jun 2026, J. Smart
# Modified: Jun 2026, J. Smart

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from typing import TYPE_CHECKING, Callable, Iterable, Literal, cast, get_args

# --- Framework Imports (Strictly for Type Hinting to avoid Circular Imports) ---
if TYPE_CHECKING:
    from ... import Settings, State, System
    from ...solve.energy.jets import JetSettings

import warnings
from dataclasses import replace
from functools import reduce

import equinox as eqx
import jax
import jax.numpy as jnp

from ...core._component import Component
from ...data.gases import Air, Gas
from ...utils import Module, NameType, field, static_field, update
from ...utils.typing import ScalarFloat

# ----------------------------------------------------------------------------------------------------------------------
#  Graph Nodes
# ----------------------------------------------------------------------------------------------------------------------

# Inputs & Nodes ---------------------------------------------------------------


class Efficiencies(Module):
    total: float | jax.Array = 1.0

    # fmt: off
    mechanical: ScalarFloat = 1.0
    electrical: ScalarFloat = 1.0
    fuel:       ScalarFloat = 1.0
    flow:       ScalarFloat = 1.0
    force:      ScalarFloat = 1.0
    # fmt: on


GraphDomain = Literal["flow", "mechanical", "electrical", "fuel", "force", "residual"]


class PACTInput(Module):
    domain: GraphDomain = static_field("flow")
    network_id: str = static_field("")
    primary: bool = static_field(False)
    _assigned: bool = static_field(False)
    name: str = static_field("Input")

    # Custom init to reorder arguments
    def __init__(
        self,
        domain: GraphDomain = "flow",
        network_id: str = "",
        primary: bool = False,
        name: NameType = "",
        _assigned: bool = False,
    ) -> None:

        self.domain = domain
        self.network_id = network_id
        self.name = " ".join(self.network_id.split(".")).title() + f" {domain}".title() + " Outputs"
        self.primary = primary
        self._assigned = _assigned

    # Define iter to make castable to tuple as (self,)
    def __iter__(self):
        yield self

    def __repr__(self) -> str:
        if self.primary:
            p_str = "Primary "
        else:
            p_str = ""
        return p_str + f"{self.domain.title()} Input: {self.network_id}"

    def get_value(self, state: State, value: str):
        return reduce(getattr, (state.energy.nodes[self.network_id], self.domain, value))


class PACTNode(Component):
    network_id: str = field("energy_node", static=True)

    inputs: tuple[PACTInput, ...] | PACTInput = field(tuple, static=True)

    def __post_init__(self):
        if isinstance(self.inputs, PACTInput):
            object.__setattr__(self, "inputs", (self.inputs,))

        for domain in get_args(GraphDomain):
            assert isinstance(self.inputs, tuple)
            domain_inputs = self._get_inputs_by_domain(domain)
            if len(domain_inputs) == 1:
                p_input = domain_inputs[0]
                p_idx = self.inputs.index(p_input)
                self.inputs = self.inputs[:p_idx] + (replace(p_input, primary=True),) + self.inputs[p_idx + 1 :]

    def __getattr__(self, item: str):
        if item.endswith("_inputs"):
            domain = item.replace("_inputs", "")
            return self._get_inputs_by_domain(domain)
        else:
            return super(PACTNode, self).__getattr__(item)

    @property
    @eqx.filter_jit
    def input_node_IDs(self):
        return tuple(set([i.network_id for i in self.inputs]))

    @property
    @eqx.filter_jit
    def input_domains(self):
        return tuple(set([i.domain for i in self.inputs]))

    @eqx.filter_jit
    def _get_inputs_by_domain(self, domain: GraphDomain | str):
        return tuple(filter(lambda i: i.domain == domain, cast(tuple, self.inputs)))

    @eqx.filter_jit
    def get_input_state(self, state: State, input: PACTInput, input_field: str):
        return getattr(getattr(state.energy.nodes[input.network_id], input.domain), input_field)

    @eqx.filter_jit
    def get_input_states(self, state: State, inputs: Iterable[PACTInput]):
        return [getattr(state.energy.nodes[i.network_id], i.domain) for i in inputs]

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
    def _get_input_array(self, state: State, inputs: Iterable[PACTInput], input_field: str):
        input_conditions = self.get_input_states(state, inputs)
        input_values = [jnp.asarray(getattr(inp, input_field)) for inp in input_conditions]
        return jnp.concatenate([jnp.atleast_2d(v) for v in input_values if v.size > 0], axis=-1)

    # Input Operations
    @eqx.filter_jit
    def apply_input_op(self, arr_func: Callable, state: State, inputs: Iterable[PACTInput], input_field: str):
        input_arr = self._get_input_array(state, inputs, input_field)
        return jnp.atleast_2d(arr_func(input_arr, axis=-1)).T

    @eqx.filter_jit
    def apply_domain_op(self, arr_func: Callable, state: State, domain: GraphDomain, input_field: str):
        inputs = self._get_inputs_by_domain(domain)
        input_arr = self._get_input_array(state, inputs, input_field)
        return jnp.atleast_2d(arr_func(input_arr, axis=-1)).T

    def transmit(self, state: State, system: System, settings: Settings):
        return state, system, settings


# Splitters --------------------------------------------------------------------


class Splitter(PACTNode):
    values: str | tuple[str] = field(tuple, static=True)
    fractions: float | Callable | tuple[float | Callable] = field(tuple, static=True)

    def __post_init__(self):
        # Set inputs
        super(Splitter, self).__post_init__()
        assert isinstance(self.inputs, tuple)
        input_nodes = self.input_node_IDs
        if len(input_nodes) != 1:
            warnings.warn(
                f"Splitters can only have one input node. Found {len(input_nodes)}: {input_nodes}.", RuntimeWarning
            )

        if isinstance(self.values, str):
            object.__setattr__(self, "values", (self.values,))
        if isinstance(self.fractions, float) or isinstance(self.fractions, Callable):
            object.__setattr__(self, "fractions", (self.fractions,))

    def transmit(self, state: State, system: System, settings: Settings):

        assert isinstance(self.fractions, tuple)
        assert isinstance(self.values, tuple)
        updated_state = state

        inp = cast(tuple, self.inputs)[0]
        ID = inp.network_id
        domain = inp.domain

        for v_idx, value in enumerate(self.values):
            domain_input = getattr(state.energy.nodes[ID], domain)
            total_input = getattr(domain_input, value)

            if callable(self.fractions[v_idx]):
                frac = self.fractions[v_idx](state)  # type: ignore
            else:
                frac = self.value_fractions

            split_input = update(
                domain_input,
                lambda t: getattr(t, value),
                jnp.atleast_2d(total_input * frac),
            )

            updated_state = update(
                updated_state,
                lambda s: getattr(s.energy.nodes[self.network_id], domain),
                split_input,
            )

        return updated_state, system, settings


# ----------------------------------------------------------------------------------------------------------------------
#  Flow Nodes
# ----------------------------------------------------------------------------------------------------------------------
class FlowOpPoint(Module):
    # fmt: off
    pressure_ratio:     ScalarFloat = 1.0
    pressure_recovery:  ScalarFloat = 1.0

    intake_temperature: ScalarFloat = 298.15
    output_temperature: ScalarFloat = 298.15

    A_intake:   ScalarFloat = 1.0
    A_throat:   ScalarFloat = 1.0
    A_exit:     ScalarFloat = 1.0

    exit_mach_number: ScalarFloat = 1e-6

    rotation_speed: ScalarFloat = 0.0
    noise_speed:    ScalarFloat = 0.0

    eff: Efficiencies = field(Efficiencies)
    # fmt: on


class BleedFlow(PACTNode):
    name: str = field("Bleed Flow", static=True)
    fractions_dict: dict[str, float | Callable] = field(dict)

    parent_id: str = field("", static=True)
    grandparent_id: str = field(tuple, static=True)

    def transmit(self, state: State, system: System, settings: Settings):

        updated_state = update(
            state,
            lambda s: s.energy.nodes[self.network_id].flow,
            state.energy.nodes[self.grandparent_id].flow,
        )

        for attr in self.fractions_dict:
            if callable(self.fractions_dict[attr]):
                frac = self.fractions_dict[attr](state)  # type: ignore
            else:
                frac = self.fractions_dict[attr]

            in_value = getattr(state.energy.nodes[self.grandparent_id].flow, attr)
            out_value = getattr(state.energy.nodes[self.parent_id].flow, attr)

            if attr == "mass_flow_rate":
                bleed_value = in_value * frac
            else:
                bleed_value = in_value + (out_value - in_value) * frac

            updated_state = update(
                updated_state,
                lambda s: getattr(s.energy.nodes[self.network_id].flow, attr),
                bleed_value,
            )

            if attr == "stagnation_enthalpy":
                fluid: Gas = state.energy.nodes[self.parent_id].flow.fluid
                T_t = fluid.invert_enthalpy(bleed_value)
                updated_state = update(
                    updated_state,
                    lambda s: s.energy.nodes[self.network_id].flow.stagnation_temperature,
                    T_t,
                )

        return updated_state, system, settings


class FlowNode[DesignType: FlowOpPoint | tuple](PACTNode):
    design_parameters: DesignType = field(FlowOpPoint)
    working_fluid: Gas = field(Air)
    add_mixer: bool = field(False)

    output_bleeds: tuple[BleedFlow, ...] = field(tuple)

    _bookkeeping: dict = field(lambda: {"bleeds": BleedFlow}, static=True)

    def __post_init__(self):
        super(FlowNode, self).__post_init__()

        if len(self.output_bleeds) > 0:
            add_mixer = not hasattr(self, "mixer")
            self_bleeds = tuple(replace(b, inputs=PACTInput("flow", "parent")) for b in self.output_bleeds)
            # BleedFlow Parent ID and Grandparent ID set in PACTNetwork.compute_topology
            object.__setattr__(self, "subcomponents", self.subcomponents + self_bleeds)
            object.__setattr__(self, "output_bleeds", tuple())
        else:
            add_mixer = self.add_mixer and not hasattr(self, "mixer")

        if add_mixer:
            parent_inputs = tuple(replace(i, network_id="parent." + i.network_id) for i in self.flow_inputs)
            mixer = FlowNode(
                name="Mixer",
                inputs=parent_inputs,
                add_mixer=False,
                design_parameters=FlowOpPoint(pressure_ratio=1.0),
            )

            other_inputs = tuple(i for i in self.inputs if i not in self.flow_inputs)
            object.__setattr__(
                self, "inputs", other_inputs + (PACTInput(domain="flow", network_id="self.mixer", primary=True),)
            )
            object.__setattr__(self, "subcomponents", self.subcomponents + (mixer,))

    def mix_inputs(self, state: State) -> tuple[Gas, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:

        M = self.get_primary_input_state(state, "flow", "mach_number")

        if len(self.flow_inputs) == 1:
            # fmt: off
            mixed_fluid = self.get_primary_input_state(state, "flow", "fluid")
            T_t         = self.get_primary_input_state(state, "flow", "stagnation_temperature")
            P_t         = self.get_primary_input_state(state, "flow", "stagnation_pressure")
            W_mix       = self.get_primary_input_state(state, "flow", "mass_flow_rate")
            FAR         = self.get_primary_input_state(state, "flow", "fuel_air_ratio")
            # fmt: on

        else:
            # fmt: off
            # Get incoming flow values
            W_list     = [i.get_value(state, "mass_flow_rate") for i in self.flow_inputs]
            T_t_list   = [i.get_value(state, "stagnation_temperature") for i in self.flow_inputs]
            h_t_list   = [i.get_value(state, "stagnation_enthalpy") for i in self.flow_inputs]
            fluid_list = [i.get_value(state, "fluid") for i in self.flow_inputs]

            # Concatenate for the thermodynamic state mixing
            W_fracs   = jnp.concatenate(W_list, axis=-1)
            T_t_fracs = jnp.concatenate(T_t_list, axis=-1)
            h_t_fracs = jnp.concatenate(h_t_list, axis=-1)
            # fmt: on

            # Calculate mixed baseline mass flow and enthalpy
            W_mix = self.apply_domain_op(jnp.sum, state, "flow", "mass_flow_rate")
            h_t_mix = jnp.sum(W_fracs * h_t_fracs, axis=-1, keepdims=True) / W_mix

            # Mixed Fluid
            mixed_mf = sum(W * f.mass_fractions for W, f in zip(W_list, fluid_list)) / W_mix
            mixed_fluid = Gas(mass_fractions=mixed_mf)

            # Invert temperature from enthalpy
            T_t_guess = jnp.sum(W_fracs * T_t_fracs, axis=-1, keepdims=True) / W_mix
            T_t = mixed_fluid.invert_enthalpy(h_t_mix, T_t_guess)
            P_t = self.get_primary_input_state(state, "flow", "stagnation_pressure")

            # Check for fuel in mixture and dilute
            try:
                p_FAR = self.get_primary_input_state(state, "flow", "fuel_air_ratio")
                p_W = self.get_primary_input_state(state, "flow", "mass_flow_rate")
                FAR = p_FAR * p_W / W_mix
            except Exception:
                FAR = jnp.atleast_2d(0.0)

        return mixed_fluid, T_t, P_t, W_mix, FAR, M

    def bleed_MFR_frac(self, state: State):
        if len(self.bleeds) > 0:
            bleed_fracs = [b.fractions_dict.get("mass_flow_rate", 0.0) for b in self.bleeds]
            actual_fracs = jnp.array([f(state) if callable(f) else f for f in bleed_fracs])
            return jnp.atleast_2d(jnp.sum(actual_fracs))
        else:
            return jnp.atleast_2d(0.0)

    @staticmethod
    def kinematic_design(gas: Gas, T_t_out, P_t_out, M_out, mdot):

        # Unpack boundary stagnation properties
        R = gas.R_specific
        gamma = gas.compute_gamma(T_t_out)

        # Compute exit static properties
        T_out = T_t_out / (1.0 + ((gamma - 1.0) / 2.0) * M_out**2)
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
        gas: Gas,
        T_t: jax.Array | float,
        P_t: jax.Array | float,
        PR: jax.Array | float,
        n_isn: jax.Array | float,
        # Ignored for subsonic flows
        M: jax.Array | float = 0.0,
        P_rec: jax.Array | float = 1.0,
    ):
        g_in = gas.compute_gamma(T_t)
        T_t_out_ideal = T_t * (PR ** ((g_in - 1.0) / g_in))
        P_t_out_ideal = P_t * PR * P_rec

        # Normal Shock Recovery
        safe_M = jnp.maximum(M, 1.0)
        ns_P_t = (
            PR
            * P_t
            * ((((g_in + 1.0) * (safe_M**2.0)) / ((g_in - 1.0) * safe_M**2.0 + 2.0)) ** (g_in / (g_in - 1.0)))
            * ((g_in + 1.0) / (2.0 * g_in * safe_M**2.0 - (g_in - 1.0))) ** (1.0 / (g_in - 1.0))
        )

        P_t_out = jnp.where(M > 1.0, ns_P_t, P_t_out_ideal)
        PR_actual = P_t_out / P_t

        # Newton-Raphson step to average T_t_out over gamma change
        def step(T_t_out_ideal, _):
            g_out = gas.compute_gamma(T_t_out_ideal)
            g_avg = 0.5 * (g_in + g_out)
            T_t_out_ideal = T_t * (PR_actual ** ((g_avg - 1.0) / g_avg))
            # new_T_t = jnp.reshape(new_T_t, T_t_out_ideal.shape)
            return T_t_out_ideal, None

        T_t_out_ideal, _ = jax.lax.scan(step, T_t_out_ideal, jnp.arange(5))

        # Compressor passes 1 / n_isn, so T_t_out is higher, Turbine passes n_isn, so T_t_out is lower
        T_t_out = T_t + (T_t_out_ideal - T_t) * n_isn

        return T_t_out, P_t_out

    @staticmethod
    def statics(
        gas: Gas,
        T_t: float | jax.Array,
        P_t: float | jax.Array,
        mdot: float | jax.Array,
        area: float | jax.Array,
    ):
        # fmt: off
        gamma   = jnp.atleast_2d(gas.compute_gamma(T_t))
        R       = jnp.atleast_2d(gas.R_specific)
        # fmt: on

        # Non-dimensional mass flow
        Q = (mdot * jnp.sqrt(R * T_t)) / (P_t * area * jnp.sqrt(gamma))

        # Newton loop to find subsonic Mach number
        def step(M, _):
            term = 1.0 + (gamma - 1.0) / 2.0 * M**2
            power = -(gamma + 1.0) / (2.0 * (gamma - 1.0))

            f = M * (term**power) - Q

            # Derivative df/dM
            df_dM = (term**power) + M * power * (term ** (power - 1.0)) * (gamma - 1.0) * M

            M = jnp.clip(M - f / df_dM, 1e-6, 0.99)

            return M, None

        M, _ = jax.lax.scan(step, 0.5 * jnp.ones_like(gamma), jnp.arange(5))

        T = jnp.atleast_2d(T_t / (1.0 + (gamma - 1.0) / 2.0 * M**2))
        P = jnp.atleast_2d(P_t / (1.0 + (gamma - 1.0) / 2.0 * M**2) ** (gamma / (gamma - 1.0)))

        # fmt: off
        h_t = jnp.atleast_2d(gas.compute_enthalpy(T_t))
        h   = jnp.atleast_2d(gas.compute_enthalpy(T))
        u   = jnp.atleast_2d(jnp.sqrt(2.0 * (h_t - h)))
        # fmt: on

        return T, P, h_t, h, u, M

    def transmit(self, state: State, system: System, settings: Settings):
        """
        Duct-like transmission when not overridden by child class
        """

        updated_state = state
        updated_system = system

        analysis_settings: JetSettings = settings.analysis.energy
        design_mode = analysis_settings.design_mode
        statics = analysis_settings.statics

        gas, T_t, P_t, W_in, FAR, M = self.mix_inputs(state)
        W_out = W_in * (1.0 - self.bleed_MFR_frac(state))

        # fmt: off
        PR    = jnp.atleast_2d(system.energy.nodes[self.network_id].design_parameters.pressure_ratio)
        P_rec = jnp.atleast_2d(system.energy.nodes[self.network_id].design_parameters.pressure_recovery)
        n_isn = jnp.atleast_2d(system.energy.nodes[self.network_id].design_parameters.eff.flow)
        # fmt: on

        if not statics:
            M = jnp.atleast_2d(0.0)

        T_t_out, P_t_out = self.stagnation(gas, T_t, P_t, PR, n_isn, M, P_rec)
        h_t_out = gas.compute_enthalpy(T_t_out)

        if design_mode:
            if statics:
                M_out = jnp.atleast_2d(system.energy.nodes[self.network_id].design_parameters.exit_mach_number)

                A_out, u_out, P_out, T_out, h_t_out, h_out = self.kinematic_design(
                    gas=gas,
                    T_t_out=T_t_out,
                    P_t_out=P_t_out,
                    M_out=M_out,
                    mdot=W_out,
                )

                updated_design_parameters = update(self.design_parameters, ("A_exit", A_out.squeeze()))

                updated_system = update(
                    updated_system,
                    lambda s: s.energy.nodes[self.network_id].design_parameters,
                    updated_design_parameters,
                )

        else:
            if statics:
                A_out = jnp.atleast_2d(self.design_parameters.A_exit)
                T_out, P_out, h_t_out, h_out, u_out, M_out = self.statics(gas, T_t_out, P_t_out, W_out, A_out)

        outputs = state.energy.nodes[self.network_id].flow

        outputs = update(outputs, "mass_flow_rate", jnp.atleast_2d(W_out))
        outputs = update(outputs, "stagnation_pressure", jnp.atleast_2d(P_t_out))
        outputs = update(outputs, "stagnation_temperature", jnp.atleast_2d(T_t_out))
        outputs = update(outputs, "stagnation_enthalpy", jnp.atleast_2d(h_t_out))
        outputs = update(outputs, "fuel_air_ratio", jnp.atleast_2d(FAR))

        if statics:
            outputs = update(outputs, "temperature", jnp.atleast_2d(T_out))
            outputs = update(outputs, "pressure", jnp.atleast_2d(P_out))
            outputs = update(outputs, "speed", jnp.atleast_2d(u_out))
            outputs = update(outputs, "mach_number", jnp.atleast_2d(M_out))
            outputs = update(outputs, "enthalpy", jnp.atleast_2d(h_out))
            outputs = update(outputs, "area", jnp.atleast_2d(A_out))

        updated_state = update(
            updated_state,
            lambda s: s.energy.nodes[self.network_id].flow,
            outputs,
        )

        return updated_state, updated_system, settings


# ----------------------------------------------------------------------------------------------------------------------
# Energy Store
# ----------------------------------------------------------------------------------------------------------------------


class EnergyStore(PACTNode):
    name: str = field("Energy Store", static=True)

    max_energy: float = 0.0

    specific_energy: float = 0.0
    specific_volume: float = 0.0


# ----------------------------------------------------------------------------------------------------------------------
# Fuel Tank
# ----------------------------------------------------------------------------------------------------------------------


class FuelTank(EnergyStore):
    name: str = field("Fuel Tank", static=True)

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


class RagoneParameters(Module):
    const_1: float = 0.0
    const_2: float = 0.0
    lower_bound: float = 0.0
    i: float = 0.0


class Battery(EnergyStore):
    name: str = field("Battery", static=True)

    max_energy: float = 0.0
    max_power: float = 0.0
    max_voltage: float = 0.0

    resistance: float = 0.0

    ragone: RagoneParameters = field(RagoneParameters)


if __name__ == "__main__":
    doms = get_args(GraphDomain)
    for d in doms:
        print(d)
