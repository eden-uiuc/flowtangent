# flowtangent/Library/Components/Energy/Propulsors.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: May 2025, Flowtangent Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .... import Aircraft, Settings, State, System
    from ....solve.energy.jets import JetSettings


import json
import warnings
from dataclasses import replace
from pathlib import Path

# package imports
import jax
import jax.numpy as jnp

from flowtangent.data import units

from ....data.gases import Air, BurnedJetA, Gas
from ....data.propellants import JetA, Propellant

# Flowtangent imports
from ....utils import Module, field, io, method_field, static_field, update
from ....utils.typing import NameType, ScalarFloat
from ..lines import PACTLine
from ..maps import _data as map_data
from ..maps._classes import CompressorMap, TurbineMap
from ..networks import NetworkParameters, PACTNetwork
from ..nodes import BleedFlow, FlowNode, FlowOpPoint, FuelTank, PACTInput, PACTNode, Splitter

# ----------------------------------------------------------------------------------------------------------------------
# Turbojet Components
# ----------------------------------------------------------------------------------------------------------------------

# Inlet ------------------------------------------------------------------------


class Inlet(FlowNode):
    @io.inputs(
        "state.freestream",
        "state.energy.mass_flow_rate",
        "system.energy.nodes['{network_id}'].design_parameters.pressure_ratio",
        "system.energy.nodes['{network_id}'].design_parameters.pressure_recovery",
        "system.energy.nodes['{network_id}'].design_parameters.eff.flow",
        "system.energy.nodes['{network_id}'].design_parameters.exit_mach_number: Optional",
    )
    @io.outputs(
        "system.energy.nodes['{network_id}'].design_parameters.A_exit: Optional",
        "state.energy.nodes['{network_id}'].flow",
    )
    def transmit(self, state: State, system: Aircraft, settings: Settings):  # type: ignore

        network_state = state.energy
        state_node = network_state.nodes[self.network_id]
        system_node = system.energy.nodes[self.network_id]
        des_params = system_node.design_parameters

        updated_system = system
        fs = state.freestream

        analysis_settings: JetSettings = settings.analysis.energy
        design_mode = analysis_settings.design_mode
        statics = analysis_settings.statics

        gas = fs.atmosphere.fluid
        T_t = fs.stagnation_temperature
        P_t = fs.stagnation_pressure
        M0 = fs.mach_number

        PR = jnp.atleast_2d(des_params.pressure_ratio)
        P_rec = jnp.atleast_2d(des_params.pressure_recovery)
        M_out = jnp.atleast_2d(des_params.exit_mach_number)

        T_t_out, P_t_out = self.stagnation(gas, T_t, P_t, PR, 1.0, M0, P_rec)
        h_t_out = gas.compute_enthalpy(T_t_out)

        if design_mode:
            if statics:
                A_out, u_out, P_out, T_out, h_t_out, h_out = self.kinematic_design(
                    gas=gas,
                    T_t_out=T_t_out,
                    P_t_out=P_t_out,
                    M_out=M_out,
                    mdot=network_state.mass_flow_rate,
                )

                updated_system = update(
                    updated_system,
                    lambda s: s.energy.nodes[self.network_id].design_parameters.A_exit,
                    A_out.squeeze(),
                )

        elif statics:
            T_out, P_out, h_t_out, h_out, u_out, M_out = self.statics(
                gas=fs.atmosphere.fluid,
                T_t=fs.stagnation_temperature,
                P_t=fs.stagnation_pressure,
                mdot=jnp.atleast_2d(network_state.mass_flow_rate),
                area=des_params.A_exit,
            )

        outputs = state_node.flow

        outputs = update(outputs, "mass_flow_rate", jnp.atleast_2d(network_state.mass_flow_rate))
        outputs = update(outputs, "stagnation_pressure", jnp.atleast_2d(P_t_out))
        outputs = update(outputs, "stagnation_temperature", jnp.atleast_2d(T_t_out))
        outputs = update(outputs, "stagnation_enthalpy", jnp.atleast_2d(h_t_out))

        if statics:
            outputs = update(outputs, "mach_number", jnp.atleast_2d(M_out))
            outputs = update(outputs, "temperature", jnp.atleast_2d(T_out))
            outputs = update(outputs, "pressure", jnp.atleast_2d(P_out))
            outputs = update(outputs, "stagnation_enthalpy", jnp.atleast_2d(h_t_out))
            outputs = update(outputs, "enthalpy", jnp.atleast_2d(h_out))
            outputs = update(outputs, "speed", jnp.atleast_2d(u_out))

        updated_state = update(state, lambda s: s.energy.nodes[self.network_id].flow, outputs)

        return updated_state, updated_system, settings


# Compressor -------------------------------------------------------------------


def _alpha_c(Nc, Nc_design):
    """
    Schedules alpha (inlet guide vane angle in degrees) according to rotation speed.
    """
    return jnp.where(Nc_design > 0.0, jnp.maximum(0.0, 90.0 - (Nc / Nc_design) * 90.0), jnp.zeros_like(Nc))


class Compressor(FlowNode):
    inputs: tuple | PACTInput = field(PACTInput("flow", "inlet"), static=True)

    map: CompressorMap = field(map_data.AXI5)

    alpha_schedule: Callable = method_field(_alpha_c)

    def __post_init__(self):
        if not isinstance(self.map, CompressorMap):
            raise TypeError(f"'{self.name}' requires a CompressorMap, got {type(self.map).__name__}")
        if self.design_parameters.eff.flow == 1.0:
            map_effs = replace(self.design_parameters.eff, flow=self.map.eff_des + 0.0)  # +0.0 to force new memalloc
            map_params = replace(self.design_parameters, eff=map_effs)
            object.__setattr__(self, "design_parameters", map_params)
        super(Compressor, self).__post_init__()

    @io.inputs(
        "state.energy.rotation_speed",
        "state.energy.{name.lower()}_Rline",
        "state.energy.nodes['{flow_inputs.network_id}'].flow",
        "system.energy.nodes['{network_id}'].design_parameters.pressure_ratio",
        "system.energy.nodes['{network_id}'].design_parameters.eff.flow",
        "system.energy.nodes['{network_id}'].design_parameters.rotation_speed",
        "system.energy.nodes['{network_id}'].design_parameters.exit_mach_number: Optional",
    )
    @io.outputs(
        "state.energy.residual.{name.lower()}_Wc",
        "state.energy.nodes['{network_id}'].flow",
        "state.energy.nodes['{network_id}'].mechanical.power",
        "system.energy.nodes['{network_id}'].design_parameters.A_exit: Optional",
        "system.energy.nodes['{network_id}'].map.s_Wc",
        "system.energy.nodes['{network_id}'].map.s_PR",
        "system.energy.nodes['{network_id}'].map.s_eff",
        "system.energy.nodes['{network_id}'].map.s_Nc",
    )
    def transmit(self, state: State, system: System, settings: Settings):

        network_state = state.energy
        state_node = network_state.nodes[self.network_id]
        system_node = system.energy.nodes[self.network_id]
        des_params = system_node.design_parameters

        updated_system = system

        analysis_settings = settings.analysis.energy
        design_mode = analysis_settings.design_mode
        statics = analysis_settings.statics

        gas, T_t, P_t, W_in, _, _ = self.mix_inputs(state)
        W_out = W_in * (1.0 - self.bleed_MFR_frac(state))

        theta_c = T_t / 288.15
        delta_c = P_t / 101325.0

        if design_mode:
            # Design Parameters
            M_out = jnp.atleast_2d(des_params.exit_mach_number)
            PR = jnp.atleast_2d(des_params.pressure_ratio)
            n_isn = jnp.atleast_2d(des_params.eff.flow)
            N_des = jnp.atleast_2d(des_params.rotation_speed)

            # Corrected Inflow
            Nc_des = N_des / jnp.sqrt(theta_c)
            Wc_tgt = W_in * jnp.sqrt(theta_c) / delta_c

            # Map Parameters
            PR_map = system_node.map.PR_des
            Wc_map = system_node.map.Wc_des
            eff_map = system_node.map.eff_des
            Nc_map = system_node.map.Nc_des

            s_Wc = (Wc_tgt / Wc_map).squeeze()
            s_PR = (PR - 1.0) / (PR_map - 1.0)
            s_eff = n_isn / eff_map
            s_Nc = (Nc_des / Nc_map).squeeze()

            T_t_out, P_t_out = self.stagnation(gas, T_t, P_t, PR, 1.0 / n_isn)

            if statics:
                A_out, u_out, P_out, T_out, h_t_out, h_out = self.kinematic_design(
                    gas=gas,
                    T_t_out=T_t_out,
                    P_t_out=P_t_out,
                    M_out=M_out,
                    mdot=W_in,
                )

                updated_design_paramters = update(system_node.design_parameters, "A_exit", A_out.squeeze())
            else:
                h_t_out = gas.compute_enthalpy(T_t_out)
                updated_design_paramters = system_node.design_parameters

            updated_map = update(
                system_node.map,
                (
                    ("s_Wc", s_Wc),
                    ("s_PR", s_PR),
                    ("s_eff", s_eff),
                    ("s_Nc", s_Nc),
                ),
            )

            updated_system = update(
                updated_system,
                lambda s: (
                    s.energy.nodes[self.network_id].design_parameters,
                    s.energy.nodes[self.network_id].map,
                ),
                (updated_design_paramters, updated_map),
            )

        else:
            if self.name.lower() == "lpc" or self.name.lower() == "fan":
                N = jnp.atleast_2d(network_state.LP_speed)
            elif self.name.lower() == "hpc":
                N = jnp.atleast_2d(network_state.HP_speed)
            else:
                N = jnp.atleast_2d(network_state.rotation_speed)
            Nc_des = system_node.design_parameters.rotation_speed
            Nc = N / jnp.sqrt(theta_c)

            alpha = self.alpha_schedule(Nc, Nc_des)
            Rline = jnp.atleast_2d(getattr(network_state, f"{self.name.lower()}_Rline"))
            # TODO: Shift to Rline scheduling on altitude, Mach number in future

            # Reference the nodal version of the map to ensure updated scalars
            PR, Wc, n_isn = system_node.map.evaluate(alpha, Nc, Rline)
            W_in = Wc * delta_c / jnp.sqrt(theta_c)

            T_t_out, P_t_out = self.stagnation(gas, T_t, P_t, PR, 1.0 / n_isn)
            h_t_out = gas.compute_enthalpy(T_t_out)

            if statics:
                T_out, P_out, h_t_out, h_out, u_out, M_out = self.statics(
                    gas,
                    T_t_out,
                    P_t_out,
                    W_in,
                    des_params.A_exit,
                )

        power = (h_t_out - jnp.atleast_2d(gas.compute_enthalpy(T_t))) * W_in

        outputs = state_node

        outputs = update(outputs, "mechanical.power", jnp.atleast_2d(power))

        outputs = update(outputs, "flow.mass_flow_rate", jnp.atleast_2d(W_out))
        outputs = update(outputs, "flow.stagnation_pressure", jnp.atleast_2d(P_t_out))
        outputs = update(outputs, "flow.stagnation_temperature", jnp.atleast_2d(T_t_out))
        outputs = update(outputs, "flow.stagnation_enthalpy", jnp.atleast_2d(h_t_out))

        if statics:
            outputs = update(outputs, "flow.temperature", jnp.atleast_2d(T_out))
            outputs = update(outputs, "flow.pressure", jnp.atleast_2d(P_out))
            outputs = update(outputs, "flow.enthalpy", jnp.atleast_2d(h_out))
            outputs = update(outputs, "flow.speed", jnp.atleast_2d(u_out))
            outputs = update(outputs, "flow.mach_number", jnp.atleast_2d(M_out))

        updated_state = update(state, lambda s: s.energy.nodes[self.network_id], outputs)

        # Residual Update
        if isinstance(system.energy.line.engine.design_parameters, tuple):
            eng_des = system.energy.line.engine.design_parameters[0]
        else:
            eng_des = system.energy.line.engine.design_parameters
        W_des = eng_des.mass_flow_rate
        Wc_res = (W_in - state.energy.mass_flow_rate) / W_des

        updated_state = update(
            updated_state,
            lambda s: getattr(s.energy.residual, f"{self.name.lower()}_Wc"),
            Wc_res,
        )

        return updated_state, updated_system, settings


# Burner -----------------------------------------------------------------------


def _burner_design(
    gas: Gas,
    T_t: jax.Array,
    P_t: jax.Array,
    T_t_out: jax.Array,
    mdot_in: jax.Array,
    LHV: jax.Array | float,
    h_t_f: jax.Array | float,
    PR: jax.Array | float,
    n_b: jax.Array | float,
):

    h_t_in = gas.compute_enthalpy(T_t)
    P_t_out = P_t * PR

    # Target exit enthalpy based on the commanded exit temperature
    h_t_out = gas.compute_enthalpy(T_t_out)

    # Simple First-Law FAR calculation using LHV
    numerator = h_t_out - h_t_in
    denominator = (LHV * n_b) + h_t_f - h_t_out

    FAR = numerator / denominator

    # Calculate explicit mass flow additions
    mdot_fuel = mdot_in * FAR
    mdot_out = mdot_in + mdot_fuel

    return P_t_out, h_t_out, jnp.atleast_2d(FAR), mdot_out


def _burner_performance(
    gas: Gas,  # Gas or BurnedGas model
    fuel: Propellant,
    T_t: jax.Array,
    P_t: jax.Array,
    mdot_in: jax.Array,
    FAR: jax.Array,
    PR: jax.Array | float,
    n_b: jax.Array | float,
):
    # 1. Pressure and Mass Flow additions
    P_t_out = P_t * PR
    mdot_fuel = mdot_in * FAR
    mdot_out = mdot_in + mdot_fuel

    # 2. Forward Energy Balance to find Exit Enthalpy
    h_t_in = gas.compute_enthalpy(T_t)

    # Derivation: m_in*h_in + m_fuel*h_fuel + m_fuel*LHV*n_b = m_out*h_out
    LHV = fuel.specific_energy
    h_t_f = fuel.enthalpy_of_formation
    h_t_out = (h_t_in + FAR * (LHV * n_b + h_t_f)) / (1.0 + FAR)

    # 3. Newton-Raphson to invert Enthalpy back to Temperature
    # Initial guess using inlet Cp to get us in the ballpark
    ox = fuel.oxidized_form(FAR)
    Cp_guess = ox.compute_Cp(T_t)
    T_t_out = T_t + (h_t_out - h_t_in) / Cp_guess

    # 5 steps is more than enough for NASA polynomials to converge perfectly
    def step(T_t_out, _):
        h_current = ox.compute_enthalpy(T_t_out)
        Cp_current = ox.compute_Cp(T_t_out)

        error = h_current - h_t_out

        # True Newton Step: x_new = x_old - f(x)/f'(x)
        T_t_out = T_t_out - (error / Cp_current)

        return T_t_out, None

    T_t_out, _ = jax.lax.scan(step, T_t_out, jnp.arange(5))

    return P_t_out, T_t_out, h_t_out, mdot_out


class Burner(FlowNode):
    inputs: tuple | PACTInput = field(PACTInput("flow", "compressor"), static=True)
    fuel: Propellant = field(JetA)

    @io.inputs(
        "state.energy.target_temperature",
        "state.energy.fuel_air_ratio",
        "state.energy.nodes['{flow_inputs.network_id}'].flow",
        "system.energy.nodes['{network_id}'].fuel.specific_energy",
        "system.energy.nodes['{network_id}'].design_parameters.pressure_ratio",
        "system.energy.nodes['{network_id}'].design_parameters.eff.flow",
        "system.energy.nodes['{network_id}'].design_parameters.exit_mach_number: Optional",
    )
    @io.outputs(
        "state.energy.nodes['{network_id}'].flow",
        "system.energy.nodes['{network_id}'].design_parameters.A_exit: Optional",
    )
    def transmit(self, state: State, system: System, settings: Settings):

        network_state = state.energy
        state_node = network_state.nodes[self.network_id]
        system_node = system.energy.nodes[self.network_id]
        des_params = system_node.design_parameters

        updated_system = system

        gas, T_t, P_t, W_in, _, _ = self.mix_inputs(state)

        LHV = system_node.fuel.specific_energy
        PR = des_params.pressure_ratio
        n_b = des_params.eff.flow

        analysis_settings = settings.analysis.energy
        design_mode = analysis_settings.design_mode
        statics = analysis_settings.statics

        if design_mode:
            T_t_out = state.energy.target_temperature

            P_t_out, h_t_out, FAR, mdot_out = _burner_design(
                gas=gas,
                T_t=T_t,
                P_t=P_t,
                T_t_out=T_t_out,
                mdot_in=W_in,
                LHV=LHV,
                h_t_f=0.0,
                PR=PR,
                n_b=n_b,
            )
            if statics:
                A_out, u_out, P_out, T_out, h_t_out, h_out = self.kinematic_design(
                    gas=gas,
                    T_t_out=T_t_out,
                    P_t_out=P_t_out,
                    M_out=des_params.exit_mach_number,
                    mdot=W_in * (1.0 + FAR),
                )

                M_out = des_params.exit_mach_number

                updated_system = update(
                    updated_system,
                    lambda s: s.energy.nodes[self.network_id].design_parameters.A_exit,
                    A_out.squeeze(),
                )

        else:
            FAR = state.energy.fuel_air_ratio

            P_t_out, T_t_out, h_t_out, mdot_out = _burner_performance(
                gas=gas,
                fuel=self.fuel,
                T_t=T_t,
                P_t=P_t,
                mdot_in=W_in,
                FAR=FAR,
                PR=PR,
                n_b=n_b,
            )

            if statics:
                T_out, P_out, h_t_out, h_out, u_out, M_out = self.statics(
                    gas,
                    T_t_out,
                    P_t_out,
                    mdot_out,
                    des_params.A_exit,
                )

        outputs = state_node.flow

        outputs = update(outputs, "stagnation_pressure", jnp.atleast_2d(P_t_out))
        outputs = update(outputs, "stagnation_temperature", jnp.atleast_2d(T_t_out))
        outputs = update(outputs, "stagnation_enthalpy", jnp.atleast_2d(h_t_out))
        outputs = update(outputs, "mass_flow_rate", jnp.atleast_2d(mdot_out))
        outputs = update(outputs, "fuel_air_ratio", jnp.atleast_2d(FAR))
        outputs = update(outputs, "fluid", BurnedJetA(FAR))

        if statics:
            outputs = update(outputs, "temperature", jnp.atleast_2d(T_out))
            outputs = update(outputs, "pressure", jnp.atleast_2d(P_out))
            outputs = update(outputs, "enthalpy", jnp.atleast_2d(h_out))
            outputs = update(outputs, "speed", jnp.atleast_2d(u_out))
            outputs = update(outputs, "mach_number", jnp.atleast_2d(M_out))

        updated_state = update(state, lambda s: s.energy.nodes[self.network_id].flow, outputs)

        return updated_state, updated_system, settings


# Turbine ----------------------------------------------------------------------


class Turbine(FlowNode):
    map: TurbineMap = field(map_data.LPT2269)

    alpha_schedule: Callable = method_field(lambda Np, Np_des: jnp.full_like(Np, 1.0))

    inputs: tuple | PACTInput = static_field(
        (PACTInput("flow", "Burner"),),
    )

    def __post_init__(self):
        if not isinstance(self.map, TurbineMap):
            raise TypeError(f"'{self.name}' requires a TurbineMap, got {type(self.map).__name__}")
        if self.design_parameters.eff.flow == 1.0:
            map_effs = replace(self.design_parameters.eff, flow=self.map.eff_des)
            map_params = replace(self.design_parameters, eff=map_effs)
            object.__setattr__(self, "design_parameters", map_params)
        super(Turbine, self).__post_init__()

    @io.inputs(
        "state.energy.{name.lower()}_PR",
        "state.energy.nodes['{flow_inputs.network_id}'].flow",
        "system.energy.nodes['{network_id}'].map",
        "system.energy.nodes['{network_id}'].design_parameters.eff.flow",
        "system.energy.nodes['{network_id}'].design_parameters.eff.mechanical",
        "system.energy.nodes['{network_id}'].design_parameters.rotation_speed",
        "system.energy.nodes['{network_id}'].design_parameters.exit_mach_number: Optional",
    )
    @io.outputs(
        "state.energy.residual.{name.lower()}_Wp",
        "state.energy.nodes['{network_id}'].flow",
        "state.energy.nodes['{network_id}'].mechanical.power",
        "system.energy.nodes['{network_id}'].map.s_Wp",
        "system.energy.nodes['{network_id}'].map.s_PR",
        "system.energy.nodes['{network_id}'].map.s_eff",
        "system.energy.nodes['{network_id}'].map.s_Np",
        "system.energy.nodes['{network_id}'].design_parameters.pressure_ratio",
        "system.energy.nodes['{network_id}'].design_parameters.A_exit: Optional",
    )
    def transmit(self, state: State, system: System, settings: Settings):

        network_state = state.energy
        state_node = network_state.nodes[self.network_id]
        system_node = system.energy.nodes[self.network_id]
        des_params = system_node.design_parameters

        updated_system = system

        analysis_settings = settings.analysis.energy
        design_mode = analysis_settings.design_mode
        statics = analysis_settings.statics

        gas, T_t, P_t, W, FAR, _ = self.mix_inputs(state)

        if design_mode:
            PR = jnp.atleast_2d(getattr(network_state, f"{self.name.lower()}_PR"))
            n_isn = jnp.atleast_1d(des_params.eff.flow)
            N_des = jnp.atleast_1d(des_params.rotation_speed)

            Np_des = N_des / jnp.sqrt(T_t)

            Wp_tgt = W * jnp.sqrt(T_t) / P_t

            Wp_map = system_node.map.Wp_des
            eff_map = system_node.map.eff_des

            s_Wp = (Wp_tgt / Wp_map).squeeze()
            s_PR = ((PR - 1.0) / (system_node.map.PR_des - 1.0)).squeeze()
            s_eff = des_params.eff.flow / eff_map
            s_Np = (Np_des / system_node.map.Np_des).squeeze()

            # Turbine passes 1 / PR to reflect pressure drop
            safe_PR = jnp.clip(PR, min=1e-5)
            T_t_out, P_t_out = self.stagnation(gas, T_t, P_t, 1.0 / safe_PR, n_isn)

            if statics:
                M_out = jnp.atleast_1d(des_params.exit_mach_number)
                A_out, u_out, P_out, T_out, h_t_out, h_out = self.kinematic_design(
                    gas=gas,
                    T_t_out=T_t_out,
                    P_t_out=P_t_out,
                    M_out=M_out,
                    mdot=W,
                )

            else:
                h_t_out = gas.compute_enthalpy(T_t_out)
                A_out = des_params.A_exit

            updated_map = update(
                system_node.map,
                (
                    ("s_Wp", s_Wp),
                    ("s_PR", s_PR),
                    ("s_eff", s_eff),
                    ("s_Np", s_Np),
                ),
            )

            updated_design = update(
                des_params,
                (
                    ("pressure_ratio", PR),
                    ("A_exit", A_out),
                ),
            )

            updated_system = update(
                updated_system,
                lambda s: (
                    s.energy.nodes[self.network_id].map,
                    s.energy.nodes[self.network_id].design_parameters,
                ),
                (updated_map, updated_design),
            )

        else:
            if self.name.lower() == "lpt":
                N = jnp.atleast_2d(network_state.LP_speed)
            elif self.name.lower() == "hpt":
                N = jnp.atleast_2d(network_state.HP_speed)
            else:
                N = jnp.atleast_2d(network_state.rotation_speed)
            Np = N / jnp.sqrt(T_t)
            Np_des = des_params.rotation_speed

            PR = jnp.atleast_2d(getattr(network_state, f"{self.name.lower()}_PR"))
            # PR = jnp.atleast_2d(system.energy.nodes[self.network_id].design_parameters.pressure_ratio)
            alpha = self.alpha_schedule(Np, Np_des)

            # Reference nodal version of the map to ensure updated parameters
            Wp, n_isn = system_node.map.evaluate(alpha, Np, PR)
            W = Wp * P_t / jnp.sqrt(T_t) * (1.0 + FAR)

            P_t_out = P_t / PR
            T_t_out, P_t_out = self.stagnation(gas, T_t, P_t, 1.0 / PR, n_isn)

            if statics:
                T_out, P_out, h_t_out, h_out, u_out, M_out = self.statics(
                    gas,
                    T_t_out,
                    P_t_out,
                    W,
                    des_params.A_exit,
                )

        # Set Output State
        h_t_in = gas.compute_enthalpy(T_t)
        h_t_out = gas.compute_enthalpy(T_t_out)
        power = (h_t_out - h_t_in) * W * des_params.eff.mechanical

        outputs = state_node

        outputs = update(outputs, "mechanical.power", jnp.atleast_2d(power))

        outputs = update(outputs, "flow.mass_flow_rate", jnp.atleast_2d(W))
        outputs = update(outputs, "flow.fuel_air_ratio", jnp.atleast_2d(FAR))
        outputs = update(outputs, "flow.stagnation_pressure", jnp.atleast_2d(P_t_out))
        outputs = update(outputs, "flow.stagnation_temperature", jnp.atleast_2d(T_t_out))
        outputs = update(outputs, "flow.stagnation_enthalpy", jnp.atleast_2d(h_t_out))

        if statics:
            outputs = update(outputs, "flow.temperature", jnp.atleast_2d(T_out))
            outputs = update(outputs, "flow.pressure", jnp.atleast_2d(P_out))
            outputs = update(outputs, "flow.enthalpy", jnp.atleast_2d(h_out))
            outputs = update(outputs, "flow.speed", jnp.atleast_2d(u_out))
            outputs = update(outputs, "flow.mach_number", jnp.atleast_2d(M_out))

        updated_state = update(state, lambda s: s.energy.nodes[self.network_id], outputs)

        # Residual Update
        if isinstance(system.energy.line.engine.design_parameters, tuple):
            eng_des = system.energy.line.engine.design_parameters[0]
        else:
            eng_des = system.energy.line.engine.design_parameters
        W_des = eng_des.mass_flow_rate
        Wp_res = (W / (1.0 + FAR) - state.energy.mass_flow_rate) / W_des
        updated_state = update(
            updated_state,
            lambda s: getattr(s.energy.residual, f"{self.name.lower()}_Wp"),
            Wp_res,
        )

        return updated_state, updated_system, settings


# Nozzle -----------------------------------------------------------------------

# Helpers ------------------------------------------------------------


def _isentropic_expansion(
    T_t: jax.Array,
    P_t: jax.Array,
    P0: jax.Array,
    gamma: jax.Array,
    PR: jax.Array | float,
):

    # Isentropic Outputs
    P_t_out = jnp.maximum(P_t * PR, P0)  # Output stagnation pressure, minimum is freestream pressure
    T_t_out = T_t  # Output stagnation temperature, adiabatically conserved

    M_out = jnp.sqrt((((P_t_out / P0) ** ((gamma - 1.0) / gamma)) - 1.0) * 2.0 / (gamma - 1.0))  # Output Mach number
    T_out = T_t_out / (1.0 + (gamma - 1.0) / 2.0 * M_out**2)  # Output temperature

    return P_t_out, T_t_out, T_out, M_out


def _mass_flux(
    gas: Gas,
    T_t: jax.Array,
    P_t: jax.Array,
    M: jax.Array,
):
    gamma = gas.compute_gamma(T_t)
    R = gas.R_specific

    M_safe = jnp.maximum(M, 1e-6)
    m_term = jnp.sqrt(gamma * M_safe)

    temp_term = (1.0 + (gamma - 1.0) / 2.0 * M_safe**2) ** (-(gamma + 1.0) / (2.0 * (gamma - 1.0)))

    return (P_t / jnp.sqrt(R * T_t)) * m_term * temp_term


# Sections -----------------------------------------------------------


def _nozzle_design(
    gas: Gas,
    T_t: jax.Array,
    P_t: jax.Array,
    mdot: jax.Array,
    P0: jax.Array,
    PR: jax.Array | float,
    n_v: jax.Array | float,
):
    # Dynamic gas properties for the exhaust flow
    gamma = gas.compute_gamma(T_t)
    R = gas.R_specific

    P_t_out, T_t_out, T_out, M_isn = _isentropic_expansion(T_t, P_t, P0, gamma, PR)

    # Supersonic Expansion / Choking Logic
    critical_PR = (1.0 + (gamma - 1.0) / 2.0) ** (gamma / (gamma - 1.0))
    is_choked = (P_t_out / P0) >= critical_PR

    M_out = jnp.maximum(M_isn, 0.001)

    # Recalculate static conditions
    P_out = P_t_out / (1.0 + (gamma - 1.0) / 2.0 * M_out**2) ** (gamma / (gamma - 1.0))
    P_out = jnp.where(is_choked, P_out, P0)

    T_out = T_t_out / (1.0 + (gamma - 1.0) / 2.0 * M_out**2)

    # Enthalpy and velocity
    h_t_out = gas.compute_enthalpy(T_t_out)
    h_out = gas.compute_enthalpy(T_out)
    u_out = jnp.sqrt(2.0 * (h_t_out - h_out)) * n_v

    # Exit area
    rho_out = P_out / (R * T_out)
    A_exit = mdot / (rho_out * u_out)

    # Throat area
    T_star = T_t / (1.0 + (gamma - 1.0) / 2.0)
    P_star = P_t / critical_PR
    rho_star = P_star / (R * T_star)
    u_star = jnp.sqrt(gamma * R * T_star)

    A_throat_choked = mdot / (rho_star * u_star)
    A_throat = jnp.where(is_choked, A_throat_choked, A_exit)

    return A_throat, A_exit, M_out, rho_out, u_out, P_out, P_t_out, T_out, T_t_out, h_out, h_t_out


def _fixed_nozzle_performance(
    gas: Gas,
    T_t: jax.Array,
    P_t: jax.Array,
    P0: jax.Array,
    diverging_section: bool,
    A_throat: jax.Array | float,
    A_exit: jax.Array | float,
    n_v: jax.Array | float,
):

    gamma = gas.compute_gamma(T_t)
    R = gas.R_specific

    # Check for choked flow
    critical_PR = (1.0 + (gamma - 1.0) / 2.0) ** (gamma / (gamma - 1.0))
    actual_PR = P_t / P0
    choked = actual_PR >= critical_PR

    safe_PR = 1.0 + jnp.sqrt((actual_PR - 1.0) ** 2 + 1e-4)

    if diverging_section:
        # Find exit Mach number
        AR = A_exit / A_throat

        def step(M_exit_sup, _):
            term = (2.0 / (gamma + 1.0)) * (1.0 + (gamma - 1.0) / 2.0 * M_exit_sup**2)
            power = (gamma + 1.0) / (2.0 * (gamma - 1.0))
            AR_calc = (1.0 / M_exit_sup) * (term**power)

            # Analytical derivative: d(A/A*) / dM
            dAR_dM = AR_calc * (M_exit_sup**2 - 1.0) / (M_exit_sup * (1.0 + (gamma - 1.0) / 2.0 * M_exit_sup**2))

            # Newton step
            M_exit_sup = jnp.maximum(M_exit_sup - (AR_calc - AR) / dAR_dM, 1.001)

            return M_exit_sup, None

        M_exit_sup, _ = jax.lax.scan(step, 2.0 * jnp.ones_like(gamma), jnp.arange(5))
        M_exit_sub = jnp.sqrt((2.0 / (gamma - 1.0)) * ((safe_PR) ** ((gamma - 1.0) / gamma) - 1.0))
        M_exit = jnp.where(choked, M_exit_sup, M_exit_sub)
        M_throat = jnp.where(choked, 1.0, M_exit)

    else:
        M_exit = jnp.where(choked, 1.0, jnp.sqrt((2.0 / (gamma - 1.0)) * (safe_PR ** ((gamma - 1.0) / gamma) - 1.0)))
        M_throat = M_exit
        A_throat = A_exit

    # Nozzle Mass Flow
    Q_dot = _mass_flux(gas, T_t, P_t, M_throat)
    mdot_out = Q_dot * A_throat

    P_out_choked = P_t / (1.0 + (gamma - 1.0) / 2.0 * M_exit**2) ** (gamma / (gamma - 1.0))
    P_out = jnp.where(choked, P_out_choked, P0)
    P_t_out = P_out * (1.0 + (gamma - 1.0) / 2.0 * M_exit**2) ** (gamma / (gamma - 1.0))

    T_out = T_t / (1.0 + (gamma - 1.0) / 2.0 * M_exit**2)
    T_t_out = T_t

    h_t_out = gas.compute_enthalpy(T_t_out)
    h_out = gas.compute_enthalpy(T_out)
    u_out = jnp.sqrt(2.0 * (h_t_out - h_out)) * n_v

    rho_out = P_out / (R * T_out)

    return mdot_out, M_exit, u_out, rho_out, P_out, P_t_out, T_out, T_t_out, h_out, h_t_out


def _variable_nozzle_performance(
    gas: Gas,
    T_t: jax.Array,
    P_t: jax.Array,
    P0: jax.Array,
    mdot_in: jax.Array,
    n_v: jax.Array | float,
):

    gamma = gas.compute_gamma(T_t)
    R = gas.R_specific

    # Determine Pressure Ratio and Choking
    critical_PR = (1.0 + (gamma - 1.0) / 2.0) ** (gamma / (gamma - 1.0))
    actual_PR = P_t / P0
    safe_PR = jnp.maximum(actual_PR, 1.00001)  # Prevent div by zero or negative root
    choked = actual_PR >= critical_PR

    # Perfect Expansion Exit Mach
    # Because we vary the nozzle to perfectly expand to ambient pressure (P_out = P0),
    # M_exit is explicitly a function of the total-to-ambient pressure ratio.
    M_out = jnp.sqrt((2.0 / (gamma - 1.0)) * ((safe_PR) ** ((gamma - 1.0) / gamma) - 1.0))

    # 3. Throat Mach
    M_throat = jnp.where(choked, 1.0, M_out)

    # 4. Explicitly Calculate Required Physical Areas
    def calc_area_for_mach(M):
        # Apply a floor to M to prevent NaN gradients in JAX if M approaches 0
        M_safe = jnp.maximum(M, 1e-5)
        m_term = jnp.sqrt(gamma) * M_safe
        temp_term = (1.0 + (gamma - 1.0) / 2.0 * M_safe**2) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))
        return (mdot_in * jnp.sqrt(R * T_t) * temp_term) / (P_t * m_term)

    A_throat = calc_area_for_mach(M_throat)
    A_exit = calc_area_for_mach(M_out)

    # 5. Thermodynamics and Kinematics
    P_out = P0  # Perfect expansion assumption guarantees this
    P_t_out = P_out * (1.0 + (gamma - 1.0) / 2.0 * M_out**2) ** (gamma / (gamma - 1.0))

    T_out = T_t / (1.0 + (gamma - 1.0) / 2.0 * M_out**2)
    T_t_out = T_t

    h_t_out = gas.compute_enthalpy(T_t_out)
    h_out = gas.compute_enthalpy(T_out)
    u_out = jnp.sqrt(jnp.maximum(2.0 * (h_t_out - h_out), 0.0)) * n_v

    rho_out = P_out / (R * T_out)

    return M_out, u_out, rho_out, P_out, P_t_out, T_out, T_t_out, h_out, h_t_out, A_throat, A_exit


class Nozzle(FlowNode):
    name: NameType = field("Core Nozzle", static=True)
    variable_exit: bool = field(False, static=True)
    diverging_section: bool = field(False, static=True)

    inputs: tuple | PACTInput = static_field((PACTInput("flow", "Turbine"),))

    def __post_init__(self):
        super(Nozzle, self).__post_init__()
        if self.variable_exit:
            if not self.diverging_section:
                warnings.warn(
                    f"Variable exit for nozzle '{self.name}' requires diverging section. "
                    "Setting diverging section to True."
                )
                object.__setattr__(self, "diverging_section", True)

    @io.inputs(
        "state.freestream",
        "system.energy.nodes['{network_id}'].design_parameters.pressure_ratio",
        "system.energy.nodes['{network_id}'].design_parameters.eff.flow",
        "system.energy.nodes['{network_id}'].design_parameters.A_throat",
        "system.energy.nodes['{network_id}'].design_parameters.A_exit",
    )
    @io.outputs(
        "state.energy.nodes['{network_id}'].flow",
        "state.energy.residual.area",
    )
    def transmit(self, state: State, system: System, settings: Settings):

        network_state = state.energy
        state_node = network_state.nodes[self.network_id]
        system_node = system.energy.nodes[self.network_id]
        des_params = system_node.design_parameters

        updated_state = state
        updated_system = system

        fs = state.freestream
        P0 = fs.pressure

        analysis_settings = settings.analysis.energy
        design_mode = analysis_settings.design_mode

        gas, T_t, P_t, W_in, _, _ = self.mix_inputs(state)

        if design_mode:
            mdot_out = W_in

            A_t, A_x, M_out, rho_out, u_out, P_out, P_t_out, T_out, T_t_out, h_out, h_t_out = _nozzle_design(
                gas=gas,
                T_t=T_t,
                P_t=P_t,
                mdot=mdot_out,
                P0=P0,
                PR=des_params.pressure_ratio,
                n_v=des_params.eff.flow,
            )

            updated_design_parameters = update(
                des_params,
                (
                    ("A_throat", A_t.squeeze()),
                    ("A_exit", A_x.squeeze()),
                ),
            )

            updated_system = update(
                updated_system,
                lambda s: s.energy.nodes[self.network_id].design_parameters,
                updated_design_parameters,
            )

        else:
            A_x = des_params.A_exit
            A_t = des_params.A_throat

            if self.variable_exit:
                M_out, u_out, rho_out, P_out, P_t_out, T_out, T_t_out, h_out, h_t_out, A_t, A_x = (
                    _variable_nozzle_performance(
                        gas=gas,
                        T_t=T_t,
                        P_t=P_t,
                        P0=P0,
                        mdot_in=W_in,
                        n_v=des_params.eff.flow,
                    )
                )

                mdot_out = W_in

                # Residual update (Turbojet/Single Flow Only)
                A_t_des = des_params.A_throat
                A_t_res = (A_t - A_t_des) / A_t_des
                updated_state = update(updated_state, "energy.residual.area", A_t_res)

            else:
                mdot_out, M_out, u_out, rho_out, P_out, P_t_out, T_out, T_t_out, h_out, h_t_out = (
                    _fixed_nozzle_performance(
                        gas=gas,
                        T_t=T_t,
                        P_t=P_t,
                        P0=P0,
                        diverging_section=system_node.diverging_section,
                        A_throat=A_t,
                        A_exit=A_x,
                        n_v=des_params.eff.flow,
                    )
                )

                # Residual update
                if isinstance(system.energy.line.engine.design_parameters, tuple):
                    eng_des = system.energy.line.engine.design_parameters[0]
                else:
                    eng_des = system.energy.line.engine.design_parameters
                updated_state = update(
                    updated_state,
                    lambda s: s.energy.nodes[self.network_id].residual.mass_flow_rate,
                    ((mdot_out - W_in) / eng_des.mass_flow_rate),
                )

        # Physical outflow
        outputs = state_node.flow

        outputs = update(outputs, "area", jnp.atleast_2d(A_x))
        outputs = update(outputs, "mass_flow_rate", jnp.atleast_2d(mdot_out))
        outputs = update(outputs, "mach_number", jnp.atleast_2d(M_out))
        outputs = update(outputs, "density", jnp.atleast_2d(rho_out))
        outputs = update(outputs, "speed", jnp.atleast_2d(u_out))
        outputs = update(outputs, "pressure", jnp.atleast_2d(P_out))
        outputs = update(outputs, "stagnation_pressure", jnp.atleast_2d(P_t_out))
        outputs = update(outputs, "temperature", jnp.atleast_2d(T_out))
        outputs = update(outputs, "stagnation_temperature", jnp.atleast_2d(T_t_out))
        outputs = update(outputs, "enthalpy", jnp.atleast_2d(h_out))
        outputs = update(outputs, "stagnation_enthalpy", jnp.atleast_2d(h_t_out))

        updated_state = update(updated_state, lambda s: s.energy.nodes[self.network_id].flow, outputs)

        return updated_state, updated_system, settings


# Turboshaft -------------------------------------------------------------------


class Turboshaft(PACTNode):
    inputs: tuple | PACTInput = (
        PACTInput("mechanical", "compressor"),
        PACTInput("mechanical", "turbine"),
    )

    @io.inputs(
        "state.energy.nodes['{mechanical_inputs.network_id}'].mechanical.power",
        "system.energy.nodes['network.line.engine'].design_parameters",
    )
    @io.outputs("state.energy.nodes['{network_id}'].residual.power")
    def transmit(self, state: State, system: System, settings: Settings):

        if settings.analysis.energy.design_mode:
            d_power = self.apply_domain_op(jnp.sum, state, "mechanical", "power") / 2e7

        else:
            if isinstance(system.energy.line.engine.design_parameters, tuple):
                des = system.energy.nodes["network.line.engine"].design_parameters[0]
            else:
                des = system.energy.nodes["network.line.engine"].design_parameters

            d_power = self.apply_domain_op(jnp.sum, state, "mechanical", "power") / des.power  # type: ignore

        outputs = state.energy.nodes[self.network_id]
        outputs = update(outputs, "residual.power", d_power)

        updated_state = update(state, lambda s: s.energy.nodes[self.network_id], outputs)

        return updated_state, system, settings


# ----------------------------------------------------------------------------------------------------------------------
# Engines
# ----------------------------------------------------------------------------------------------------------------------


def _engine_performance(
    u0,
    P0,
    g,
    delta_SFC,
    v_fan_nozzle,
    A_fan_nozzle,
    P_fan_nozzle,
    v_core_nozzle,
    A_core_nozzle,
    P_core_nozzle,
    fuel_air_ratio,
    mdot_core,
    BPR,
):

    # 1. Calculate mass flows
    mdot_fan = mdot_core * BPR
    mdot_in_core = mdot_core / (1.0 + fuel_air_ratio)  # Strip fuel for inlet momentum

    # 2. Raw Dimensional Thrust (Gross Thrust - Ram Drag)
    # Core
    gross_thrust_core = (mdot_core * v_core_nozzle) + (P_core_nozzle - P0) * A_core_nozzle
    ram_drag_core = mdot_in_core * u0
    F_core = gross_thrust_core - ram_drag_core

    # Fan
    gross_thrust_fan = (mdot_fan * v_fan_nozzle) + (P_fan_nozzle - P0) * A_fan_nozzle
    ram_drag_fan = mdot_fan * u0
    F_fan = gross_thrust_fan - ram_drag_fan

    # 3. Total Actual Thrust (in Newtons)
    F_actual = F_core + F_fan

    # 4. Power and Efficiency (Calculated directly from F_actual to avoid singularities)
    p = F_actual * u0

    mdot_fuel = mdot_in_core * fuel_air_ratio

    # Protect against divide-by-zero if fuel flow is exactly 0.0
    safe_mdot_fuel = jnp.maximum(mdot_fuel, 1e-9)
    safe_F_actual = jnp.maximum(F_actual, 1e-9)

    I_sp = F_actual / (safe_mdot_fuel * g)
    TSFC = (safe_mdot_fuel / safe_F_actual) * (1.0 - delta_SFC)  # / units.hr

    # Fuel flow in kg/hr
    ff = mdot_fuel  # * units.parse('kg/hr')

    specific_thrust_core = F_actual / mdot_core

    return F_actual, specific_thrust_core, I_sp, TSFC, p, ff


def _TurbojetSetup():

    inlet = Inlet()
    comp = Compressor()
    burn = Burner()
    turb = Turbine()
    shaft = Turboshaft()
    nozz = Nozzle()

    return (inlet, comp, burn, turb, shaft, nozz)


def _ABTurbojetSetup():

    base_components = _TurbojetSetup()
    ab = Burner(
        name="Afterburner",
        inputs=(PACTInput("flow", "turbine"),),
    )
    nozz = replace(
        base_components[-1],
        inputs=(
            PACTInput("flow", "afterburner"),
            PACTInput("fuel", "afterburner"),
        ),
    )
    return base_components[:-1] + (ab, nozz)


class JetGeometry(Module):
    xe: ScalarFloat = 1.0
    ye: ScalarFloat = 1.0
    Ce: ScalarFloat = 2.0


class JetKinematics(Module):
    """
    Exit Mach numbers for turbojet components
    """

    inlet: ScalarFloat = 0.6

    compressor: ScalarFloat = 0.3
    burner: ScalarFloat = 0.1
    turbine: ScalarFloat = 0.4


class TurbojetOpPoint[KinType: JetKinematics | FanKinematics](FlowOpPoint):
    name: NameType = field("SLS", static=True)  # Sea-Level-Static operation point

    # Performance Parameters
    thrust: ScalarFloat = 0.0
    SLS_thrust: ScalarFloat = 0.0
    delta_SFC: ScalarFloat = 0.0  # noqa: N815

    # Flight Conditions
    altitude: ScalarFloat = 0.0
    mach_number: ScalarFloat = 1e-6

    temperature: ScalarFloat = 288.15  # Kelvin
    stagnation_temperature: ScalarFloat = 288.15  # Kelvin

    pressure: ScalarFloat = 101325.0  # Pascal
    stagnation_pressure: ScalarFloat = 101325.0  # Pascal

    # Component Parameters
    inlet_pressure_recovery: ScalarFloat = 0.999
    overall_pressure_ratio: ScalarFloat = 20.0  # Compressor PR, 'OPR' by convention
    burner_pressure_ratio: ScalarFloat = 0.97

    turbine_intake_temperature: ScalarFloat = 0.0
    afterburner_exit_temperature: ScalarFloat = 0.0

    # Variable/Residual Values
    FAR: ScalarFloat = 1e-2
    TSFC: ScalarFloat = 0.0
    compressor_Rline: ScalarFloat = 2.0  # noqa: N815
    mass_flow_rate: ScalarFloat = 100 * units.kg / units.s

    # Single Spool Variables
    rotation_speed: ScalarFloat = 8_000 * units.rpm
    turbine_PR: ScalarFloat = 5.0  # noqa: N815
    power: ScalarFloat = 2e7 * units.W

    station_mach_numbers: KinType = static_field(JetKinematics)

    def update_state(self, state: State):
        a0 = state.freestream.atmosphere.compute_speed_of_sound(self.altitude)
        M0 = self.mach_number

        op_state = update(
            state,
            (
                ("frames.inertial.position_vector", jnp.array([[0.0, 0.0, -self.altitude]])),
                ("freestream.mach_number", jnp.atleast_2d(self.mach_number)),
                ("frames.inertial.velocity_vector", jnp.atleast_2d(jnp.array([[(a0 * M0).item(), 0.0, 0.0]]))),
                ("energy.target_thrust", jnp.atleast_2d(self.thrust)),
                ("energy.target_temperature", jnp.atleast_2d(self.turbine_intake_temperature)),
            ),
        )
        op_state = op_state.expand_time()

        return op_state


class TurbojetEngine(FlowNode[TurbojetOpPoint]):
    name: str = field("Engine", static=True)
    subcomponents: tuple = field(_TurbojetSetup)

    plug_diameter: ScalarFloat = 0.0

    working_fluid: Gas = field(Air)
    design_parameters: TurbojetOpPoint = field(TurbojetOpPoint)

    inputs: tuple | PACTInput = field(
        (
            PACTInput("flow", "self.core_nozzle"),
            PACTInput("fuel", "self.burner"),
            PACTInput("residual", "self.turboshaft"),
        ),
        static=True,
    )

    installation_geometry: JetGeometry = field(JetGeometry)

    _bookkeeping: dict = static_field(
        lambda: {
            "compressors": Compressor,
            "turbines": Turbine,
            "nozzles": Nozzle | Nozzle,
            "shafts": Turboshaft,
            "ducts": Splitter,
        },
    )

    @classmethod
    def build_custom(
        cls,
        variable_nozzle: bool = True,
        cd_nozzle: bool = True,
        afterburner: bool = False,
        turbofan: bool = False,
        **kwargs,
    ):

        # if turbofan:
        #     if afterburner:
        #         base_components = _ABTurbofanSetup()
        #     else:
        #         base_components = _TurbofanSetup()
        # else:
        #     if afterburner:
        #         base_components = _ABTurbojetSetup()
        #     else:
        #         base_components = _TurbojetSetup()

        inlet = Inlet()
        comp = Compressor()
        comb = Burner()
        turb = Turbine()
        shaft = Turboshaft()
        diverging = variable_nozzle or cd_nozzle
        nozz = Nozzle(variable_exit=variable_nozzle, diverging_section=diverging)

        custom_subs = (inlet, comp, comb, turb, shaft, nozz)

        return cls(subcomponents=custom_subs, **kwargs)

    @classmethod
    def from_json(cls, filepath: str | Path):

        with open(filepath, "r") as f:
            data = json.load(f)

        # Extract Type
        engine_cat = data.get("category", "civil").lower()
        engine_type = data.get("type", "turbojet").lower()
        engine_ab = data.get("AB", False)

        if engine_type == "turbofan":
            if engine_cat == "civil":
                # Create Synthetic TOC point
                des_kwargs = {
                    "name": "TOC",
                    "mach_number": data.get("Cruise Mach", 0.8),
                    "altitude": data.get("Cruise Alt (kft)", 35.0) * 1000.0 * units.ft,
                    "thrust": data.get("Takeoff Thrust (lbf)", 0.0) * 0.25 * units.lbf,
                    "SLS_thrust": data.get("Takeoff Thrust (lbf)", 0.0) * units.lbf,
                    "bypass_ratio": data.get("Takeoff BR", 0.0),
                    "overall_pressure_ratio": data.get("Takeoff OPR", 0.0),
                    "turbine_intake_temperature": (data.get("TIT (F)", 2300.0) + 459.67) * units.R,
                    "mass_flow_rate": data.get("Takeoff Airflow (lbm/s)", 0.0) * units.lbm / units.s,
                    "TSFC": data.get("Cruise TSFC", 0.0) * units.liter / units.hr,
                }

                cruise_kwargs = {
                    "name": "Cruise",
                    "mach_number": data.get("Cruise Mach", 0.8),
                    "altitude": data.get("Cruise Alt (kft)", 35.0) * 1000.0 * units.ft,
                    "thrust": data.get("Cruise Thrust (lbf)", 0.0) * 0.25 * units.lbf,
                    "SLS_thrust": data.get("Takeoff Thrust (lbf)", 0.0) * units.lbf,
                    "bypass_ratio": data.get("Takeoff BR", 0.0),
                    "overall_pressure_ratio": data.get("Takeoff OPR", 0.0),
                    "turbine_intake_temperature": (data.get("TIT (F)", 2300.0) + 459.67) * units.R,
                    "mass_flow_rate": data.get("Takeoff Airflow (lbm/s)", 0.0) * 0.1 * units.lbm / units.s,
                    "TSFC": data.get("Cruise TSFC", 0.0) * units.liter / units.hr,
                }

                takeoff_kwargs = {
                    "name": "Takeoff",
                    "mach_number": 1e-6,
                    "altitude": 0.0,
                    "thrust": data.get("Takeoff Thrust (lbf)", 0.0) * units.lbf,
                    "SLS_thrust": data.get("Takeoff Thrust (lbf)", 0.0) * units.lbf,
                    "bypass_ratio": data.get("Takeoff BR", 0.0),
                    "overall_pressure_ratio": data.get("Takeoff OPR", 0.0),
                    "turbine_intake_temperature": (data.get("TIT (F)", 2300.0) + 459.67) * units.R,
                    "mass_flow_rate": data.get("Takeoff Airflow (lbm/s)", 0.0) * units.lbm / units.s,
                    "TSFC": 0.0,
                }

                des_params = tuple(TurbofanDesign(**k) for k in [des_kwargs, cruise_kwargs, takeoff_kwargs])

            else:
                des_kwargs = {
                    "mach_number": 1e-6,
                    "altitude": 0.0,
                    "thrust": data.get("Thrust (lbf)", 0.0) * units.lbf,
                    "SLS_thrust": data.get("Thrust (lbf)", 0.0) * units.lbf,
                    "bypass_ratio": data.get("BR", 0.0),
                    "overall_pressure_ratio": data.get("OPR", 0.0),
                    "fan_pressure_ratio": data.get("FPR", 0.0),
                    "turbine_intake_temperature": (data.get("TIT (F)", 2300.0) + 459.67) * units.R,
                    "afterburner_exit_temperature": (data.get("AET (F)", -459.67) + 459.67) * units.R,
                    "mass_flow_rate": data.get("Takeoff Airflow (lbm/s)", 0.0) * units.lbm / units.s,
                }

                des_params = TurbofanDesign(**des_kwargs)

            return TurbojetEngine(
                subcomponents=_ABTurbofanSetup() if engine_ab else _TurbofanSetup(),
                inputs=(
                    PACTInput("flow", "self.afterburner"),
                    PACTInput("fuel", "self.afterburner"),
                    PACTInput("fuel", "self.burner"),
                    PACTInput("residual", "self.lp_shaft"),
                    PACTInput("residual", "self.hp_shaft"),
                )
                if engine_ab
                else (
                    PACTInput("flow", "self.core_nozzle"),
                    PACTInput("flow", "self.fan_nozzle"),
                    PACTInput("fuel", "self.burner"),
                    PACTInput("residual", "self.lp_shaft"),
                    PACTInput("residual", "self.hp_shaft"),
                ),
                design_parameters=des_params,
            )

        elif engine_type == "turbojet":
            return TurbojetEngine(
                subcomponents=_ABTurbojetSetup() if engine_ab else _TurbojetSetup(),
                inputs=(
                    PACTInput("flow", "self.afterburner"),
                    PACTInput("fuel", "self.afterburner"),
                    PACTInput("fuel", "self.burner"),
                    PACTInput("residual", "self.turboshaft"),
                )
                if engine_ab
                else (
                    PACTInput("flow", "self.core_nozzle"),
                    PACTInput("flow", "self.fan_nozzle"),
                    PACTInput("fuel", "self.burner"),
                    PACTInput("residual", "self.turboshaft"),
                ),
                design_parameters=TurbojetOpPoint(
                    mach_number=1e-6,
                    altitude=0.0,
                    thrust=data.get("Takeoff Thrust (lbf)", 0.0) * units.lbf,
                    SLS_thrust=data.get("Takeoff Thrust (lbf)", 0.0) * units.lbf,
                    overall_pressure_ratio=data.get("Takeoff OPR", 0.0),
                    turbine_intake_temperature=(data.get("TIT (F)", 2300.0) + 459.67) * units.R,
                    mass_flow_rate=data.get("Takeoff Airflow (lbm/s)", 0.0) * units.lbm / units.s,
                ),
            )

    def design_update(self):

        des = self.design_parameters

        OPR = des.overall_pressure_ratio
        if isinstance(des, TurbofanDesign):
            k = (OPR / 240.0) ** (1.0 / 3.0)
            fan_PR = 3.0 * k
            LPC_PR = 4.0 * k
            HPC_PR = 20.0 * k

            des_engine = update(
                self,
                (
                    ("inlet.design_parameters.pressure_recovery", des.inlet_pressure_recovery),
                    ("fan.design_parameters.rotation_speed", des.lp_rotation_speed),
                    ("fan.design_parameters.pressure_ratio", fan_PR),
                    ("lpc.design_parameters.rotation_speed", des.lp_rotation_speed),
                    ("lpc.design_parameters.pressure_ratio", LPC_PR),
                    ("hpc.design_parameters.rotation_speed", des.hp_rotation_speed),
                    ("hpc.design_parameters.pressure_ratio", HPC_PR),
                    ("burner.design_parameters.pressure_ratio", des.burner_pressure_ratio),
                    ("burner.design_parameters.output_temperature", des.turbine_intake_temperature),
                    ("hpt.design_parameters.rotation_speed", des.hp_rotation_speed),
                    ("lpt.design_parameters.rotation_speed", des.lp_rotation_speed),
                ),
            )
        else:
            des_engine = update(
                self,
                (
                    ("compressor.design_parameters.rotation_speed", des.rotation_speed),
                    ("compressor.design_parameters.pressure_ratio", OPR),
                    ("burner.design_parameters.pressure_ratio", des.burner_pressure_ratio),
                    ("burner.design_parameters.output_temperature", des.turbine_intake_temperature),
                    ("turbine.design_parameters.rotation_speed", des.rotation_speed),
                    ("turbine.design_parameters.pressure_ratio", des.turbine_PR),
                ),
            )

        return des_engine

    @io.inputs(
        "state.freestream",
        "state.energy.throttle",
        "state.energy.nodes['{flow_inputs.network_id}'].flow",
        "system.energy.nodes['{network_id}'].design_parameters",
    )
    @io.outputs(
        "state.energy.nodes['{network_id}'].force.thrust",
        "state.energy.nodes['{network_id}'].force.nondimensional_thrust",
        "state.energy.nodes['{network_id}'].force.specific_impulse",
        "state.energy.nodes['{network_id}'].fuel.TSFC",
        "state.energy.nodes['{network_id}'].fuel.flow_rate",
        "state.energy.nodes['{network_id}'].flow.mass_flow_rate",
        "state.energy.nodes['{network_id}'].mechanical.power",
    )
    def transmit(self, state: State, system: System, settings: Settings):

        fs = state.freestream
        FAR = state.energy.nodes[self.network_id + ".burner"].flow.fuel_air_ratio

        # Core Flow
        core_flow = next((f for f in self.flow_inputs if "core" in f.network_id), None)

        v_core = self.get_input_state(state, core_flow, "speed") if core_flow is not None else 0.0
        A_core = self.get_input_state(state, core_flow, "area") if core_flow is not None else 0.0
        P_core = self.get_input_state(state, core_flow, "pressure") if core_flow is not None else 0.0

        mdot_core = self.get_input_state(state, core_flow, "mass_flow_rate") if core_flow is not None else 1.0

        # Fan Flow
        fan_flow = next((f for f in self.flow_inputs if "fan" in f.network_id), None)

        v_fan = self.get_input_state(state, fan_flow, "speed") if fan_flow is not None else 0.0
        A_fan = self.get_input_state(state, fan_flow, "area") if fan_flow is not None else 0.0
        P_fan = self.get_input_state(state, fan_flow, "pressure") if fan_flow is not None else 0.0

        BPR = getattr(state.energy, "bypass_ratio", 0.0)
        des = system.energy.nodes[self.network_id].design_parameters
        if isinstance(des, tuple):
            des = des[0]
        else:
            des = des

        F, F_sp, I_sp, TSFC, p, ff = _engine_performance(
            u0=fs.speed,
            P0=fs.pressure,
            g=fs.gravity,
            delta_SFC=des.delta_SFC,
            v_fan_nozzle=v_fan,
            A_fan_nozzle=A_fan,
            P_fan_nozzle=P_fan,
            v_core_nozzle=v_core,
            A_core_nozzle=A_core,
            P_core_nozzle=P_core,
            fuel_air_ratio=FAR,
            mdot_core=mdot_core,
            BPR=BPR,
        )

        outputs = state.energy.nodes[self.network_id]

        outputs = update(outputs, "force.thrust", F)
        outputs = update(outputs, "force.nondimensional_thrust", F_sp)
        outputs = update(outputs, "force.specific_impulse", I_sp)
        outputs = update(outputs, "fuel.TSFC", TSFC)
        outputs = update(outputs, "fuel.flow_rate", ff)
        outputs = update(outputs, "flow.mass_flow_rate", mdot_core)
        outputs = update(outputs, "mechanical.power", p)

        outputs = update(outputs, "residual.thrust", (F - des.thrust) / des.thrust)
        outputs = update(outputs, "residual.power", self.apply_domain_op(jnp.sum, state, "residual", "power"))

        updated_state = update(state, lambda s: s.energy.nodes[self.network_id], outputs)

        return updated_state, system, settings


# Makes BPR split serializable for save/load
class BPRSplit(Module):
    is_bypass: bool = field(True, static=True)

    def __call__(self, state):
        bpr = state.energy.bypass_ratio
        if self.is_bypass:
            return bpr / (1.0 + bpr)
        else:
            return 1.0 / (1.0 + bpr)


def _TurbofanSetup():

    inlet = Inlet()
    fan = Compressor(
        name="Fan",
        map=map_data.Fan,
    )

    # Core Flow ----------------------------------------------------------------

    core_flow = Splitter(
        name="Core Flow",
        inputs=PACTInput("flow", "fan"),
        values=("mass_flow_rate",),
        fractions=BPRSplit(is_bypass=False),
    )
    core_duct = FlowNode(name="Core Duct", inputs=PACTInput("flow", "core flow"))

    # Compressors
    lpc = Compressor(name="LPC", map=map_data.LPC, inputs=PACTInput("flow", "core duct"))

    c_stat = FlowNode(name="Compressor Stator", inputs=PACTInput("flow", "lpc"))

    hpc = Compressor(
        name="HPC",
        map=map_data.HPC,
        inputs=PACTInput("flow", "compressor stator"),
        output_bleeds=(
            BleedFlow(
                name="outlet",
                fractions_dict={"mass_flow_rate": 0.05, "stagnation_pressure": 0.5, "stagnation_enthalpy": 0.5},
            ),
            BleedFlow(
                name="LPT cooling",
                fractions_dict={"mass_flow_rate": 0.05, "stagnation_pressure": 0.5, "stagnation_enthalpy": 0.5},
            ),
            BleedFlow(
                name="nozzle cooling",
                fractions_dict={"mass_flow_rate": 0.02, "stagnation_pressure": 0.5, "stagnation_enthalpy": 0.5},
            ),
        ),
    )

    cooling = FlowNode(
        name="Cooling Duct",
        inputs=PACTInput("flow", "hpc"),
        output_bleeds=(
            BleedFlow(
                name="HPT cooling",
                fractions_dict={"mass_flow_rate": 0.05, "stagnation_pressure": 0.5, "stagnation_enthalpy": 0.5},
            ),
            BleedFlow(
                name="LPT cooling",
                fractions_dict={"mass_flow_rate": 0.10, "stagnation_pressure": 0.5, "stagnation_enthalpy": 0.5},
            ),
        ),
    )

    # burner
    comb = Burner(inputs=PACTInput("flow", "cooling_duct"))

    # Turbines
    hpt = Turbine(
        name="HPT",
        map=map_data.HPT,
        inputs=(
            PACTInput("flow", "burner", primary=True),
            PACTInput("flow", "cooling_duct.hpt_cooling"),
        ),
    )

    t_stat = FlowNode(
        name="Turbine Stator",
        inputs=(
            PACTInput("flow", "hpt", primary=True),
            PACTInput("flow", "hpc.lpt_cooling"),
            PACTInput("flow", "cooling_duct.lpt_cooling"),
        ),
    )

    lpt = Turbine(name="LPT", map=map_data.LPT, inputs=PACTInput("flow", "turbine_stator"))

    # Turboshafts
    lp_shaft = Turboshaft(
        name="LP Shaft",
        inputs=(
            PACTInput("mechanical", "lpc"),
            PACTInput("mechanical", "fan"),
            PACTInput("mechanical", "lpt"),
        ),
    )

    hp_shaft = Turboshaft(
        name="HP Shaft",
        inputs=(
            PACTInput("mechanical", "hpc"),
            PACTInput("mechanical", "hpt"),
        ),
    )

    # Core Nozzle
    cn_duct = FlowNode(
        name="Core Nozzle Duct",
        inputs=(
            PACTInput("flow", "lpt", primary=True),
            PACTInput("flow", "hpc.nozzle_cooling"),
        ),
    )
    c_nozz = Nozzle(inputs=PACTInput("flow", "core_nozzle_duct"))

    # Bypass Flow --------------------------------------------------------------
    fan_flow = Splitter(
        name="Fan Flow",
        inputs=PACTInput("flow", "fan"),
        values=("mass_flow_rate",),
        fractions=BPRSplit(is_bypass=True),
    )

    fn_duct = FlowNode(
        name="Fan Duct",
        inputs=PACTInput("flow", "fan flow"),
        output_bleeds=(BleedFlow(name="outlet", fractions_dict={"mass_flow_rate": 0.005}),),
    )

    f_nozz = Nozzle(name="Fan Nozzle", inputs=(PACTInput("flow", "fan duct")))

    return (
        inlet,
        fan,
        core_flow,
        core_duct,
        lpc,
        c_stat,
        hpc,
        cooling,
        comb,
        hpt,
        t_stat,
        lpt,
        lp_shaft,
        hp_shaft,
        cn_duct,
        c_nozz,
        fan_flow,
        fn_duct,
        f_nozz,
    )


def _ABTurbofanSetup():

    base_components = _TurbofanSetup()
    ab = Burner(
        name="Afterburner",
        inputs=(
            PACTInput("flow", "fan_nozzle"),
            PACTInput("flow", "core_nozzle"),
        ),
        add_mixer=True,
    )
    return base_components + (ab,)


class FanKinematics(Module):
    """
    Exit Mach Numbers for turbofan components
    """

    inlet: ScalarFloat = 0.75
    fan: ScalarFloat = 0.45

    core_duct: ScalarFloat = 0.35
    fan_duct: ScalarFloat = 0.45

    lpc: ScalarFloat = 0.3
    compressor_stator: ScalarFloat = 0.35
    hpc: ScalarFloat = 0.25
    cooling_duct: ScalarFloat = 0.3

    burner: ScalarFloat = 0.1

    hpt: ScalarFloat = 0.35
    turbine_stator: ScalarFloat = 0.3
    lpt: ScalarFloat = 0.4

    core_nozzle_duct: ScalarFloat = 0.45


class TurbofanDesign(TurbojetOpPoint[FanKinematics]):
    # Variable Values
    bypass_ratio: ScalarFloat = 0.0
    fan_pressure_ratio: ScalarFloat = 0.0

    lp_rotation_speed: ScalarFloat = 5_000 * units.rev / units.mins
    hp_rotation_speed: ScalarFloat = 15_000 * units.rev / units.mins

    HPT_PR: ScalarFloat = 5.0
    LPT_PR: ScalarFloat = 3.0

    station_mach_numbers: FanKinematics = field(FanKinematics, static=True)


def TurbofanEngine(**kwargs):

    return TurbojetEngine(
        subcomponents=_TurbofanSetup(),
        inputs=(
            PACTInput("flow", "self.core_nozzle"),
            PACTInput("flow", "self.fan_nozzle"),
            PACTInput("fuel", "self.burner"),
            PACTInput("residual", "self.lp_shaft"),
            PACTInput("residual", "self.hp_shaft"),
        ),
        design_parameters=TurbofanDesign(),
        **kwargs,
    )


# ----------------------------------------------------------------------------------------------------------------------
#  Lines
# ----------------------------------------------------------------------------------------------------------------------

# Turbojet ---------------------------------------------------------------------


def _TurbojetLineSetup():
    return TurbojetEngine(), FuelTank()


class TurbojetLine(PACTLine):
    subcomponents: tuple = field(_TurbojetLineSetup)

    inputs: tuple | PACTInput = field(
        (
            PACTInput("fuel", "self.engine"),
            PACTInput("force", "self.engine"),
            PACTInput("residual", "self.engine"),
        ),
        static=True,
    )

    tank_draw_ratios: tuple[float, ...] = field((1.0,))

    _bookkeeping: dict = field(
        lambda: {
            "engines": TurbojetEngine,
            "stores": FuelTank,
            "fuel_tanks": FuelTank,
        },
        static=True,
    )

    @io.inputs(
        # "state.energy.nodes['{fuel_tanks.network_id}'].mass",
        "state.energy.nodes['{fuel_inputs.network_id}'].fuel.flow_rate",
        # "system.energy.nodes['{fuel_tanks.network_id}'].selector_ratio",
        # "system.energy.nodes['{fuel_tanks.network_id}'].mass_properties.total",
        "system.energy.nodes['{network_id}'].tank_draw_ratios",
    )
    @io.outputs(
        # "state.energy.nodes['{fuel_tanks}'].fuel.flow_rate",
        "state.mass.rate_of_change",
        "state.energy.nodes['{network_id}'].force.thrust",
        "state.energy.nodes['{network_id}'].residual.thrust",
        "state.energy.nodes['{network_id}'].residual.power",
    )
    def transmit(self, state: State, system: System, settings: Settings):

        # Fuel Burn ------------------------------------------------------------
        total_fuel_burn = self.apply_domain_op(jnp.sum, state, "fuel", "flow_rate")

        #  Compute fuel fraction
        total_fuel_mass = jnp.sum(jnp.asarray([t.mass_properties.total for t in self.fuel_tanks]))
        current_fuel_mass = jnp.sum(jnp.asarray([state.energy.nodes[t.network_id].mass for t in self.fuel_tanks]))
        fuel_fraction = current_fuel_mass / jnp.where(total_fuel_mass > 1e-6, total_fuel_mass, 1e-6)

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
        updated_state = update(
            state,
            lambda s: tuple(s.energy.nodes[t.network_id].fuel.flow_rate for t in self.fuel_tanks),
            tank_burns,
        )

        updated_state = update(
            updated_state,
            ("mass.rate_of_change", updated_state.mass.rate_of_change - total_fuel_burn),
        )

        outputs = state.energy.nodes[self.network_id]

        # Total Thrust ---------------------------------------------------------

        outputs = update(outputs, "force.thrust", self.apply_domain_op(jnp.sum, updated_state, "force", "thrust"))
        outputs = update(outputs, "residual.thrust", self.apply_domain_op(jnp.sum, updated_state, "residual", "thrust"))
        outputs = update(outputs, "residual.power", self.apply_domain_op(jnp.sum, updated_state, "residual", "power"))

        updated_state = update(state, lambda s: s.energy.nodes[self.network_id], outputs)

        return updated_state, system, settings


# Turbofan ---------------------------------------------------------------------


def _TurbofanLineSetup():
    return TurbofanEngine(), FuelTank()


def TurbofanLine(**kwargs):

    if "subcomponents" not in kwargs:
        kwargs["subcomponents"] = _TurbofanLineSetup()

    return TurbojetLine(**kwargs)


# ----------------------------------------------------------------------------------------------------------------------
#  Turbojet Energy Networks
# ----------------------------------------------------------------------------------------------------------------------


class _JetNetwork[DesignType: JetNetParameters](PACTNetwork[DesignType]):
    """
    Jet network shell without design parameters.
    """

    inputs: tuple | PACTInput = field(
        (
            PACTInput("force", "network.line"),
            PACTInput("residual", "network.line"),
        )
    )

    @io.inputs(
        "state.energy.nodes['{force_inputs.network_id}'].force.thrust",
        "state.energy.nodes['{residual_inputs.network_id}'].residual.power",
        "state.energy.target_thrust",
    )
    @io.outputs(
        "state.energy.total_force_vector",
        "state.energy.residual.thrust",
        "state.energy.residual.power",
    )
    def transmit(self, state: State, system: System, settings: Settings):

        updated_state = state

        # Total Thrust----------------------------------------------------------

        total_thrust = jnp.atleast_2d(self.apply_domain_op(jnp.sum, state, "force", "thrust"))
        total_force_vector = jnp.hstack((total_thrust, jnp.zeros((total_thrust.shape[0], 2))))

        updated_state = update(
            updated_state,
            (
                ("energy.total_force_vector", total_force_vector),
                ("energy.residual.thrust", (total_thrust - state.energy.target_thrust) / state.energy.target_thrust),
            ),
        )

        # Power Imbalance (Single Spool Only) ----------------------------------

        total_d_power = self.apply_domain_op(jnp.sum, updated_state, "residual", "power")

        updated_state = update(updated_state, "energy.residual.power", total_d_power)

        return updated_state, system, settings


# Turbojet ---------------------------------------------------------------------


def _TurbojetNetworkSetup():
    return (TurbojetLine(name="Line"),)


class JetNetParameters(NetworkParameters):
    number_of_engines: int = field(1, static=True)


class TurbojetNetwork(_JetNetwork[JetNetParameters]):
    subcomponents: tuple = field(_TurbojetNetworkSetup)
    design_parameters: JetNetParameters = field(JetNetParameters)


# Turbofan ---------------------------------------------------------------------


def _TurbofanNetworkSetup():
    return (TurbofanLine(),)


class TurbofanNetwork(_JetNetwork[JetNetParameters]):
    subcomponents: tuple = field(_TurbofanNetworkSetup)
