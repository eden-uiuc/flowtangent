# RCAIDE/Framework/Analyses/Propulsion/sizing.py
# (c) Copyright 2026 Aerospace Research Community LLC
#
# Created: Jun 2026, J. Smart
# Modified: Jun 2026, J. Smart

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING

# --- Framework Imports (Strictly for Type Hinting to avoid Circular Imports) ---
if TYPE_CHECKING:
    from RCAIDE.library.components.energy.propulsors import TurbojetEngine

import warnings

import equinox as eqx
import jax.numpy as jnp

from RCAIDE.library.atmospheres import USStandard1976
from RCAIDE.library.components.energy.lines import TurbojetEnergyLine
from RCAIDE.library.components.energy.networks import EnergyNetwork

from RCAIDE.framework import Aircraft, Settings, State
from RCAIDE.framework.missions.initialize import initialize_energy
from RCAIDE.framework.missions.update import update_freestream

from .graph_network import build_analysis_from_network

# ----------------------------------------------------------------------------------------------------------------------
#  Propulsion Sizing Analyses
# ----------------------------------------------------------------------------------------------------------------------


def update_design_parameters(turbojet: TurbojetEngine):
    design_thrust = turbojet.design_parameters.thrust

    if design_thrust == 0.0:
        warnings.warn(
            "Attempted to calculate sea-level static thrust without reference design thrust. "
            f"Please set {turbojet.tag}.design_parameters.total_thrust."
        )
        return turbojet

    atmo = USStandard1976()
    T0 = atmo.compute_temperature(0.0)
    a0 = atmo.compute_speed_of_sound(0.0)
    M0 = 0.01

    sls_state = eqx.tree_at(
        lambda s: (
            s.freestream.altitude,
            s.freestream.gravity,
            s.freestream.mach_number,
            s.freestream.speed_of_sound,
            s.frames.inertial.velocity_vector,
            s.freestream.temperature,
            s.freestream.pressure,
            s.freestream.density,
            s.freestream.dynamic_viscosity,
            s.freestream.gamma,
            s.freestream.Cp,
            s.freestream.R,
        ),
        State().expand_rows(1),
        (
            jnp.atleast_2d(0.0),
            jnp.atleast_2d(9.81),
            jnp.atleast_2d(M0),
            jnp.atleast_2d(a0),
            jnp.atleast_2d(jnp.array([[a0 * M0, 0.0, 0.0]])),
            jnp.atleast_2d(T0),
            jnp.atleast_2d(atmo.compute_pressure(0.0)),
            jnp.atleast_2d(atmo.compute_density(0.0)),
            jnp.atleast_2d(atmo.compute_dynamic_viscosity(0.0)),
            jnp.atleast_2d(turbojet.working_fluid.compute_gamma(T0)),
            jnp.atleast_2d(turbojet.working_fluid.compute_Cp(T0)),
            jnp.atleast_2d(turbojet.working_fluid.R_specific),
        ),
    )

    sls_line = TurbojetEnergyLine(subcomponents=(turbojet,), fuel_inputs=("self.engine_1",))
    sls_network = EnergyNetwork(subcomponents=(sls_line,))
    sls_system = Aircraft(subcomponents=(sls_network,))

    sls_state, sls_system, sls_settings = initialize_energy(sls_state, sls_system, Settings())
    sls_state, sls_system, sls_settings = update_freestream(sls_state, sls_system, sls_settings)
    sls_state = eqx.tree_at(
        lambda s: s.energy.nodes["energy_network.turbojet_energy_line.engine_1"].throttle,
        sls_state,
        jnp.atleast_2d(1.0),
    )

    sls_analysis = build_analysis_from_network(sls_system.energy)

    sls_state, sls_system, sls_settings = sls_analysis(sls_state, sls_system, sls_settings)
    sls_engine = sls_system.energy.nodes["energy_network.turbojet_energy_line.engine_1"]
    sls_thrust = sls_state.energy.nodes["energy_network.turbojet_energy_line.engine_1"].outputs.force.thrust.item(0)

    updated_params = eqx.tree_at(lambda d: d.SLS_thrust, sls_engine.design_parameters, sls_thrust)
    updated_turbojet = eqx.tree_at(lambda s: s.design_parameters, turbojet, updated_params)

    return updated_turbojet


# def
