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
    from RCAIDE.Library.Components.Energy.Propulsors import TurbojetEngine

import warnings

import jax.numpy as jnp
import equinox as eqx

from RCAIDE.Library.Atmospheres import USStandard1976
from RCAIDE.Library.Components.Energy.Lines.Jets import TurbojetEnergyLine
from RCAIDE.Library.Components.Energy.Networks import EnergyNetwork


from RCAIDE.Framework import State, Aircraft, Settings
from RCAIDE.Framework.Missions.Initialize import initialize_energy
from .GraphNetwork import build_analysis_from_network
# ----------------------------------------------------------------------------------------------------------------------
#  Propulsion Sizing Analyses
# ----------------------------------------------------------------------------------------------------------------------

def update_design_parameters(turbojet: TurbojetEngine):
        design_thrust = turbojet.design_parameters.total_thrust
        
        if design_thrust == 0.0:
            warnings.warn("Attempted to calculate sea-level static thrust without reference design thrust. "
            f"Please set {turbojet.tag}.design_parameters.total_thrust.")
            return turbojet
        
        atmo = USStandard1976()
        T0 = atmo.compute_temperature(0.0)
        a0 = atmo.compute_speed_of_sound(0.0)
        M0 = 0.01

        sls_state = eqx.tree_at(lambda s:(
            s.freestream.altitude,
            s.freestream.gravity,
            s.freestream.mach_number,
            s.freestream.speed_of_sound,
            s.freestream.speed,
            s.freestream.temperature,
            s.freestream.pressure,
            s.freestream.density,
            s.freestream.dynamic_viscosity,
            s.freestream.gamma,
            s.freestream.Cp,
            s.freestream.R,
        ), State().expand_rows(1),
        (
            jnp.atleast_1d(0.),
            jnp.atleast_1d(9.81),
            jnp.atleast_1d(M0),
            jnp.atleast_1d(a0),
            jnp.atleast_1d(a0*M0),
            jnp.atleast_1d(T0),
            jnp.atleast_1d(atmo.compute_pressure(0.0)),
            jnp.atleast_1d(atmo.compute_density(0.0)),
            jnp.atleast_1d(atmo.compute_dynamic_viscosity(0.0)),
            jnp.atleast_1d(turbojet.working_fluid.compute_gamma(T0)),
            jnp.atleast_1d(turbojet.working_fluid.compute_Cp(T0)),
            jnp.atleast_1d(turbojet.working_fluid.R_specific),
        ))
        
        sls_line = TurbojetEnergyLine(subcomponents=(turbojet,), fuel_inputs=("Engine 1",))
        sls_network = EnergyNetwork(subcomponents=(sls_line,))
        sls_system = Aircraft(subcomponents=(sls_network,))


        sls_state, sls_system, settings = initialize_energy(sls_state, sls_system, Settings())
        sls_analysis = build_analysis_from_network(sls_system.energy_networks[0])

        sls_state, sls_system, settings = sls_analysis(sls_state, sls_system, settings)
        sls_engine = sls_system.energy_networks[0].lines[0].engines[0]

        sls_thrust = sls_state.energy.nodes[sls_engine.tag].outputs.force.thrust.item(0)

        updated_params = eqx.tree_at(lambda d: d.SLS_thrust, sls_engine.design_parameters, sls_thrust)
        updated_turbojet = eqx.tree_at(lambda s: s.design_parameters, sls_engine, updated_params)

        return updated_turbojet