# RCAIDE/Framework/analyses/energy/turbojets.py
# (c) Copyright 2026 Aerospace Research Community LLC
#
# Created: Jun 2026, J Smart
# Modified: Jun 2026, J Smart

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from __future__ import annotations
from typing import TYPE_CHECKING
# --- Framework Imports (Strictly for Type Hinting to avoid Circular Imports) ---
if TYPE_CHECKING:
    from RCAIDE.framework import System
    from RCAIDE.framework.conditions.Energy import TurbojetNetworkConditions
    from RCAIDE.library.components.energy.networks import TurbojetEnergyNetwork, TurbojetDesign

from dataclasses import replace

import jax.numpy as jnp
import equinox as eqx

from ..residual import ResidualAnalysis
from .graph_network import build_analysis_from_network

from RCAIDE.utils import init_field, DataPath

from RCAIDE.library import units

from RCAIDE.framework import State, Settings
from RCAIDE.framework.settings import EnergyAnalysisSettings
from RCAIDE.framework.analyses.residual import ResidualAnalysis
from RCAIDE.framework.conditions.Controls import Control, Residual

from RCAIDE.framework.missions.initialize import initialize_energy
from RCAIDE.framework.missions.update import update_freestream

# ----------------------------------------------------------------------------------------------------------------------
#  Design Point Turbojet Analysis
# ----------------------------------------------------------------------------------------------------------------------

def design_turbojet(system: System):

    # Setup test state according to design parameters

    network: TurbojetEnergyNetwork = system.energy_networks[0]
    des: TurbojetDesign = network.design_parameters

    alt = des.altitude
    M0 = des.mach_number

    atmo = des.atmosphere_model
    a0 = atmo.compute_speed_of_sound(alt)

    des_state = eqx.tree_at(
        lambda s: (
            s.frames.inertial.position_vector,
            s.freestream.mach_number,
            s.frames.inertial.velocity_vector,
        ),
        State().expand_rows(1),
        (
            jnp.array([[0., 0., -alt]]),
            jnp.atleast_2d(M0),
            jnp.atleast_2d(jnp.array([[a0 * M0, 0.0, 0.0]])),
        ),
    )

    des_e_settings = EnergyAnalysisSettings(design_mode=True)
    des_settings = eqx.tree_at(lambda s: s.analysis.energy, Settings(DEBUG_MODE=True), des_e_settings)
    des_state, des_system, des_settings = initialize_energy(des_state, system, des_settings)
    des_state, des_system, des_settings = update_freestream(des_state, des_system, des_settings)

    # Build design analysis

    base_analysis = build_analysis_from_network(des_system.energy_networks[0])

    mass_ctrl = Control(
        tag="Mass Flow Rate",
        state_path=DataPath(("energy", "design_mass_flow_rate")),
        initial_value=des.initial_MFR,
    )

    turb_ctrl = Control(
        tag="Turbine Pressure Ratio",
        state_path=DataPath(("energy", "design_turbine_PR")),
        initial_value=des.initial_turb_PR,
    )

    thrust_res = Residual(
        tag="Design Thrust Residual",
        get_value=lambda s: s.energy.outputs.residual.thrust
    )

    work_res = Residual(
        tag="Work Residual",
        get_value=lambda s: s.energy.outputs.residual.work
    )

    design_analysis = ResidualAnalysis(
        tag="Turbojet Design",
        analyze=base_analysis,
        controls=(mass_ctrl, turb_ctrl),
        residuals=(thrust_res, work_res)
    )

    des_state, des_system, des_settings = design_analysis(des_state, des_system, des_settings)
    
    return des_state, des_system

# ----------------------------------------------------------------------------------------------------------------------
#  Off-Design Turbojet Analysis
# ----------------------------------------------------------------------------------------------------------------------

def _TurbojetControls():

    Rline = Control(
        tag="Rline",
        state_path=DataPath(("energy", "Rline")),
    )

    turb_PR = Control(
        tag="Turbine Pressure Ratio",
        state_path=DataPath(("energy", "turbine_PR")),
    )

    N = Control(
        tag="Rotation Speed",
        state_path=DataPath(("energy", "rotation_speed")),
    )

    return Rline, turb_PR, N

def _TurbojetResiduals():

    # Mass flow residual, mostly dependent on Rline
    d_mass = d_work = Residual(
        tag="Mass",
        get_value=lambda s: s.energy.mass_flow_imbalance
    )
    
    # Work residual, mostly depended on turbine PR
    d_work = Residual(
        tag="Work",
        get_value=lambda s: s.energy.work_imbalance
    )
    
    def target_thrust_balance(state: State):
        network_state: TurbojetNetworkConditions = state.energy
        return network_state.total_force_vector - network_state.design_thrust_vector
    
    # Thrust residual, mostly dependent on N
    thrust = Residual(
        tag="Thrust",
        get_value=target_thrust_balance
    )

    return d_mass, d_work, thrust


class TurbojetAnalysis(ResidualAnalysis):

    tag: str = init_field("Turbojet Design", static=True)

    controls: tuple[Control, ...] = init_field(_TurbojetControls)
    residuals: tuple[Residual, ...] = init_field(_TurbojetResiduals)

    def __call__(self, state: State, system: System, settings: Settings) -> tuple[State, System, Settings]:
        
        # Temporarily set design mode on
        design_settings = eqx.tree_at(lambda s: s.analysis.energy.design_mode, settings, True)
        
        # Run residual analysis
        analysis_state, analysis_system, analysis_settings = super().__call__(state, system, design_settings)

        # Turn design mode off
        analysis_settings = eqx.tree_at(lambda s: s.analysis.energy.design_mode, analysis_settings, False)

        return analysis_state, analysis_system, analysis_settings

    


    
