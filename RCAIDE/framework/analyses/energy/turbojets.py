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
from jaxopt import Broyden

from ..residual import ResidualAnalysis
from .graph_network import build_analysis_from_network

from RCAIDE.utils import DataPath

from RCAIDE.framework import State, Aircraft, Settings
from RCAIDE.framework.settings import EnergyAnalysisSettings
from RCAIDE.framework.analyses.residual import ResidualAnalysis
from RCAIDE.framework.conditions.controls import Control, Residual

from RCAIDE.framework.missions.initialize import initialize_energy
from RCAIDE.framework.missions.update import update_freestream

# ----------------------------------------------------------------------------------------------------------------------
#  Design Point Turbojet Analysis
# ----------------------------------------------------------------------------------------------------------------------

def design_turbojet(state: State, system: Aircraft, settings: Settings) -> tuple[State, Aircraft, Settings]:

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
    des_settings = eqx.tree_at(lambda s: s.analysis.energy, settings, des_e_settings)
    des_state, des_system, des_settings = initialize_energy(des_state, system, des_settings)
    des_state, des_system, des_settings = update_freestream(des_state, des_system, des_settings)

    # Build design analysis

    base_analysis = build_analysis_from_network(des_system.energy_networks[0])

    mass_ctrl = Control(
        tag="Mass Flow Rate",
        state_path=DataPath(("energy", "mass_flow_rate")),
        initial_value=des.initial_MFR,
    )

    turb_ctrl = Control(
        tag="Turbine Pressure Ratio",
        state_path=DataPath(("energy", "turbine_PR")),
        initial_value=des.initial_turb_PR,
    )

    thrust_res = Residual(
        tag="Design Thrust",
        get_value=lambda s: s.energy.outputs.residual.thrust
    )

    work_res = Residual(
        tag="Work",
        get_value=lambda s: s.energy.outputs.residual.work
    )

    design_analysis = ResidualAnalysis(
        tag="Turbojet Design",
        analyze=base_analysis,
        controls=(mass_ctrl, turb_ctrl),
        residuals=(thrust_res, work_res),
        solver=Broyden
    )

    des_state, des_system, des_settings = design_analysis(des_state, des_system, des_settings)
    
    return des_state, des_system, settings

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

    W = Control(
        tag="Mass Flow Rate",
        state_path=DataPath(("energy", "mass_flow_rate")),
    )

    FAR = Control(
        tag="Fuel Air Ratio",
        state_path=DataPath(("energy", "fuel_air_ratio"))
    )


    return Rline, turb_PR, N, W, FAR

def _TurbojetResiduals():

    d_Wc = Residual(
        tag="Mass Flow Balance",
        get_value=lambda s: s.energy.outputs.residual.Wc
    )

    d_Wp = Residual(
        tag="Mass Flow Balance",
        get_value=lambda s: s.energy.outputs.residual.Wp
    )

    d_mdot = Residual(
        tag="Mass Flow Rate",
        get_value=lambda s: s.energy.outputs.residual.mass_flow_rate
    )
    
    # Work residual, mostly depended on turbine PR
    d_work = Residual(
        tag="Work",
        get_value=lambda s: s.energy.outputs.residual.work
    )
    
    # Thrust residual, mostly dependent on N
    d_thrust = Residual(
        tag="Thrust",
        get_value=lambda s: s.energy.outputs.residual.thrust
    )

    return d_Wc, d_Wp, d_mdot, d_work, d_thrust


def TurbojetPerformance(network: TurbojetEnergyNetwork):

    return ResidualAnalysis(
        tag="Turbojet Performance",
        analyze=build_analysis_from_network(network),
        controls=_TurbojetControls(),
        residuals=_TurbojetResiduals()
    )


    
