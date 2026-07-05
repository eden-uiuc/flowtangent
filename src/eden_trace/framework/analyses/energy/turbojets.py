# Trace/Framework/analyses/energy/turbojets.py
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
    from src.eden_trace.library.components.energy.networks import TurbojetEnergyNetwork, TurbojetDesign
    from src.eden_trace.library.components.energy.maps import CompressorMap, TurbineMap

import jax.numpy as jnp
import equinox as eqx
from jaxopt import LevenbergMarquardt, GaussNewton

from ..residual import ResidualAnalysis
from .graph_network import build_analysis_from_network

from src.eden_trace.utils import DataPath

from src.eden_trace.library import units

from src.eden_trace.framework import State, Aircraft, Settings
from src.eden_trace.framework.settings import EnergyAnalysisSettings
from src.eden_trace.framework.analyses.residual import ResidualAnalysis
from src.eden_trace.framework.conditions.controls import Control, Residual

from src.eden_trace.framework.missions.initialize import initialize_energy
from src.eden_trace.framework.missions.update import update_freestream

# ----------------------------------------------------------------------------------------------------------------------
#  Design Point Turbojet Analysis
# ----------------------------------------------------------------------------------------------------------------------

def design_turbojet(state: State, system: Aircraft, settings: Settings) -> tuple[State, Aircraft, Settings]:

    # Setup test state according to design parameters

    network: TurbojetEnergyNetwork = system.energy
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

    base_analysis = build_analysis_from_network(des_system.energy)

    mass_ctrl = Control(
        tag="Mass Flow Rate",
        state_path=DataPath(("energy", "mass_flow_rate")),
        initial_value=des.initial_MFR,
        bounds=(
            1e-3 * units.kg / units.s,
            5e3  * units.kg / units.s,
        ),
    )

    turb_ctrl = Control(
        tag="Turbine Pressure Ratio",
        state_path=DataPath(("energy", "turbine_PR")),
        initial_value=des.initial_turb_PR,
        bounds=(1.001, 1e2),
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
        solver=LevenbergMarquardt
    )

    des_state, des_system, des_settings = design_analysis(des_state, des_system, des_settings)
    des_net = des_system.energy.sync_and_clear_nodes()
    des_system = des_system.replace_subcomponent(des_net)
    
    return des_state, des_system, settings

# ----------------------------------------------------------------------------------------------------------------------
#  Off-Design Turbojet Analysis
# ----------------------------------------------------------------------------------------------------------------------

def _TurbojetControls(
    initial_Rline: float | jnp.ndarray = 2.0,
    R_line_bnds: tuple = (1.0, 2.6),
    initial_turb_PR: float | jnp.ndarray = 5.0,
    turb_PR_bnds: tuple = (1.0, 2.6),
    initial_RPM: float | jnp.ndarray = 10_000 * units.rev / units.mins,
    RPM_bnds: tuple = (1.0, 2.6),
    initial_MFR: float | jnp.ndarray = 50 * units.kg / units.s,
    MFR_bnds: tuple = (10., 100.),
    initial_FAR: float | jnp.ndarray = 1e-2,
    FAR_bnds = (1e-4, 5e-2)
):

    Rline = Control(
        tag="Rline",
        state_path=DataPath(("energy", "Rline")),
        initial_value=initial_Rline,
        bounds=R_line_bnds
    )

    turb_PR = Control(
        tag="Turbine Pressure Ratio",
        state_path=DataPath(("energy", "turbine_PR")),
        initial_value=initial_turb_PR,
        bounds=turb_PR_bnds
    )

    N = Control(
        tag="Rotation Speed",
        state_path=DataPath(("energy", "rotation_speed")),
        initial_value=initial_RPM,
        bounds=RPM_bnds,
    )

    W = Control(
        tag="Mass Flow Rate",
        state_path=DataPath(("energy", "mass_flow_rate")),
        initial_value=initial_MFR,
        bounds=MFR_bnds,
    )

    FAR = Control(
        tag="Fuel Air Ratio",
        state_path=DataPath(("energy", "fuel_air_ratio")),
        initial_value=initial_FAR,
        bounds=FAR_bnds
    )

    return (
        Rline, 
        turb_PR,
        N,
        W,
        FAR
    )

def _TurbojetResiduals():

    d_Wc = Residual(
        tag="Compressor Mass Flow",
        get_value=lambda s: s.energy.outputs.residual.Wc
    )

    d_Wp = Residual(
        tag="Turbine Mass Flow",
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

    return (
        d_Wc,
        d_Wp,
        d_mdot,
        d_work,
        d_thrust
    )


def TurbojetPerformance(
        network: TurbojetEnergyNetwork,
        initial_Rline: float | jnp.ndarray = 2.0,
        initial_turb_PR: float | jnp.ndarray = 5.0,
        initial_RPM: float | jnp.ndarray = 10_000 * units.rev / units.mins,
        initial_MFR: float | jnp.ndarray = 100 * units.kg / units.s,
        initial_FAR: float | jnp.ndarray = 1e-2,
    ):

    # Compressor Map Bounds ----------------------------------------------------

    comp =  network.line.engine.compressor
    c_des = comp.design_parameters
    c_map: CompressorMap = comp.map
    
    Nc_bnds = (min(c_map.Nc_grid).item() * c_des.rotation_speed * 0.5,
               max(c_map.Nc_grid).item() * c_des.rotation_speed * 1.5)
    
    R_bnds = (min(c_map.Rline_grid).item() * 0.5,
              max(c_map.Rline_grid).item() * 1.5)
    
    Wc_bnds = (jnp.min(c_map.Wc_table).item() * c_map.s_Wc * 0.5,
               jnp.max(c_map.Wc_table).item() * c_map.s_Wc * 1.5)
    
    # Turbine Map Bounds -------------------------------------------------------

    turb =  network.line.engine.turbine
    t_des = turb.design_parameters
    t_map: TurbineMap = turb.map

    Np_bnds = (min(t_map.Np_grid).item() * t_des.rotation_speed / 100,
               max(t_map.Np_grid).item() * t_des.rotation_speed / 100)
    
    PR_bnds = (min(t_map.PR_grid).item() * 0.5,
               max(t_map.PR_grid).item() * 1.5)
    
    Wp_bnds = (jnp.min(t_map.Wp_table).item(),
               jnp.max(t_map.Wp_table).item(),)
    
    # Composite Bounds -------------------------------------------------------00
    
    RPM_bnds = (max(Nc_bnds[0], Np_bnds[0]) * 0.5,
                min(Nc_bnds[1], Np_bnds[1]) * 1.5)
    
    FAR_bnds = (1e-4, 5e-2)

    return ResidualAnalysis(
        tag="Turbojet Performance",
        analyze=build_analysis_from_network(network),
        controls=_TurbojetControls(
            initial_Rline=initial_Rline,
            R_line_bnds=R_bnds,
            initial_turb_PR=initial_turb_PR,
            turb_PR_bnds=PR_bnds,
            initial_RPM=initial_RPM,
            RPM_bnds=RPM_bnds,
            initial_MFR=initial_MFR,
            MFR_bnds=Wc_bnds,
            initial_FAR=initial_FAR,
            FAR_bnds=FAR_bnds,
        ),
        solver=GaussNewton,
        residuals=_TurbojetResiduals()
    )


    
