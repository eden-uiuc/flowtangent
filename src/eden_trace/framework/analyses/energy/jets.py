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
    from eden_trace.library.components.energy.networks import TurbojetNetwork, TurbofanNetwork
    from eden_trace.library.components.energy.maps.classes import CompressorMap, TurbineMap

from dataclasses import replace

import jax.numpy as jnp
import equinox as eqx

from .graph_network import build_analysis_from_network

from eden_trace.utils import DataPath, init_field

from eden_trace.library import units
from eden_trace.library.components.energy.jets.classes import Nozzle, TurbojetEngine, TurbofanDesign, TurbojetDesign

from ..residual import ResidualAnalysis
from ... import State, System, Aircraft, Settings, Process, ProcessStep
from ...settings import EnergyAnalysisSettings
from ...conditions.controls import Control, Residual
from ...simulation.initialize import initialize_energy
from ...simulation.update import update_freestream

# ----------------------------------------------------------------------------------------------------------------------
#  Jet Analysis Settings
# ----------------------------------------------------------------------------------------------------------------------

class JetSettings(EnergyAnalysisSettings):

    design_mode: bool = init_field(False, static=True)
    statics: bool = init_field(False, static=True)

# ----------------------------------------------------------------------------------------------------------------------
#  Single Point Design Analysis
# ----------------------------------------------------------------------------------------------------------------------

def _design_update(state: State, system: Aircraft, settings: Settings) -> tuple[State, System, Settings, Process]:

    network: TurbojetNetwork | TurbofanNetwork = system.energy
    engine: TurbojetEngine = network.line.engine
    des: TurbojetDesign | TurbofanDesign = engine.design_parameters
    if isinstance(des, tuple):
        des = des[0]

    # State Setup --------------------------------------------------------------

    alt = des.altitude
    M0 = des.mach_number

    atmo = state.freestream.atmosphere
    a0 = atmo.compute_speed_of_sound(alt).squeeze()

    des_state = eqx.tree_at(
        lambda s: (
            s.freestream.mach_number,
            s.frames.inertial.position_vector,
            s.frames.inertial.velocity_vector,
        ),
        state.expand_rows(1),
        (
            jnp.atleast_2d(M0),
            jnp.array([[0., 0., -alt]]),
            jnp.atleast_2d(jnp.array([[a0 * M0, 0.0, 0.0]])),
        ))

    # System Setup -------------------------------------------------------------

    statics = settings.analysis.energy.statics
    if statics:
        MN_dict = des.exit_mach_numbers.as_dict()
        for node in MN_dict:
            engine = eqx.tree_at(
                lambda e: getattr(e, node).design_parameters.exit_mach_number,
                engine,
                MN_dict[node])
    
    # Approximate 20:4:3 pressure ratio stage split
    OPR = des.overall_pressure_ratio
    if isinstance(des, TurbofanDesign):
        k = (OPR / 240.0 ) ** (1.0 / 3.0)
        fan_PR = 3.0 * k
        LPC_PR = 4.0 * k
        HPC_PR = 20.0 * k
        
        des_engine = eqx.tree_at(lambda e: (
            e.inlet.design_parameters.pressure_recovery,
            e.fan.design_parameters.rotation_speed,
            e.fan.design_parameters.pressure_ratio,
            e.lpc.design_parameters.rotation_speed,
            e.lpc.design_parameters.pressure_ratio,
            e.hpc.design_parameters.rotation_speed,
            e.hpc.design_parameters.pressure_ratio,
            e.burner.design_parameters.pressure_ratio,
            e.burner.design_parameters.output_temperature,
            e.hpt.design_parameters.rotation_speed,
            e.lpt.design_parameters.rotation_speed,
        ),
        engine,(
            des.inlet_pressure_recovery,
            des.lp_rotation_speed,
            fan_PR,
            des.lp_rotation_speed,
            LPC_PR,
            des.hp_rotation_speed,
            HPC_PR,
            des.burner_pressure_ratio,
            des.turbine_intake_temperature,
            des.hp_rotation_speed,
            des.lp_rotation_speed,
        ))
    else:
        des_engine = eqx.tree_at(lambda e: (
            e.compressor.design_parameters.rotation_speed,
            e.compressor.design_parameters.pressure_ratio,
            e.burner.pressure_ratio,
            e.burner.output_temperature,
            e.turbine.design_parameters.rotation_speed,
        ),
        engine,(
            des.inlet_pressure_recovery,
            des.rotation_speed,
            OPR,
            des.burner_pressure_ratio,
            des.turbine_intake_temperature,
            des.rotation_speed,
        ))

    des_system = eqx.tree_at(
        lambda s: s.energy.line.engine,
        system,
        des_engine)

    # Intialize and build analysis
    if not isinstance(settings.analysis.energy, JetSettings):
        des_e_settings = JetSettings(design_mode=True)
        des_settings = eqx.tree_at(lambda s: s.analysis.energy, settings, des_e_settings)
    else:
        if not settings.analysis.energy.design_mode:
            raise ValueError(f"Attempted to call engine design with design mode setting set to False.")
        des_settings = settings
        if settings.analysis.energy.build_network:
            des_state, des_system, des_settings = initialize_energy(des_state, des_system, des_settings)
        
    des_state, des_system, des_settings = update_freestream(des_state, des_system, des_settings)
    base_analysis = build_analysis_from_network(des_system.energy)

    return des_state, des_system, des_settings, base_analysis

def design_turbojet(state: State, system: Aircraft, settings: Settings) -> tuple[State, Aircraft, Settings]:

    # Setup test state according to design parameters

    des_state, des_system, des_settings, base_analysis = _design_update(state, system, settings)
    des: TurbojetDesign = des_system.energy.design_parameters

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

    d_thrust = Residual(
        tag="Design Thrust",
        get_value=lambda s: s.energy.outputs.residual.thrust
    )

    d_power = Residual(
        tag="Power Imbalance",
        get_value=lambda s: s.energy.outputs.residual.power
    )

    design_analysis = ResidualAnalysis(
        tag="Turbojet Design",
        analyze=base_analysis,
        controls=(mass_ctrl, turb_ctrl),
        residuals=(d_thrust, d_power),
    )

    des_state, des_system, des_settings = design_analysis(des_state, des_system, des_settings)
    
    if des_settings.analysis.energy.clear_nodes:
        des_net = des_system.energy.sync_and_clear_nodes()
        des_system = des_system.replace_subcomponent(des_net)
    
    return des_state, des_system, settings

def design_turbofan(state: State, system: Aircraft, settings: Settings) -> tuple[State, Aircraft, Settings]:

    # Setup test state according to design parameters
    des_state, des_system, des_settings, base_analysis = _design_update(state, system, settings)

    des: TurbofanDesign = des_system.energy.line.engine.design_parameters

    # Set Design Bypass Ratio
    des_state = eqx.tree_at(
        lambda s: s.energy.bypass_ratio,
        des_state,
        des_system.energy.line.engine.design_parameters.bypass_ratio
    )

    # Controls Setup
    mass_ctrl = Control(
        tag="Mass Flow Rate",
        state_path=DataPath(("energy", "mass_flow_rate")),
        initial_value=des.mass_flow_rate,
        bounds=(
            1e-3 * units.kg / units.s,
            5e3  * units.kg / units.s,
        ),
    )

    LPT_ctrl = Control(
        tag="LPT Pressure Ratio",
        state_path=DataPath(("energy", "lpt_PR")),
        initial_value=des.LPT_PR,
        bounds=(1.001, 1e2),
    )

    HPT_ctrl = Control(
        tag="HPT Pressure Ratio",
        state_path=DataPath(("energy", "hpt_PR")),
        initial_value=des.HPT_PR,
        bounds=(1.001, 1e2),
    )

    # Residuals Setup
    d_thrust = Residual(
        tag="Design Thrust",
        get_value=lambda s: s.energy.outputs.residual.thrust
    )

    d_LP_power = Residual(
        tag="LP Power Imbalance",
        get_value=lambda s: s.energy.nodes['network.line.engine.lp_shaft'].outputs.residual.power
    )

    d_HP_power = Residual(
        tag="HP Power Imbalance",
        get_value=lambda s: s.energy.nodes['network.line.engine.hp_shaft'].outputs.residual.power
    )

    design_analysis = ResidualAnalysis(
        tag=f"Turbofan Design",
        analyze=base_analysis,
        controls=(mass_ctrl, LPT_ctrl, HPT_ctrl),
        residuals=(d_thrust, d_LP_power, d_HP_power)
    )

    # Run design analysis and update paramters
    des_state, des_system, des_settings = design_analysis(des_state, des_system, des_settings)
    if des_settings.analysis.energy.clear_nodes:
        des_net = des_system.energy.sync_and_clear_nodes()
        des_system = des_system.replace_subcomponent(des_net)
    
    return des_state, des_system, settings

# ----------------------------------------------------------------------------------------------------------------------
#  Off-Design Performance Analysis
# ----------------------------------------------------------------------------------------------------------------------

def turbojet_performance(
        network: TurbojetNetwork,
        initial_Rline: float | jnp.ndarray = 2.0,
        initial_turb_PR: float | jnp.ndarray = 5.0,
        initial_RPM: float | jnp.ndarray = 10_000 * units.rev / units.mins,
        initial_MFR: float | jnp.ndarray = 100 * units.kg / units.s,
        initial_FAR: float | jnp.ndarray = 1e-2,
    ):

    # Compressor Map Bounds ----------------------------------------------------

    comp =  network.line.engine.compressor
    c_map: CompressorMap = comp.map
    
    Nc_bnds = (min(c_map.Nc_grid).item() * c_map.s_Nc * 0.5,
               max(c_map.Nc_grid).item() * c_map.s_Nc * 1.5)
    
    R_bnds = (min(c_map.Rline_grid).item() * 0.5,
              max(c_map.Rline_grid).item() * 1.5)
    
    Wc_bnds = (jnp.min(c_map.Wc_table).item() * c_map.s_Wc * 0.5,
               jnp.max(c_map.Wc_table).item() * c_map.s_Wc * 1.5)
    
    # Turbine Map Bounds -------------------------------------------------------

    turb =  network.line.engine.turbine
    t_map: TurbineMap = turb.map

    
    PR_bnds = (min(t_map.PR_grid).item() * 0.5,
               max(t_map.PR_grid).item() * 1.5)
    
    Wp_bnds = (jnp.min(t_map.Wp_table).item(),
               jnp.max(t_map.Wp_table).item(),)
    
    # Composite Bounds ---------------------------------------------------------
    
    FAR_bnds = (1e-4, 0.03)

    # Control Setup -----------------------------------------------------------
    
    Rline = Control(
        tag="Rline",
        state_path=DataPath(("energy", "Rline")),
        initial_value=initial_Rline,
        bounds=R_bnds,
        scaling='logistic'
    )

    turb_PR = Control(
        tag="Turbine Pressure Ratio",
        state_path=DataPath(("energy", "turbine_PR")),
        initial_value=initial_turb_PR,
        bounds=PR_bnds,
        scaling='logistic'
    )

    N = Control(
        tag="Rotation Speed",
        state_path=DataPath(("energy", "rotation_speed")),
        initial_value=initial_RPM,
        bounds=Nc_bnds,
        scaling='logistic'
    )

    W = Control(
        tag="Mass Flow Rate",
        state_path=DataPath(("energy", "mass_flow_rate")),
        initial_value=initial_MFR,
        bounds=Wc_bnds,
        scaling='logistic'
    )

    FAR = Control(
        tag="Fuel Air Ratio",
        state_path=DataPath(("energy", "fuel_air_ratio")),
        initial_value=initial_FAR,
        bounds=FAR_bnds,
        scaling='logistic'
    )
    
    # Residual Setup -----------------------------------------------------------

    d_m_nozz = Residual(tag="Mass Flow Rate", get_value=lambda s: s.energy.outputs.residual.mass_flow_rate)
    
    d_power = Residual(tag="Power Imbalance", get_value=lambda s: s.energy.outputs.residual.power)
    
    d_thrust = Residual(tag="Thrust", get_value=lambda s: s.energy.outputs.residual.thrust)

    d_Wc = Residual(tag="Compressor Mass Flow", get_value=lambda s: s.energy.outputs.residual.compressor_Wc)

    d_Wp = Residual(tag="Turbine Mass Flow", get_value=lambda s: s.energy.outputs.residual.turbine_Wp)

    d_area = Residual(tag="Throat Area", get_value=lambda s: s.energy.outputs.residual.area)

    # Variable Setup -----------------------------------------------------------
    
    ctrls = (N, W, FAR, Rline, turb_PR)
    base_res = (d_power, d_thrust, d_Wc, d_Wp)
    
    if isinstance(network.line.engine.core_nozzle, Nozzle):
        res = base_res + (d_area,)
    else:    
        res = base_res + (d_m_nozz,)

    # Construct Analysis -------------------------------------------------------

    return ResidualAnalysis(
        tag="Turbojet Performance",
        analyze=build_analysis_from_network(network),
        controls=ctrls,
        residuals=res
    )

def turbofan_performance(
        network: TurbofanNetwork,
        OD_points: tuple[TurbofanDesign]
    ):

    # Fan Map Bounds -----------------------------------------------------------

    fan =  network.line.engine.fan
    fan_map: CompressorMap = fan.map   
    
    fan_R_bnds = (min(fan_map.Rline_grid).item() * 0.5,
                  max(fan_map.Rline_grid).item() * 1.5)

    # LPC Map Bounds -----------------------------------------------------------

    lpc =  network.line.engine.lpc
    lpc_map: CompressorMap = lpc.map
    
    lpc_Nc_bnds = (min(lpc_map.Nc_grid).item() * lpc_map.s_Nc * 0.5,
                   max(lpc_map.Nc_grid).item() * lpc_map.s_Nc * 1.5)
    
    lpc_R_bnds = (min(lpc_map.Rline_grid).item() * 0.5,
                  max(lpc_map.Rline_grid).item() * 1.5)
    
    lpc_Wc_bnds = (jnp.min(lpc_map.Wc_table).item() * lpc_map.s_Wc * 0.5,
                   jnp.max(lpc_map.Wc_table).item() * lpc_map.s_Wc * 1.5)
    
    # HPC Map Bounds -----------------------------------------------------------

    hpc =  network.line.engine.hpc
    hpc_map: CompressorMap = hpc.map
    
    hpc_Nc_bnds = (min(hpc_map.Nc_grid).item() * hpc_map.s_Nc * 0.5,
                   max(hpc_map.Nc_grid).item() * hpc_map.s_Nc * 1.5)
    
    hpc_R_bnds = (min(hpc_map.Rline_grid).item() * 0.5,
                  max(hpc_map.Rline_grid).item() * 1.5)
    
    # HPT Map Bounds -----------------------------------------------------------

    hpt =  network.line.engine.hpt
    hpt_map: TurbineMap = hpt.map

    hpt_PR_bnds = (min(hpt_map.PR_grid).item() * 0.5,
                   max(hpt_map.PR_grid).item() * 1.5)
    
    # LPT Map Bounds -----------------------------------------------------------

    lpt =  network.line.engine.lpt
    lpt_map: TurbineMap = hpt.map

    lpt_PR_bnds = (min(lpt_map.PR_grid).item() * 0.5,
                   max(lpt_map.PR_grid).item() * 1.5)

    
    def OD_array(attr_name: str):
        return jnp.array([getattr(d, attr_name) for d in OD_points]).reshape((-1, 1))

    # Control Setup -----------------------------------------------------------
    
    n_OD = len(OD_points)

    FAN_Rline = Control(
        tag="Fan Rline",
        state_path=DataPath(("energy", "fan_Rline")),
        initial_value=jnp.array([fan_map.Rline_des] * n_OD).reshape((-1, 1)),
        bounds=fan_R_bnds,
    )

    LP_Rline = Control(
        tag="LPC Rline",
        state_path=DataPath(("energy", "lpc_Rline")),
        initial_value=jnp.array([lpc_map.Rline_des] * n_OD).reshape((-1, 1)),
        bounds=lpc_R_bnds,
    )

    HP_Rline = Control(
        tag="HPC Rline",
        state_path=DataPath(("energy", "hpc_Rline")),
        initial_value=jnp.array([hpc_map.Rline_des] * n_OD).reshape((-1, 1)),
        bounds=hpc_R_bnds,
    )

    HPT_PR = Control(
        tag="HPT Pressure Ratio",
        state_path=DataPath(("energy", "hpt_PR")),
        initial_value=OD_array("HPT_PR"),
        bounds=hpt_PR_bnds,
    )

    LPT_PR = Control(
        tag="LPT Pressure Ratio",
        state_path=DataPath(("energy", "lpt_PR")),
        initial_value=OD_array("LPT_PR"),
        bounds=lpt_PR_bnds,
    )

    LPN = Control(
        tag="LP Rotation Speed",
        state_path=DataPath(("energy", "LP_speed")),
        initial_value=OD_array("lp_rotation_speed"),
        # bounds=lpc_Nc_bnds,
        scaling='linear'
    )

    HPN = Control(
        tag="HP Rotation Speed",
        state_path=DataPath(("energy", "HP_speed")),
        initial_value=OD_array("hp_rotation_speed"),
        # bounds=hpc_Nc_bnds,
        scaling='linear'
    )

    W = Control(
        tag="Mass Flow Rate",
        state_path=DataPath(("energy", "mass_flow_rate")),
        initial_value=OD_array("mass_flow_rate"),
        # bounds=lpc_Wc_bnds,
        scaling='linear'
    )

    FAR = Control(
        tag="Fuel Air Ratio",
        state_path=DataPath(("energy", "fuel_air_ratio")),
        initial_value=jnp.atleast_2d(0.01),
        bounds=(1e-4, 0.03),
    )

    BPR = Control(
        tag="Bypass Ratio",
        state_path=DataPath(("energy", "bypass_ratio")),
        initial_value=OD_array("bypass_ratio"),
        bounds=(1.0, 20.0),
    )
    
    # Residual Setup -----------------------------------------------------------    
    d_fWc = Residual(tag="Fan Mass Flow", get_value=lambda s: s.energy.outputs.residual.fan_Wc)
    d_lWc = Residual(tag="LPC Mass Flow", get_value=lambda s: s.energy.outputs.residual.lpc_Wc)
    d_hWc = Residual(tag="HPC Mass Flow", get_value=lambda s: s.energy.outputs.residual.hpc_Wc)
    
    d_lWp = Residual(tag="LPT Mass Flow", get_value=lambda s: s.energy.outputs.residual.lpt_Wp)
    d_hWp = Residual(tag="HPT Mass Flow", get_value=lambda s: s.energy.outputs.residual.hpt_Wp)

    d_thrust = Residual(tag="Thrust",     get_value=lambda s: s.energy.outputs.residual.thrust)

    d_LP_power = Residual(
        tag="LP Power Imbalance",
        get_value=lambda s: s.energy.nodes['network.line.engine.lp_shaft'].outputs.residual.power)

    d_HP_power = Residual(
        tag="HP Power Imbalance",
        get_value=lambda s: s.energy.nodes['network.line.engine.hp_shaft'].outputs.residual.power)
    
    d_W_core = Residual(
        tag="Core MFR",
        get_value=lambda s: s.energy.nodes['network.line.engine.core_nozzle'].outputs.residual.mass_flow_rate)
    d_W_byp = Residual(
        tag="Bypass MFR",
        get_value=lambda s: s.energy.nodes['network.line.engine.fan_nozzle'].outputs.residual.mass_flow_rate)

    # Variable Setup -----------------------------------------------------------
    
    ctrls = (
        FAN_Rline, LP_Rline, HP_Rline,
        HPT_PR, LPT_PR,
        HPN, LPN,
        W, FAR, BPR
    )

    res = (
        d_fWc, d_lWc, d_hWc,
        d_lWp, d_hWp,
        d_thrust,
        d_LP_power, d_HP_power,
        d_W_core, d_W_byp,
    )

    # Construct Analysis -------------------------------------------------------

    return ResidualAnalysis(
        tag="Turbofan Performance",
        analyze=build_analysis_from_network(network),
        controls=ctrls,
        residuals=res
    )
# ----------------------------------------------------------------------------------------------------------------------
#  Multi-Point Design Analysis
# ----------------------------------------------------------------------------------------------------------------------

def _design_update_OD(state: State, system: Aircraft, settings: Settings) -> tuple[State, State, System, Settings, Process]:

    engine = system.energy.line.engine
    design_points = engine.design_parameters
    assert(len(design_points) > 1)

    # Design Point Setup
    analysis_settings = replace(settings.analysis.energy, design_mode=True)
    updated_settings = eqx.tree_at(lambda s: s.analysis.energy, settings, analysis_settings)

    des_state, des_system, des_settings, _ = _design_update(state, system, updated_settings)
    des_system = eqx.tree_at(lambda s: s.energy.line.engine.design_parameters, des_system, design_points[0])
    
    # Off-Design Setup
    OD_points = design_points[1:]
    n_OD = len(OD_points)

    alt     = jnp.array([-d.altitude for d in OD_points]).reshape((-1, 1))
    M0      = jnp.array([d.mach_number for d in OD_points]).reshape((-1, 1))
    F_tgt   = jnp.array([d.thrust for d in OD_points]).reshape((-1, 1))

    atmo = des_state.freestream.atmosphere
    a0 = atmo.compute_speed_of_sound(alt)

    OD_state = eqx.tree_at(
        lambda s: (
            s.freestream.mach_number,
            s.freestream.altitude,
            s.frames.inertial.position_vector,
            s.frames.inertial.velocity_vector,
            s.energy.target_thrust,
        ),
        des_state.expand_rows(n_OD),
        (
            M0,
            -alt,
            jnp.zeros((n_OD, 3)).at[:,-1].set(alt.reshape(-1)),
            jnp.zeros((n_OD, 3)).at[:,0].set((a0 * M0).reshape(-1)),
            F_tgt,
        ))
    
    OD_analysis = eqx.tree_at(
        lambda t:t.tag,
        turbofan_performance(des_system.energy, OD_points),
        "Off-Design Analysis"
    )
    
    return des_state, OD_state, des_system, des_settings, OD_analysis


def design_turbofan_mp(state: State, system: Aircraft, settings: Settings) -> tuple[State, Aircraft, Settings]:

    des_state, OD_state, des_system, des_settings, OD_analysis = _design_update_OD(state, system, settings)
    des_settings = eqx.tree_at(lambda s: (
        s.analysis.energy.build_network,
        # s.analysis.energy.clear_nodes,
    ), 
        des_settings,
        (
            False,
            # False
        )
    )

    engine = system.energy.line.engine
    design_points = engine.design_parameters
    assert(len(design_points)>1), "Multipoint turbofan design called with only one design point specified."

    design_guess: TurbofanDesign = design_points[0]
    OD_points = design_points[1:]

    F_ctrl = Control(
        tag="Design Thrust",
        state_path=DataPath(("energy", "target_thrust")),
        initial_value=design_guess.thrust,
        bounds=(1.0, 1e6)
    )
    T_ctrl = Control(
        tag="Design TIT",
        state_path=DataPath(("energy", "target_temperature")),
        initial_value=design_guess.turbine_intake_temperature,
        bounds=(1.0, 3e3))

    d_F = Residual(
        "Off-Design Thrust",
        get_value=lambda s: s.energy.outputs.residual.thrust)

    OD_TSFC = jnp.array([d.thrust for d in OD_points]).reshape((-1, 1))
    d_TSFC = Residual(
        tag="Off-Design TSFC",
        get_value=lambda s: jnp.where(OD_TSFC, (s.energy.nodes['network.line.engine'].outputs.fuel.TSFC - OD_TSFC)/OD_TSFC, OD_TSFC))
    
    def state_swap(swap_state, swap_system, swap_settings):
        
        updated_state, updated_system, updated_settings = initialize_energy(OD_state, swap_system, swap_settings)
        updated_system = updated_system.update_network_topology()
        
        updated_state = eqx.tree_at(lambda s:(
                s.energy.target_thrust,
                s.energy.target_temperature,
                s.time.N
            ),
            updated_state,
            (
                OD_state.energy.target_thrust,
                OD_state.energy.target_temperature,
                len(OD_points)))
        
        updated_settings = eqx.tree_at(
            lambda s: s.analysis.energy,
            updated_settings,
            JetSettings(statics=updated_settings.analysis.energy.statics) # Turn off design mode by instatiating new settings
        )
        
        return updated_state, updated_system, updated_settings

    MP_inner_loop = Process(
        tag="Multi-Point Turbofan Analysis",
        steps=(
            ProcessStep(tag="Initial Design", function=design_turbofan),
            ProcessStep(tag="State Swap", function=state_swap),
            OD_analysis
        )
    )

    MP_outer_loop = ResidualAnalysis(
        tag="Multi-Point Turbofan Design",
        analyze=MP_inner_loop,
        controls=(F_ctrl, T_ctrl),
        residuals=(d_F, d_TSFC)
    )

    final_state, final_system, final_settings = MP_outer_loop(des_state, des_system, des_settings)
    final_net = final_system.energy.sync_and_clear_nodes()
    final_system = final_system.replace_subcomponent(final_net)

    final_settings = eqx.tree_at(lambda s: s.analysis.energy.design_mode, final_settings, False)

    return final_state, final_system, final_settings
