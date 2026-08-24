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
    
import jax.numpy as jnp
import equinox as eqx
import optimistix as optx

from ..residual import ResidualAnalysis
from .graph_network import build_analysis_from_network

from eden_trace.utils import DataPath, init_field

from eden_trace.library import units
from eden_trace.library.components.energy.jets.classes import VariableNozzle, TurbojetEngine, TurbofanEngine, TurbofanDesign, TurbojetDesign

from eden_trace.framework import State, System, Aircraft, Settings, Process
from eden_trace.framework.settings import EnergyAnalysisSettings
from eden_trace.framework.analyses.residual import ResidualAnalysis
from eden_trace.framework.conditions.controls import Control, Residual

from eden_trace.framework.simulation.initialize import initialize_energy
from eden_trace.framework.simulation.update import update_freestream

# ----------------------------------------------------------------------------------------------------------------------
#  Jet Analysis Settings
# ----------------------------------------------------------------------------------------------------------------------

class JetSettings(EnergyAnalysisSettings):

    design_mode: bool = init_field(False, static=True)
    statics: bool = init_field(False, static=True)

# ----------------------------------------------------------------------------------------------------------------------
#  Design Point Analysis
# ----------------------------------------------------------------------------------------------------------------------

def _design_update(state: State, system: Aircraft, settings: Settings) -> tuple[State, System, Settings, Process]:

    network: TurbojetNetwork | TurbofanNetwork = system.energy
    engine: TurbojetEngine = network.line.engine
    des: TurbojetDesign | TurbofanDesign = engine.design_parameters

    # State Setup --------------------------------------------------------------

    alt = des.altitude
    M0 = des.mach_number

    atmo = state.freestream.atmosphere
    a0 = atmo.compute_speed_of_sound(alt).squeeze()

    des_state = eqx.tree_at(
        lambda s: (
            s.frames.inertial.position_vector,
            s.freestream.mach_number,
            s.frames.inertial.velocity_vector,
        ),
        state.expand_rows(1),
        (
            jnp.array([[0., 0., -alt]]),
            jnp.atleast_2d(M0),
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
    des_e_settings = JetSettings(design_mode=True)
    des_settings = eqx.tree_at(lambda s: s.analysis.energy, settings, des_e_settings)
    des_state, des_system, des_settings = initialize_energy(des_state, des_system, des_settings)
    des_state, des_system, des_settings = update_freestream(des_state, des_system, des_settings)
    base_analysis = build_analysis_from_network(des_system.energy)

    return des_state, des_system, des_settings, base_analysis

def DesignTurbojet(state: State, system: Aircraft, settings: Settings) -> tuple[State, Aircraft, Settings]:

    # Setup test state according to design parameters

    des_state, des_system, des_settings, base_analysis = _design_update(state, system, settings)
    des: TurbojetDesign = des_system.energy.design_parameters

    mass_ctrl = Control(
        tag="Mass Flow Rate",
        state_path=DataPath(("energy", "mass_flow_rate")),
        initial_value=des.mass_flow_rate,
        bounds=(
            1e-3 * units.kg / units.s,
            5e3  * units.kg / units.s,
        ),
    )

    turb_ctrl = Control(
        tag="Turbine Pressure Ratio",
        state_path=DataPath(("energy", "turbine_PR")),
        initial_value=des.turbine_PR,
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
    des_net = des_system.energy.sync_and_clear_nodes()
    des_system = des_system.replace_subcomponent(des_net)
    
    return des_state, des_system, settings

def DesignTurbofan(state: State, system: Aircraft, settings: Settings) -> tuple[State, Aircraft, Settings]:

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
        state_path=DataPath(("energy", "LPT_PR")),
        initial_value=des.LPT_PR,
        bounds=(1.001, 1e2),
    )

    HPT_ctrl = Control(
        tag="HPT Pressure Ratio",
        state_path=DataPath(("energy", "HPT_PR")),
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
    des_net = des_system.energy.sync_and_clear_nodes()
    des_system = des_system.replace_subcomponent(des_net)
    
    return des_state, des_system, settings

# ----------------------------------------------------------------------------------------------------------------------
#  Off-Design Turbojet Analysis
# ----------------------------------------------------------------------------------------------------------------------

def TurbojetPerformance(
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

    d_Wc = Residual(tag="Compressor Mass Flow", get_value=lambda s: s.energy.outputs.residual.Wc)

    d_Wp = Residual(tag="Turbine Mass Flow", get_value=lambda s: s.energy.outputs.residual.Wp)

    d_area = Residual(tag="Throat Area", get_value=lambda s: s.energy.outputs.residual.area)

    # Variable Setup -----------------------------------------------------------
    
    ctrls = (N, W, FAR, Rline, turb_PR)
    base_res = (d_power, d_thrust, d_Wc, d_Wp)
    
    if isinstance(network.line.engine.core_nozzle, VariableNozzle):
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


    
