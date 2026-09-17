# flowtangent/Framework/analyses/energy/turbojets.py
# (c) Copyright 2026 Aerospace Research Community LLC
#
# Created: Jun 2026, J Smart
# Modified: Jun 2026, J Smart

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from typing import TYPE_CHECKING

# --- Framework Imports (Strictly for Type Hinting to avoid Circular Imports) ---
if TYPE_CHECKING:
    from ... import Aircraft, Process, ProcessStep, Settings, State, System
    from ...components.energy.jets import TurbofanNetwork, TurbojetNetwork
    from ...components.energy.maps._classes import CompressorMap, TurbineMap

from dataclasses import replace

import jax.numpy as jnp

from flowtangent.data import units
from flowtangent.utils import TreePath, field

from ...components.energy.jets._classes import TurbofanDesign, TurbojetEngine, TurbojetOpPoint
from ...sim.initialize import initialize_energy
from ...sim.update import update_freestream
from ...utils import update
from .._batched import BatchedAnalysis
from .._implicit import ImplicitAnalysis, Residual, Variable
from .._settings import EnergyAnalysisSettings
from ._energy_network import PACTAnalysis

# ----------------------------------------------------------------------------------------------------------------------
#  API Setup
# ----------------------------------------------------------------------------------------------------------------------

__all__ = [
    "JetSettings",
    "build_turbofan_design",
    "build_turbojet_design",
    "build_turbojet_performance",
    "build_turbofan_performance",
]

# ----------------------------------------------------------------------------------------------------------------------
#  Jet Analysis Settings
# ----------------------------------------------------------------------------------------------------------------------


class JetSettings(EnergyAnalysisSettings):
    design_mode: bool = field(False, static=True)
    statics: bool = field(False, static=True)


# ----------------------------------------------------------------------------------------------------------------------
#  Single Point Design Analysis
# ----------------------------------------------------------------------------------------------------------------------


def _design_update(state: State, system: System, settings: Settings) -> tuple[State, System, Settings, Process]:

    network: TurbojetNetwork | TurbofanNetwork = system.energy
    engine: TurbojetEngine = network.line.engine
    des: TurbojetOpPoint | TurbofanDesign = engine.design_parameters
    if isinstance(des, tuple):
        des = des[0]

    # State Setup --------------------------------------------------------------

    alt = des.altitude
    M0 = des.mach_number

    atmo = state.freestream.atmosphere
    a0 = atmo.compute_speed_of_sound(alt).squeeze()

    des_state = update(
        state.expand_time(1),
        (
            ("freestream.mach_number", jnp.atleast_2d(M0)),
            ("frames.inertial.position_vector", jnp.array([[0.0, 0.0, -alt]])),
            ("frames.inertial.velocity_vector", jnp.atleast_2d(jnp.array([[a0 * M0, 0.0, 0.0]]))),
        ),
    )

    # System Setup -------------------------------------------------------------

    statics = settings.analysis.energy.statics
    if statics:
        MN_dict = vars(des.station_mach_numbers)
        _ = MN_dict.pop("name", None)
        for node in MN_dict:
            engine = update(engine, lambda e: getattr(e, node).design_parameters.exit_mach_number, MN_dict[node])

    # Approximate 20:4:3 pressure ratio stage split
    OPR = des.overall_pressure_ratio
    if isinstance(des, TurbofanDesign):
        k = (OPR / 240.0) ** (1.0 / 3.0)
        fan_PR = 3.0 * k
        LPC_PR = 4.0 * k
        HPC_PR = 20.0 * k

        des_engine = update(
            engine,
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
            engine,
            (
                ("compressor.design_parameters.rotation_speed", des.rotation_speed),
                ("compressor.design_parameters.pressure_ratio", OPR),
                ("burner.design_parameters.pressure_ratio", des.burner_pressure_ratio),
                ("burner.design_parameters.output_temperature", des.turbine_intake_temperature),
                ("turbine.design_parameters.rotation_speed", des.rotation_speed),
                ("turbine.design_parameters.pressure_ratio", des.turbine_PR),
            ),
        )

    des_system = update(system, "energy.line.engine", des_engine)

    # Intialize and build analysis
    if not isinstance(settings.analysis.energy, JetSettings):
        des_e_settings = JetSettings(design_mode=True)
        des_settings = update(settings, "analysis.energy", des_e_settings)
    else:
        if not settings.analysis.energy.design_mode:
            raise ValueError("Attempted to call engine design with design mode setting set to False.")
        des_settings = settings
        if settings.analysis.energy.build_network:
            des_state, des_system, des_settings = initialize_energy(des_state, des_system, des_settings)

    des_state = update(
        des_state,
        (
            ("energy.target_thrust", jnp.atleast_2d(des.thrust)),
            ("energy.target_temperature", jnp.atleast_2d(des.turbine_intake_temperature)),
        ),
    )

    des_state, des_system, des_settings = update_freestream(des_state, des_system, des_settings)
    base_analysis = PACTAnalysis(des_system.energy)

    return des_state, des_system, des_settings, base_analysis


def build_turbojet_design(state: State, system: System, settings: Settings) -> tuple[State, System, Settings, ImplicitAnalysis]:

    # Setup test state according to design parameters

    des_state, des_system, des_settings, base_analysis = _design_update(state, system, settings)
    des: TurbojetOpPoint = des_system.energy.line.engine.design_parameters

    mass_var = Variable(
        name="Mass Flow Rate",
        state_path=TreePath(("energy", "mass_flow_rate")),
        initial_value=des.mass_flow_rate,
        bounds=(
            1e-3 * units.kg / units.s,
            5e3 * units.kg / units.s,
        ),
    )

    turb_var = Variable(
        name="Turbine Pressure Ratio",
        state_path=TreePath(("energy", "turbine_PR")),
        initial_value=des.turbine_PR,
        bounds=(1.001, 1e2),
    )

    d_thrust = Residual(name="Design Thrust", state_path="energy.residual.thrust")
    d_power = Residual(name="Power Imbalance", state_path="energy.residual.power")

    design_analysis = ImplicitAnalysis(
        name="Turbojet Design",
        analyze=base_analysis,
        variables=(mass_var, turb_var),
        residuals=(d_thrust, d_power),
    )

    return des_state, des_system, des_settings, design_analysis


def build_turbofan_design(state: State, system: Aircraft, settings: Settings) -> ImplicitAnalysis:

    # Setup test state according to design parameters
    _, des_system, _, base_analysis = _design_update(state, system, settings)

    des: TurbofanDesign = des_system.energy.line.engine.design_parameters

    # Variables Setup
    mass_var = Variable(
        name="Mass Flow Rate",
        state_path=TreePath(("energy", "mass_flow_rate")),
        initial_value=des.mass_flow_rate,
        bounds=(
            1e-3 * units.kg / units.s,
            5e3 * units.kg / units.s,
        ),
    )

    LPT_var = Variable(
        name="LPT Pressure Ratio",
        state_path=TreePath(("energy", "lpt_PR")),
        initial_value=des.LPT_PR,
        bounds=(1.001, 1e2),
    )

    HPT_var = Variable(
        name="HPT Pressure Ratio",
        state_path=TreePath(("energy", "hpt_PR")),
        initial_value=des.HPT_PR,
        bounds=(1.001, 1e2),
    )

    # Residuals Setup
    d_thrust = Residual(name="Design Thrust", state_path="energy.residual.thrust")

    d_LP_power = Residual(
        name="LP Power Imbalance", value_func=lambda s: s.energy.nodes["network.line.engine.lp_shaft"].residual.power
    )

    d_HP_power = Residual(
        name="HP Power Imbalance", value_func=lambda s: s.energy.nodes["network.line.engine.hp_shaft"].residual.power
    )

    design_analysis = ImplicitAnalysis(
        name="Turbofan Design",
        analyze=base_analysis,
        variables=(mass_var, LPT_var, HPT_var),
        residuals=(d_thrust, d_LP_power, d_HP_power),
    )

    return design_analysis


# ----------------------------------------------------------------------------------------------------------------------
#  Off-Design Performance Analysis
# ----------------------------------------------------------------------------------------------------------------------


def build_turbojet_performance(
    network: TurbojetNetwork,
    operating_parameters: TurbojetOpPoint,
):

    op = operating_parameters

    # Compressor Map Bounds ----------------------------------------------------

    comp = network.line.engine.compressor
    c_map: CompressorMap = comp.map

    R_bnds = (
        min(c_map.Rline_grid).item() * 0.5,
        max(c_map.Rline_grid).item() * 1.5,
    )

    # Turbine Map Bounds -------------------------------------------------------

    turb = network.line.engine.turbine
    t_map: TurbineMap = turb.map

    PR_bnds = (
        min(t_map.PR_grid).item() * 0.5,
        max(t_map.PR_grid).item() * 1.5,
    )

    # Composite Bounds ---------------------------------------------------------

    FAR_bnds = (1e-4, 0.03)

    # Variable Setup -----------------------------------------------------------

    Rline = Variable(
        name="Rline",
        state_path=TreePath(("energy", "compressor_Rline")),
        initial_value=op.compressor_Rline,
        bounds=R_bnds,
        scaling="log_bounded",
    )

    turb_PR = Variable(
        name="Turbine Pressure Ratio",
        state_path=TreePath(("energy", "turbine_PR")),
        initial_value=op.turbine_PR,
        bounds=PR_bnds,
        scaling="log_bounded",
    )

    N = Variable(
        name="Rotation Speed",
        state_path=TreePath(("energy", "rotation_speed")),
        initial_value=op.rotation_speed,
        bounds=(op.rotation_speed * 0.5, op.rotation_speed * 2.0),
        scaling="log_bounded",
    )

    W = Variable(
        name="Mass Flow Rate",
        state_path=TreePath(("energy", "mass_flow_rate")),
        initial_value=op.mass_flow_rate,
        bounds=(op.mass_flow_rate * 0.5, op.mass_flow_rate * 2.0),
        scaling="log_bounded",
    )

    FAR = Variable(
        name="Fuel Air Ratio",
        state_path=TreePath(("energy", "fuel_air_ratio")),
        initial_value=op.FAR,
        bounds=FAR_bnds,
        scaling="log_bounded",
    )

    # Residual Setup -----------------------------------------------------------

    d_m_nozz = Residual(name="Mass Flow Rate", state_path="energy.residual.mass_flow_rate")
    d_power = Residual(name="Power Imbalance", state_path="energy.residual.power")
    d_thrust = Residual(name="Thrust", state_path="energy.residual.thrust")
    d_Wc = Residual(name="Compressor Mass Flow", state_path="energy.residual.compressor_Wc")
    d_Wp = Residual(name="Turbine Mass Flow", state_path="energy.residual.turbine_Wp")
    d_area = Residual(name="Throat Area", state_path="energy.residual.area")

    # Variable Setup -----------------------------------------------------------

    vars = (N, W, FAR, Rline, turb_PR)
    base_res = (d_power, d_thrust, d_Wc, d_Wp)

    if network.line.engine.core_nozzle.variable_exit:
        res = base_res + (d_area,)
    else:
        res = base_res + (d_m_nozz,)

    # Construct Analysis -------------------------------------------------------

    return ImplicitAnalysis(
        name="Turbojet Performance",
        analyze=PACTAnalysis(network),
        variables=vars,
        residuals=res,
    )


def build_turbofan_performance(network: TurbofanNetwork):

    # Fan Map Bounds -----------------------------------------------------------

    fan = network.line.engine.fan
    fan_map: CompressorMap = fan.map

    fan_R_bnds = (
        min(fan_map.Rline_grid).item() * 0.5,
        max(fan_map.Rline_grid).item() * 1.5,
    )

    # LPC Map Bounds -----------------------------------------------------------

    lpc = network.line.engine.lpc
    lpc_map: CompressorMap = lpc.map

    lpc_R_bnds = (
        min(lpc_map.Rline_grid).item() * 0.5,
        max(lpc_map.Rline_grid).item() * 1.5,
    )

    # HPC Map Bounds -----------------------------------------------------------

    hpc = network.line.engine.hpc
    hpc_map: CompressorMap = hpc.map

    hpc_R_bnds = (
        min(hpc_map.Rline_grid).item() * 0.5,
        max(hpc_map.Rline_grid).item() * 1.5,
    )

    # HPT Map Bounds -----------------------------------------------------------

    hpt = network.line.engine.hpt
    hpt_map: TurbineMap = hpt.map

    hpt_PR_bnds = (
        min(hpt_map.PR_grid).item() * 0.5,
        max(hpt_map.PR_grid).item() * 1.5,
    )

    # LPT Map Bounds -----------------------------------------------------------

    lpt = network.line.engine.lpt
    lpt_map: TurbineMap = lpt.map

    lpt_PR_bnds = (
        min(lpt_map.PR_grid).item() * 0.5,
        max(lpt_map.PR_grid).item() * 1.5,
    )

    # Variable Setup -----------------------------------------------------------

    FAN_Rline = Variable(
        name="Fan Rline",
        state_path=TreePath(("energy", "fan_Rline")),
        initial_value=jnp.array([fan_map.Rline_des]).reshape((-1, 1)),
        bounds=fan_R_bnds,
    )

    LP_Rline = Variable(
        name="LPC Rline",
        state_path=TreePath(("energy", "lpc_Rline")),
        initial_value=jnp.array([lpc_map.Rline_des]).reshape((-1, 1)),
        bounds=lpc_R_bnds,
    )

    HP_Rline = Variable(
        name="HPC Rline",
        state_path=TreePath(("energy", "hpc_Rline")),
        initial_value=jnp.array([hpc_map.Rline_des]).reshape((-1, 1)),
        bounds=hpc_R_bnds,
    )

    HPT_PR = Variable(
        name="HPT Pressure Ratio",
        state_path=TreePath(("energy", "hpt_PR")),
        initial_value=network.line.engine.design_parameters.HPT_PR,
        bounds=hpt_PR_bnds,
    )

    LPT_PR = Variable(
        name="LPT Pressure Ratio",
        state_path=TreePath(("energy", "lpt_PR")),
        initial_value=network.line.engine.design_parameters.LPT_PR,
        bounds=lpt_PR_bnds,
    )

    LPN = Variable(
        name="LP Rotation Speed",
        state_path=TreePath(("energy", "LP_speed")),
        initial_value=network.line.engine.design_parameters.lp_rotation_speed,
        bounds=(1000 * units.rev / units.mins, 10000 * units.rev / units.mins),
    )

    HPN = Variable(
        name="HP Rotation Speed",
        state_path=TreePath(("energy", "HP_speed")),
        initial_value=network.line.engine.design_parameters.hp_rotation_speed,
        bounds=(3000 * units.rev / units.mins, 20000 * units.rev / units.mins),
    )

    W = Variable(
        name="Mass Flow Rate",
        state_path=TreePath(("energy", "mass_flow_rate")),
        initial_value=network.line.engine.design_parameters.mass_flow_rate,
        # bounds=lpc_Wc_bnds,
        scaling="linear",
    )

    FAR = Variable(
        name="Fuel Air Ratio",
        state_path=TreePath(("energy", "fuel_air_ratio")),
        initial_value=jnp.atleast_2d(0.01),
        bounds=(1e-4, 0.03),
    )

    BPR = Variable(
        name="Bypass Ratio",
        state_path=TreePath(("energy", "bypass_ratio")),
        initial_value=network.line.engine.design_parameters.bypass_ratio,
        bounds=(1.0, 20.0),
    )

    # Residual Setup -----------------------------------------------------------
    d_fWc = Residual(name="Fan Mass Flow", get_value=lambda s: s.energy.residual.fan_Wc)
    d_lWc = Residual(name="LPC Mass Flow", get_value=lambda s: s.energy.residual.lpc_Wc)
    d_hWc = Residual(name="HPC Mass Flow", get_value=lambda s: s.energy.residual.hpc_Wc)

    d_lWp = Residual(name="LPT Mass Flow", get_value=lambda s: s.energy.residual.lpt_Wp)
    d_hWp = Residual(name="HPT Mass Flow", get_value=lambda s: s.energy.residual.hpt_Wp)

    d_thrust = Residual(name="Thrust", get_value=lambda s: s.energy.residual.thrust)

    d_LP_power = Residual(
        name="LP Power Imbalance", get_value=lambda s: s.energy.nodes["network.line.engine.lp_shaft"].residual.power
    )

    d_HP_power = Residual(
        name="HP Power Imbalance", get_value=lambda s: s.energy.nodes["network.line.engine.hp_shaft"].residual.power
    )

    d_W_core = Residual(
        name="Core MFR", get_value=lambda s: s.energy.nodes["network.line.engine.core_nozzle"].residual.mass_flow_rate
    )
    d_W_byp = Residual(
        name="Bypass MFR", get_value=lambda s: s.energy.nodes["network.line.engine.fan_nozzle"].residual.mass_flow_rate
    )

    # Variable Setup -----------------------------------------------------------

    vars = (
        FAN_Rline,
        LP_Rline,
        HP_Rline,
        HPT_PR,
        LPT_PR,
        HPN,
        LPN,
        W,
        FAR,
        BPR,
    )

    res = (
        d_fWc,
        d_lWc,
        d_hWc,
        d_lWp,
        d_hWp,
        d_thrust,
        d_LP_power,
        d_HP_power,
        d_W_core,
        d_W_byp,
    )

    # Construct Analysis -------------------------------------------------------

    return ImplicitAnalysis(
        name="Turbofan Performance",
        analyze=PACTAnalysis(network),
        variables=vars,
        residuals=res,
    )


# ----------------------------------------------------------------------------------------------------------------------
#  Multi-Point Design Analysis
# ----------------------------------------------------------------------------------------------------------------------


def _design_update_batched(
    state: State, system: Aircraft, settings: Settings
) -> tuple[State, Aircraft, Settings, Process]:

    engine = system.energy.line.engine
    design_points = engine.design_parameters
    assert len(design_points) > 1

    # Design Point Setup
    analysis_settings = replace(settings.analysis.energy, design_mode=True)
    updated_settings = update(settings, "analysis.energy", analysis_settings)

    des_state, des_system, des_settings, _ = _design_update(state, system, updated_settings)
    des_system = update(des_system, "energy.line.engine.design_parameters", design_points[0])
    des_e_setts = replace(des_settings.analysis.energy, design_mode=True)
    des_n_setts = replace(des_settings.numerical, sum_residuals=True)
    des_settings = update(
        des_settings,
        (
            ("numerical", des_n_setts),
            ("analysis.energy", des_e_setts),
        ),
    )

    # Set Up State Inputs
    OD_points = design_points[1:]
    n_OD = len(OD_points)

    # fmt: off
    alt_val = jnp.array([d.altitude for d in OD_points]).reshape((-1, 1))
    a0_val  = des_state.freestream.atmosphere.compute_speed_of_sound(alt_val)
    M0_val  = jnp.array([d.mach_number for d in OD_points]).reshape((-1, 1))
    x_val   = -jnp.zeros((n_OD, 3)).at[:,-1].set(alt_val.reshape(-1))
    v_val   = jnp.zeros((n_OD, 3)).at[:,0].set((a0_val * M0_val).reshape(-1))
    F_val   = jnp.array([d.thrust for d in OD_points]).reshape((-1, 1))
    T_val   = jnp.array([d.turbine_intake_temperature for d in OD_points]).reshape((-1, 1))

    # State Values
    alt     = TreePath("state.freestream.altitude", value=alt_val)
    M0      = TreePath("state.freestream.mach_number", value=M0_val)
    x       = TreePath("state.frames.inertial.position_vector", value=x_val)
    v       = TreePath("state.frames.inertial.velocity_vector", value=v_val)

    # Outer Loop Variables
    F       = TreePath("state.energy.target_thrust", value=F_val)
    T       = TreePath("state.energy.target_temperature", value=T_val)
    # fmt: on

    OD_analysis = BatchedAnalysis(
        name="Off-Design Analysis",
        analyze=build_turbofan_performance(des_system.energy),
        state_inputs=(
            alt,
            M0,
            x,
            v,
            F,
            T,
        ),
    )

    return des_state, des_system, des_settings, OD_analysis


def design_turbofan_mp(state: State, system: Aircraft, settings: Settings) -> tuple[State, Aircraft, Settings]:

    # Set up Inner Loop

    des_state, des_system, des_settings, OD_analysis = _design_update_batched(state, system, settings)
    des_analysis = build_turbofan_design(des_state, des_system, des_settings)
    des_state, des_system, des_settings = des_analysis.initialize(des_state, des_system, des_settings)

    engine = system.energy.line.engine
    design_points = engine.design_parameters
    assert len(design_points) > 1, "Multipoint turbofan design called with only one design point specified."

    # Set up Outer Loop

    design_guess: TurbofanDesign = design_points[0]
    OD_points = design_points[1:]

    F_var = Variable(
        name="Design Thrust",
        state_path=TreePath(("energy", "target_thrust")),
        initial_value=design_guess.thrust,
        bounds=(1.0, 1e6),
    )
    T_var = Variable(
        name="Design TIT",
        state_path=TreePath(("energy", "target_temperature")),
        initial_value=design_guess.turbine_intake_temperature,
        bounds=(1.0, 3e3),
    )

    OD_F = jnp.array([d.thrust for d in OD_points]).reshape((-1, 1)).at[0, 0].set(0.0)
    d_F = Residual("Off-Design Thrust", get_value=lambda s: jnp.where(OD_F, s.energy.residual.thrust, OD_F))

    OD_TSFC = jnp.array([d.TSFC for d in OD_points]).reshape((-1, 1))
    d_TSFC = Residual(
        name="Off-Design TSFC",
        get_value=lambda s: jnp.where(
            OD_TSFC, (s.energy.nodes["network.line.engine"].fuel.TSFC - OD_TSFC) / OD_TSFC, OD_TSFC
        ),
    )

    def split_residuals(swap_state, swap_system, swap_settings):

        updated_settings = update(
            swap_settings,
            ("numerical", replace(swap_settings.numerical, sum_residuals=False)),
        )

        return swap_state, swap_system, updated_settings

    def design_handover(swap_state, swap_system, swap_settings):

        updated_settings = update(
            swap_settings,
            ("analysis.energy", replace(swap_settings.analysis.energy, design_mode=False)),
        )

        return swap_state, swap_system, updated_settings

    def settings_reset(swap_state, swap_system, swap_settings):
        return swap_state, swap_system, des_settings

    MP_inner_loop = Process(
        name="Multi-Point Turbofan Analysis",
        steps=(
            ProcessStep(name="Inner Residual Switch", function=split_residuals),
            des_analysis,
            ProcessStep(name="Design Handover", function=design_handover),
            OD_analysis,
            ProcessStep(name="Outer Residual Switch", function=settings_reset),
        ),
    )

    MP_outer_loop = ImplicitAnalysis(
        name="Multi-Point Turbofan Design",
        analyze=MP_inner_loop,
        variables=(F_var, T_var),
        residuals=(d_F, d_TSFC),
        # solver='hybr'
    )

    final_state, final_system, final_settings = MP_outer_loop.run(des_state, des_system, des_settings, initialize=True)
    final_net = final_system.energy.sync_and_clear_nodes()
    final_system = final_system.replace_subcomponent(final_net)

    return final_state, final_system, final_settings
