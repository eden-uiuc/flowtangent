import flowtangent as ft

import json

import jax.numpy as jnp
import equinox as eqx
import numpy as np
import pandas as pd

from pathlib import Path
from dataclasses import replace


from flowtangent.utils import save_data, load_data, format_array, configure_environment, LoggingSettings, update
from flowtangent.solve import BatchedAnalysis

from flowtangent.data import units
from flowtangent.components.energy.jets import TurbojetEngine, TurbojetOpPoint, TurbojetLine, TurbojetNetwork, JetNetParameters, JetKinematics
from flowtangent.solve.energy.jets import build_turbojet_design, build_turbojet_performance, JetSettings
from flowtangent.sim.initialize import initialize_energy
from flowtangent.sim.update import update_freestream



# Control Board
DEV = False
DEBUG = False
VERBOSE = True

DESIGN_POINT = True
OFF_DESIGN_0 = False
OFF_DESIGN_1 = True
BATCHED_OD1  = True

def system_setup():

    station_MNs = JetKinematics(
        inlet=0.6,
        compressor=0.02,
        burner=0.02,
        turbine=0.4
    )

    engine_design = TurbojetOpPoint(
        thrust=11_800 * units.lbf,
        mass_flow_rate=168.45 * units.lbm/units.s,
        rotation_speed=8070. * units.rpm,
        overall_pressure_ratio=13.5,
        turbine_PR=4.46,
        turbine_intake_temperature=2370.0 * units.R,
        station_mach_numbers=station_MNs
    )

    engine = TurbojetEngine.build_custom(variable_nozzle=True, design_parameters=engine_design)
    
    comp_design = update(engine.compressor, "design_parameters.eff.flow", 0.83)

    burn_design = update(engine.burner,
                         (
                           ("design_parameters.output_temperature", 2370 * units.R),
                           ("design_parameters.pressure_ratio", 0.97),
                         )
                    )
    turb_design = update(engine.turbine, "design_parameters.eff.flow", 0.86)

    nozz_design = update(engine.core_nozzle, "design_parameters.eff.flow", 1.0)

    nozz_design = replace(nozz_design, variable_exit=True)

    des_engine = update(engine, (
        ("compressor", comp_design),
        ("burner", burn_design),
        ("turbine", turb_design),
        ("core_nozzle", nozz_design),
        )
    )

    line = TurbojetLine(name="Line", subcomponents=(des_engine,),)
    net = TurbojetNetwork(subcomponents=(line,))
    sys = ft.Aircraft(name="Simple Turbojet System", subcomponents=(net,))

    return sys

def off_design_point(
    system: ft.Aircraft,
    settings: ft.Settings,
    op_point: TurbojetOpPoint
):

    print("="*80)
    print(f" {op_point.name} Analysis")
    print("-"*80)

    network: TurbojetNetwork = system.energy
    des: JetNetParameters = network.design_parameters

    alt = op_point.altitude
    M0 = op_point.mach_number
    thrust = op_point.thrust

    atmo = des.atmosphere
    a0 = atmo.compute_speed_of_sound(alt)

    od_state = update(
        ft.State().expand_time(),
        (
            ("frames.inertial.position_vector", jnp.array([[0., 0., -alt]])),
            ("freestream.mach_number", jnp.atleast_2d(M0)),
            ("frames.inertial.velocity_vector", jnp.atleast_2d(jnp.array([[(a0 * M0).item(), 0.0, 0.0]]))),
        ),
    )

    od_analysis = build_turbojet_performance(network, op_point)

    od_state, od_system, od_settings = initialize_energy(od_state, system, settings)
    od_state, od_system, od_settings = update_freestream(od_state, od_system, od_settings)
    od_state = update(od_state, "energy.target_thrust", jnp.atleast_2d(thrust))

    new_settings = JetSettings(design_mode=False, statics=od_settings.analysis.energy.statics)
    od_settings = update(od_settings, "analysis.energy", new_settings)
    od_state, od_system, od_settings = od_analysis.run(od_state, od_system, od_settings)

    od_thermal, od_static = validate_design_point(
            data_dir / f"turbojet_{op_point.name}.json",
            od_state,
            point_name=str(op_point.name)
        )

    od_thermal.to_csv(data_dir / f"{op_point.name}_thermal.csv")
    od_static.to_csv(data_dir / f"{op_point.name}_static.csv")

    return od_state, od_system, od_settings

def batched_off_design(
    system: ft.Aircraft,
    settings: ft.Settings,
    op_point: TurbojetOpPoint        
):

    print("="*80)
    print(f" {op_point.name} Analysis")
    print("-"*80)

    network: TurbojetNetwork = system.energy
    des: JetNetParameters = network.design_parameters

    alt = op_point.altitude
    M0 = op_point.mach_number
    thrust = op_point.thrust

    atmo = des.atmosphere
    a0 = atmo.compute_speed_of_sound(alt)

    od_state = update(
        ft.State().expand_time(),
        (
            ("frames.inertial.position_vector", jnp.array([[0., 0., -alt]])),
            ("freestream.mach_number", jnp.atleast_2d(M0)),
            ("frames.inertial.velocity_vector", jnp.atleast_2d(jnp.array([[(a0 * M0).item(), 0.0, 0.0]]))),
        ),
    )

    od_analysis = build_turbojet_performance(network, op_point)
    batched_od = BatchedAnalysis(name=f"Batched {op_point.name} Analysis", analyze=od_analysis)

    od_state, od_system, od_settings = initialize_energy(od_state, system, settings)
    od_state, od_system, od_settings = update_freestream(od_state, od_system, od_settings)
    od_state = update(od_state, "energy.target_thrust", jnp.atleast_2d(thrust))

    new_settings = JetSettings(design_mode=False, statics=od_settings.analysis.energy.statics)
    od_settings = update(od_settings, "analysis.energy", new_settings)
    od_state, od_system, od_settings = batched_od.run(od_state, od_system, od_settings)

    return od_state, od_system, od_settings

def validate_design_point(pycycle_json_path, ft_state, point_name: str="Design"):
    """
    Loads PyCycle JSON results and compares them against the Flowtangent state.
    """
    
    # 1. Load the JSON
    with open(pycycle_json_path, 'r') as f:
        pycycle_data = json.load(f)
        
    # 2. Map PyCycle flow stations to Flowtangent network IDs
    station_map = {
        'fc.Fl_O':     ft_state.freestream,
        'inlet.Fl_O':  ft_state.energy.nodes['network.line.engine.inlet'].flow,
        'comp.Fl_O':   ft_state.energy.nodes['network.line.engine.compressor'].flow,
        'burner.Fl_O': ft_state.energy.nodes['network.line.engine.burner'].flow,
        'turb.Fl_O':   ft_state.energy.nodes['network.line.engine.turbine'].flow,
        'nozz.Fl_O':   ft_state.energy.nodes['network.line.engine.core_nozzle'].flow,
    }

    station_names = {
        'fc.Fl_O':     "FS",
        'inlet.Fl_O':  "inlet".title(),
        'comp.Fl_O':   "comp.".title(),
        'burner.Fl_O': "burner".title(),
        'turb.Fl_O':   "turb.".title(),
        'nozz.Fl_O':   "nozz.".title(),
    }
    
    # 3. Map PyCycle properties to Flowtangent names
    thermal_map = {
        'W':     ('mass_flow_rate', units.lbm/units.s),
        'Pt':    ('stagnation_pressure', units.psi),
        'Tt':    ('stagnation_temperature', units.R),
        # 'ht':    ('stagnation_enthalpy', units.btu/units.lbm),
    }

    static_map = {
        'Ps':    ('pressure', units.psi),
        'Ts':    ('temperature', units.R),
        'MN':    ('mach_number', 1.0),
        # 'V':     ('speed', units.ft/units.s),
    }
    
    # 4. Assemble the Comparison
    thermal_records = []
    static_records = []

    def get_records(pyc_station, prop_map):
        node = station_map.get(pyc_station)
        records = []
        for pyc_prop, pyc_val in pyc_props.items():
                if pyc_prop in prop_map:
                    prop_name, pyc_units = prop_map[pyc_prop]
                    value = getattr(node, prop_name, None)
                    if value:
                        ft_val = np.asarray(value).item()

                        pyc_val *= pyc_units
                        diff = ft_val - pyc_val

                        if abs(pyc_val) > 1e-12:
                            rel_error = (diff / pyc_val)
                        else:
                            rel_error = np.nan if abs(ft_val) > 1e-12 else 0.0

                        records.append({
                            'Station': station_names[pyc_station],
                            'Property': pyc_prop,
                            'PyCycle Val': pyc_val,
                            'FlowTan Val': ft_val,
                            'Diff': diff,
                            'Rel. Error': rel_error,
                            'Mag. Error': np.abs(rel_error)
                        })
        return records

    
    for pyc_station, pyc_props in pycycle_data.get('flow_stations', {}).items():
        if pyc_station not in station_map:
            continue
        thermal_records.extend(get_records(pyc_station, thermal_map))
        static_records.extend(get_records(pyc_station, static_map))
            
    # Convert to DataFrame
    def make_df(records, name):

        df = pd.DataFrame(records)
        
        pd.set_option('display.max_rows', None)
        pd.set_option('display.float_format', '{:.4e}'.format)
        
        print("\n" + "="*80)
        print(f" PyCycle vs. FlowTangent {point_name} {name} Validation")
        print("-"*80)
        print(df.drop(columns='Mag. Error').to_string(index=False))
        
        print("\n"+"-"*80)
        print(" Error Magnitude Summary:")
        print(f" - Mean: {df['Mag. Error'].mean():.4e}")
        print(f" - Min:  {df['Mag. Error'].min():.4e}")
        print(f" - Max:  {df['Mag. Error'].max():.4e}")
        print("\n" + "="*80 + "\n")

        return df

    thermal_df = make_df(thermal_records, "Thermal")
    statics_df = make_df(static_records, "Static")
    return thermal_df, statics_df


if __name__ == "__main__":


    # Build Turbojet------------------------------------------------------------
    test_dir = Path(__file__).resolve().parent
    data_dir = test_dir / "ft_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    system = system_setup()
    settings = eqx.tree_at(
        lambda s: s.analysis.energy,
        ft.Settings(_DEV_MODE=DEV, DEBUG_MODE=DEBUG, verbose=VERBOSE,
                 logging=LoggingSettings(log_dir=test_dir/"ft_logs")),
        JetSettings(design_mode=DESIGN_POINT, statics=True)
    )

    configure_environment(settings)
    

    if DESIGN_POINT:
        print("="*80)
        print(" Design Point Analysis")
        print("-"*80)

        des_st, des_sys, des_set, des_analysis = build_turbojet_design(
            state=ft.State(),
            system=system,  # type: ignore
            settings=settings,
        )

        des_st, des_sys, des_set = des_analysis.run(des_st, des_sys, des_set) # type: ignore

        des_sys = des_sys.replace_subcomponent(des_sys.energy.sync_and_clear_nodes())

        save_data(des_sys, data_dir / "turbojet.fts")
        save_data(des_st, data_dir / "turbojet_design_state.fts")

        thermal_df, statics_df = validate_design_point(data_dir / "turbojet_DESIGN.json", des_st)
        thermal_df.to_csv(data_dir / "DESIGN_thermal.csv")
        statics_df.to_csv(data_dir / "DESIGN_statics.csv")

    else:
        des_sys: ft.Aircraft = load_data(data_dir / "turbojet.fts")

    # Update topology after loading
    des_sys = des_sys.update_network_topology()

    print("="*80)
    print(" System Validation")
    print("-"*80)

    for comp in des_sys.energy.line.engine.subcomponents:
        if hasattr(comp, "design_parameters") and comp.design_parameters:
            d = comp.design_parameters
            A_i = d.A_intake if d.A_intake else 1.0
            A_t = d.A_throat if d.A_throat else 1.0
            A_x = d.A_exit if d.A_throat else 1.0
            d_params = {"Intake Area": A_i, "Throat Area": A_t, "Exit Area":A_x}
            real_params = {k:a for k, a in d_params.items() if a != 1.0}
            if any(real_params):
                print(f"{comp.name}:")
                for p in real_params:
                    print(f" - {p:<11}: {format_array(real_params[p])}")
    
    print("Compressor Map Scaling:")
    c_map = des_sys.energy.line.engine.compressor.map
    print(f" - {'s_Wc':<11}: {format_array(c_map.s_Wc)}")
    print(f" - {'s_PR':<11}: {format_array(c_map.s_PR)}")
    print(f" - {'s_eff':<11}: {format_array(c_map.s_eff)}")
    print(f" - {'s_Nc':<11}: {format_array(c_map.s_Nc)}")

    print("Turbine Map Scaling:")
    t_map = des_sys.energy.line.engine.turbine.map
    print(f" - {'s_Wp':<11}: {format_array(t_map.s_Wp)}")
    print(f" - {'s_PR':<11}: {format_array(t_map.s_PR)}")
    print(f" - {'s_eff':<11}: {format_array(t_map.s_eff)}")
    print(f" - {'s_Np':<11}: {format_array(t_map.s_Np)}")
    print("="*80)
    print ("\n\n")


    if OFF_DESIGN_0:

        OD0 = TurbojetOpPoint(
            name="OD0",
            mach_number=1e-6,
            altitude = 0.0,
            thrust = 11_000 * units.lbf,
            compressor_Rline = 2.0,
            turbine_PR = 3.88,
            rotation_speed = 8197.38 * units.rpm,
            mass_flow_rate = 70.0,
            FAR = 0.0168
        )
        
        OD0_st, OD0_sys, OD0_set = off_design_point(
            system=des_sys,
            settings=settings,
            op_point=OD0
        )

    OD1 = TurbojetOpPoint(
                name="OD1",
                mach_number=0.2,
                altitude = 5_000 * units.ft,
                thrust = 8_000 * units.lbf,
                compressor_Rline = 2.0,
                turbine_PR = 4.669,
                rotation_speed = 8197.38 * units.rpm,
                mass_flow_rate = 168.45 * units.parse('lbm/s'),
                FAR = 0.0168
            )
    
    if OFF_DESIGN_1:
        
        OD1_st, OD1_sys, OD1_set = off_design_point(
            system=des_sys,
            settings=settings,
            op_point=OD1
        )

    if BATCHED_OD1:

        BOD1_st, BOD1_sys, BOD1_set = batched_off_design(
            system=des_sys,
            settings=settings,
            op_point=OD1
        )
    

    