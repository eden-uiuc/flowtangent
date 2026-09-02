import os
import sys

def numerical_environment():
    # 1. JAX Memory/Precision Config (Safe everywhere)
    os.environ["JAX_ENABLE_X64"] = "True"
    
    # 2. NUMA / Hardware Auto-Detection
    if sys.platform == "linux":
        # A simple heuristic: if you have a massive amount of cores, 
        # it's likely the Threadripper workstation.
        cpu_count = os.cpu_count() or 1
        if cpu_count > 16:  # Adjust threshold based on your hardware
            try:
                # Bind to the first 16 cores (Node 0) to prevent cross-NUMA memory latency
                node_0_cores = set(range(16))
                os.sched_setaffinity(0, node_0_cores)
                
                # Tell OpenMP to respect this boundary
                os.environ["OMP_PROC_BIND"] = "true"
                os.environ["OMP_PLACES"] = "cores"
                print(f"Hardware Config: NUMA affinity set to Node 0 (16 cores).")
            except Exception as e:
                print(f"Hardware Config Warning: Could not set CPU affinity: {e}")

    cache_path = os.path.expanduser("~/.eden_trace/jax_cache")
    os.makedirs(cache_path, exist_ok=True)
    os.environ["JAX_COMPILATION_CACHE_DIR"] = cache_path


numerical_environment()

import json

import jax.numpy as jnp
import equinox as eqx
import numpy as np
import pandas as pd

from pathlib import Path
from dataclasses import replace

from eden_trace.utils import save_data, load_data, format_array, configure_environment

from eden_trace.library import units
from eden_trace.library.components.energy.networks import TurbojetNetwork, JetNetDesign
from eden_trace.library.components.energy.jets.classes import TurbojetEngine, TurbojetDesign
from eden_trace.library.components.energy.lines import TurbojetLine

from eden_trace.framework import State, Aircraft, Settings
from eden_trace.framework.settings import LoggingSettings
from eden_trace.framework.analyses.energy.jets import turbojet_design, turbojet_performance, JetSettings
from eden_trace.framework.simulation.initialize import initialize_energy
from eden_trace.framework.simulation.update import update_freestream

def system_setup():

    engine_design = TurbojetDesign(
        thrust=11_800 * units.lbf,
        mass_flow_rate=168.45 * units.lbm/units.s,
        rotation_speed=8070. * units.rpm,
        overall_pressure_ratio=13.5,
        turbine_PR=4.46,
        turbine_intake_temperature=2370.0 * units.R,
    )
    engine = TurbojetEngine.build_custom(variable_nozzle=True, design_parameters=engine_design)
    
    inlet_design = eqx.tree_at(lambda i:
        i.design_parameters.exit_mach_number,
        engine.inlet,
        0.6
    )
    
    comp_design = eqx.tree_at(lambda c:
        (
            c.design_parameters.exit_mach_number,
            c.design_parameters.eff.flow,
        ),
        engine.compressor,
        (
            0.02,
            0.83
        )
    )

    burn_design = eqx.tree_at(lambda b:
        (
            b.design_parameters.output_temperature,
            b.design_parameters.pressure_ratio,
            b.design_parameters.exit_mach_number,
        ),
        engine.burner,
        (
            2370 * units.R,
            0.97,
            0.02,
        )
    )

    turb_design = eqx.tree_at(lambda t:
        (
            t.design_parameters.eff.flow,
            t.design_parameters.exit_mach_number,
        ),
        engine.turbine,
        (
            0.86,
            0.4,
        )
    )

    nozz_design = eqx.tree_at(
        lambda n: n.design_parameters.eff.flow,
        engine.core_nozzle,
        1.0
    )

    nozz_design = replace(nozz_design, variable_exit=True)

    des_engine = eqx.tree_at(lambda e:
        (
            e.inlet,
            e.compressor,
            e.burner,
            e.turbine,
            e.core_nozzle,
        ), engine,
        (
            inlet_design,
            comp_design,
            burn_design,
            turb_design,
            nozz_design,
        )                
    )

    line = TurbojetLine(tag="Line", subcomponents=(des_engine,),)
    
    net_design = JetNetDesign(
        altitude=0.0,
        mach_number=1e-6,
        thrust=11_800 * units.lbf,
    )
    
    net = TurbojetNetwork(subcomponents=(line,), design_parameters=net_design)
    
    sys = Aircraft(tag="Simple Turbojet System", subcomponents=(net,))

    return sys

def off_design_point(
    M0: float,
    alt: float,
    thrust: float,
    system: Aircraft,
    settings: Settings,
    initial_Rline: float | jnp.ndarray = 2.0,
    initial_turb_PR: float | jnp.ndarray = 5.0,
    initial_RPM: float | jnp.ndarray = 1000 * units.rpm,
    initial_MFR: float | jnp.ndarray = 100 * units.kg / units.s,
    initial_FAR: float | jnp.ndarray = 1e-4,
):

    network: TurbojetNetwork = system.energy
    des: JetNetDesign = network.design_parameters

    atmo = des.atmosphere_model
    a0 = atmo.compute_speed_of_sound(alt)

    od_state = eqx.tree_at(
        lambda s: (
            s.frames.inertial.position_vector,
            s.freestream.mach_number,
            s.frames.inertial.velocity_vector,
        ),
        State().expand_time(1),
        (
            jnp.array([[0., 0., -alt]]),
            jnp.atleast_2d(M0),
            jnp.atleast_2d(jnp.array([[(a0 * M0).item(), 0.0, 0.0]])),
        ),
    )

    od_analysis = turbojet_performance(
        network,
        initial_Rline,
        initial_turb_PR,
        initial_RPM,
        initial_MFR,
        initial_FAR,
    )

    od_state, od_system, od_settings = initialize_energy(od_state, system, settings)
    od_state, od_system, od_settings = update_freestream(od_state, od_system, od_settings)
    od_state = eqx.tree_at(
        lambda s: (
            s.energy.target_thrust,
        ),
        od_state,
        (
            jnp.atleast_2d(thrust),
        )
    )

    new_settings = JetSettings(design_mode=False, statics=od_settings.analysis.energy.statics)
    od_settings = eqx.tree_at(lambda s: s.analysis.energy, od_settings, new_settings)
    od_state, od_system, od_settings = od_analysis.run(od_state, od_system, od_settings, initialize=True)

    return od_state, od_system, od_settings

def validate_design_point(pycycle_json_path, Trace_state, point_name: str="Design"):
    """
    Loads PyCycle JSON results and compares them against the Trace state.
    """
    
    # 1. Load the JSON
    with open(pycycle_json_path, 'r') as f:
        pycycle_data = json.load(f)
        
    # 2. Map PyCycle flow stations to Trace network IDs
    station_map = {
        # 'fc.Fl_O':     'freestream',
        'inlet.Fl_O':  'network.line.engine.inlet',
        'comp.Fl_O':   'network.line.engine.compressor',
        'burner.Fl_O': 'network.line.engine.burner',
        'turb.Fl_O':   'network.line.engine.turbine',
        'nozz.Fl_O':   'network.line.engine.core_nozzle'
    }
    
    # 3. Map PyCycle properties to Trace tags
    property_map = {
        'W':     ('mass_flow_rate', units.lbm/units.s),
        'Pt':    ('stagnation_pressure', units.psi),
        'Tt':    ('stagnation_temperature', units.R),
        'ht':    ('stagnation_enthalpy', units.btu/units.lbm),
        'Ps':    ('pressure', units.psi),
        'Ts':    ('temperature', units.R),
        'MN':    ('mach_number', 1.0),
        'V':     ('speed', units.ft/units.s),
    }

    # User-defined extraction function
    def get_Trace_value(state, network_id, prop_tag):

        node = state.energy.nodes[network_id]
        value = np.asarray(getattr(node.flow, prop_tag))
        if value.size == 1:
            return value.item()
        else:
            return None
    
    # 4. Assemble the Comparison
    records = []
    
    for pyc_prop, pyc_val in pycycle_data['flow_stations']['fc.Fl_O'].items():
        
        if pyc_prop == 'W':
            continue
        elif pyc_prop == "ht":
            continue
        
        if pyc_prop in property_map:
            prop_tag, pyc_units = property_map[pyc_prop]
            value = getattr(Trace_state.freestream, prop_tag)
            Trace_val = np.asarray(value).item()

            pyc_val *= pyc_units
            diff = Trace_val - pyc_val

            if abs(pyc_val) > 1e-12:
                    rel_error = (diff / pyc_val)
            else:
                rel_error = np.nan if abs(Trace_val) > 1e-12 else 0.0
            
            records.append({
                'Station': "fc",
                'Property': pyc_prop,
                'PyCycle Val': pyc_val,
                'FlowTan Val': Trace_val,
                'Diff': diff,
                'Rel. Error': rel_error,
                'Mag. Error': np.abs(rel_error)
            })
    
    for pyc_station, pyc_props in pycycle_data.get('flow_stations', {}).items():
        
        Trace_node_id = station_map.get(pyc_station)
        if not Trace_node_id:
            continue
            
        for pyc_prop, pyc_val in pyc_props.items():

            if pyc_prop == "ht":
                continue
            
            Trace_tag, pyc_units = property_map.get(pyc_prop, (None, None))
            if not Trace_tag or pyc_val is None:
                continue
                
            # Grab the Trace value
            Trace_val = get_Trace_value(Trace_state, Trace_node_id, Trace_tag)
            if Trace_val is not None:
                # Calculate metrics
                pyc_val *= pyc_units
                diff = Trace_val - pyc_val
                
                # Protect against divide-by-zero if PyCycle value is exactly 0.0 (like altitude or Mn=0)
                if abs(pyc_val) > 1e-12:
                    rel_error = (diff / pyc_val)
                else:
                    # If PyCycle is 0, use absolute difference as the "error" metric visually, or set to NaN
                    rel_error = np.nan if abs(Trace_val) > 1e-12 else 0.0
                    
                records.append({
                    'Station': pyc_station.split('.')[0],
                    'Property': pyc_prop,
                    'PyCycle Val': pyc_val,
                    'FlowTan Val': Trace_val,
                    'Diff': diff,
                    'Rel. Error': rel_error,
                    'Mag. Error': np.abs(rel_error)
                })
            
    # Convert to DataFrame
    df = pd.DataFrame(records)
    
    # Optional: Format the DataFrame for cleaner printing
    pd.set_option('display.max_rows', None)
    pd.set_option('display.float_format', '{:.4e}'.format)
    
    print("\n" + "="*80)
    print(f" PyCycle vs. FlowTangent {point_name} Point Validation")
    print("-"*80)
    print(df.drop(columns='Mag. Error').to_string(index=False))
    
    print("\n"+"-"*80)
    print(" Error Magnitude Summary:")
    print(f" - Mean: {df['Mag. Error'].mean():.4e}")
    print(f" - Min:  {df['Mag. Error'].min():.4e}")
    print(f" - Max:  {df['Mag. Error'].max():.4e}")
    print("\n" + "="*80 + "\n")

    return df

if __name__ == "__main__":
    
    # Control Board
    DEV = False
    DEBUG = False
    VERBOSE = True

    STATICS = False

    DESIGN_POINT = True
    OFF_DESIGN_0 = True
    OFF_DESIGN_1 = True

    # Build Turbojet------------------------------------------------------------
    test_dir = Path(__file__).resolve().parent
    data_dir = test_dir / "ft_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    system = system_setup()
    settings = eqx.tree_at(
        lambda s: s.analysis.energy,
        Settings(_DEV_MODE=DEV, DEBUG_MODE=DEBUG, verbose=VERBOSE,
                 logging=LoggingSettings(log_dir=test_dir/"ft_logs")),
        JetSettings(design_mode=DESIGN_POINT, statics=STATICS)
    )
    configure_environment(settings)
    

    if DESIGN_POINT:
        print("="*80)
        print(" Design Point Analysis")
        print("-"*80)

        des_st, des_sys, des_set = turbojet_design(
            state=State(),
            system=system,  # type: ignore
            settings=settings,
            initialize=True
        )

        des_sys = des_sys.replace_subcomponent(des_sys.energy.sync_and_clear_nodes())

        save_data(des_sys, data_dir / "turbojet.fts")
        save_data(des_st, data_dir / "turbojet_design_state.fts")

        validation_df = validate_design_point(data_dir / "turbojet_DESIGN.json", des_st)
        validation_df.to_csv(data_dir / "DESIGN_validation.csv")
     
    else:
        des_sys: Aircraft = load_data(data_dir / "turbojet.fts")
    
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
                print(f"{comp.tag}:")
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
        print("="*80)
        print(" Off Design Point 0 Analysis")
        print("-"*80)
        
        OD0_st, OD0_sys, OD0_set = off_design_point(
            M0=1e-6,
            alt=0.0,
            thrust=11_000 * units.lbf,
            system=des_sys,
            settings=settings,
            initial_Rline=2.0,
            initial_turb_PR=3.88,
            initial_RPM=8197.38 * units.rpm,
            initial_MFR=70.00,
            initial_FAR=0.0168,
        )

        OD0_df = validate_design_point(
            data_dir / "turbojet_OD0.json",
            OD0_st,
            point_name="Off Design 0"
        )
        OD0_df.to_csv(
            data_dir / "OD0_validation.csv"
        )
    
    if OFF_DESIGN_1:
        print("="*80)
        print(" Off Design Point 1 Analysis")
        print("-"*80)
        
        OD1_st, OD1_sys, OD1_set = off_design_point(
            M0=0.2,
            alt=5_000 * units.ft,
            thrust=8_000 * units.lbf,
            system=des_sys,
            settings=settings,
            initial_Rline= 2.0,
            initial_turb_PR=4.669,
            initial_RPM= 8197.38 * units.rpm,
            initial_MFR= 168.45 * units.parse('lbm/s'),
            initial_FAR= 0.01680,
        )

        OD0_df = validate_design_point(
            data_dir / "turbojet_OD1.json",
            OD1_st,
            point_name="Off Design 1"
        )
        OD0_df.to_csv(data_dir / "OD1_validation.csv")

    
    