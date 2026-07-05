import json

import jax.numpy as jnp
import equinox as eqx
import numpy as np
import pandas as pd

from src.eden_trace.utils import save_data, load_data, format_array

from src.eden_trace.library import units
from src.eden_trace.library.components.energy.networks import TurbojetEnergyNetwork, TurbojetDesign
from src.eden_trace.library.components.energy.propulsors import TurbojetEngine, JetDesign
from src.eden_trace.library.components.energy.lines import TurbojetEnergyLine

from src.eden_trace.framework import State, Aircraft, Settings
from src.eden_trace.framework.analyses.energy.turbojets import design_turbojet, TurbojetPerformance
from src.eden_trace.framework.missions.initialize import initialize_energy
from src.eden_trace.framework.missions.update import update_freestream

def system_setup():

    engine_design = JetDesign(
        thrust=11_800 * units.lbf,
    )
    engine = TurbojetEngine(tag="Engine", design_parameters=engine_design)
    
    inlet_design = eqx.tree_at(lambda c:
        c.design_parameters.exit_mach_number,
        engine.inlet_nozzle,
        0.6
    )
    
    comp_design = eqx.tree_at(lambda c:
        (
            c.design_parameters.pressure_ratio,
            c.design_parameters.rotation_speed,
            c.efficiencies.flow,
        ),
        engine.compressor,
        (
            13.5,
            8070. * units.rev/units.mins,
            0.83
        )
    )

    burn_design = eqx.tree_at(lambda b:
        (
            b.design_parameters.output_temperature,
            b.design_parameters.pressure_ratio,
        ),
        engine.combustor,
        (
            2370 * units.R,
            0.97,
        )
    )

    turb_design = eqx.tree_at(lambda t:
        (
            t.efficiencies.flow,
            t.design_parameters.rotation_speed,
        ),
        engine.turbine,
        (
            0.86,
            8070. * units.rev/units.mins,
        )
    )

    engine = eqx.tree_at(lambda e:
        (
            e.inlet_nozzle,
            e.compressor,
            e.combustor,
            e.turbine
        ), engine,
        (
            inlet_design,
            comp_design,
            burn_design,
            turb_design
        )                
    )

    line = TurbojetEnergyLine(tag="Line", subcomponents=(engine,),)
    
    net_design = TurbojetDesign(
        altitude=0.0,
        mach_number=1e-6,
        thrust=11_800 * units.lbf,
        initial_MFR=168.453135137 * units.lbm / units.s,
        initial_turb_PR=4.46138725662
    )
    
    net = TurbojetEnergyNetwork(subcomponents=(line,), design_parameters=net_design)
    
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
    initial_RPM: float | jnp.ndarray = 1000 * units.rev / units.mins,
    initial_MFR: float | jnp.ndarray = 100 * units.kg / units.s,
    initial_FAR: float | jnp.ndarray = 1e-4,
):

    network: TurbojetEnergyNetwork = system.energy
    des: TurbojetDesign = network.design_parameters

    atmo = des.atmosphere_model
    a0 = atmo.compute_speed_of_sound(alt)

    od_state = eqx.tree_at(
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

    od_analysis = TurbojetPerformance(
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
    od_state, od_system, od_settings = od_analysis(od_state, od_system, od_settings)

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
        'inlet.Fl_O':  'network.line.engine.inlet_nozzle',
        'comp.Fl_O':   'network.line.engine.compressor',
        'burner.Fl_O': 'network.line.engine.combustor',
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
        value = np.asarray(getattr(node.outputs.flow, prop_tag))
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
                'Trace Val': Trace_val,
                'Diff': diff,
                'Rel. Error': rel_error
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
                    'Trace Val': Trace_val,
                    'Diff': diff,
                    'Rel. Error': rel_error
                })
            
    # Convert to DataFrame
    df = pd.DataFrame(records)
    
    # Optional: Format the DataFrame for cleaner printing
    pd.set_option('display.max_rows', None)
    pd.set_option('display.float_format', '{:.4e}'.format)
    
    print("\n" + "="*80)
    print(f" PyCycle vs. Trace {point_name} Point Validation")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")
    
    return df

if __name__ == "__main__":
    
    # Control Board
    DEBUG = False
    VERBOSE = True

    DESIGN_POINT = True
    OFF_DESIGN_0 = True
    OFF_DESIGN_1 = False

    # Build Turbojet------------------------------------------------------------
    system = system_setup()

    settings = Settings(DEBUG_MODE=DEBUG, verbose=VERBOSE)

    if DESIGN_POINT:
        print("-"*60)
        print(" Design Point Analysis")
        print("-"*60)
        st, sys, set = design_turbojet(
            state=State(),
            system=system,
            settings=settings,
        )   

        validation_df = validate_design_point(
            "./Tests/PyCycle/turbojet_DESIGN.json",
            st
        )
        validation_df.to_csv(
            "./Tests/PyCycle/DESIGN_validation.csv"
        )

        save_data(sys, "./Tests/PyCycle/simple_turbojet.rcs")
    else:
        sys: Aircraft = load_data("./Tests/PyCycle/simple_turbojet.rcs")
    
    sys = sys.sort_network_topology()

    print("-"*60)
    print(" System Validation")
    print("-"*60)

    for comp in sys.energy.line.engine.subcomponents:
        if hasattr(comp, "design_parameters") and comp.design_parameters:
            d = comp.design_parameters
            A_i = d.A_intake
            A_t = d.A_throat
            A_x = d.A_exit
            AR = d.A_ratio
            d_params = {"Intake Area": A_i, "Throat Area": A_t, "Exit Area":A_x, "Area_Ratio":AR}
            real_params = {k:a for k, a in d_params.items() if a != 1.0}
            if any(real_params):
                print(f"{comp.tag}:")
                for p in real_params:
                    print(f" - {p:<11}: {format_array(real_params[p])}")
    
    print("Compressor Map Scaling:")
    c_map = sys.energy.line.engine.compressor.map
    print(f" - s_Wc:  {format_array(c_map.s_Wc)}")
    print(f" - s_eff: {format_array(c_map.s_eff)}")

    print("Turbine Map Scaling:")
    t_map = sys.energy.line.engine.turbine.map
    print(f" - s_Wp:  {format_array(t_map.s_Wp)}")
    print(f" - s_eff: {format_array(t_map.s_eff)}")
    print ("\n\n")


    if OFF_DESIGN_0:
        print("-"*60)
        print(" Off Design Point 0 Analysis")
        print("-"*60)
        
        OD0_st, OD0_sys, OD0_set = off_design_point(
            M0=1e-6,
            alt=0.0,
            thrust=11_000 * units.lbf,
            system=sys,
            settings=settings,
            initial_Rline= 2.0,
            initial_turb_PR=3.86,
            initial_RPM= 8197.38 * units.rev / units.mins,
            initial_MFR= 166.073 * units.lbm / units.s,
            initial_FAR= 0.01680,
        )

        OD0_df = validate_design_point(
            "./Tests/PyCycle/turbojet_OD0.json",
            OD0_st,
            point_name="Off Design 0"
        )
        OD0_df.to_csv(
            "./Tests/PyCycle/OD0_validation.csv"
        )
    
    if OFF_DESIGN_1:
        print("-"*60)
        print(" Beginning Off Design Point 1 Analysis")
        print("-"*60)
        
        OD1_st, OD1_sys, OD1_set = off_design_point(
            M0=0.2,
            alt=5_000 * units.ft,
            thrust=8_000 * units.lbf,
            system=sys,
            settings=settings,
            initial_Rline= 2.0,
            initial_turb_PR=3.86,
            initial_RPM= 8197.38 * units.rev / units.mins,
            initial_MFR= 145.5 * units.lbm / units.s,
            initial_FAR= 0.01680,
        )

        OD0_df = validate_design_point(
            "./Tests/PyCycle/turbojet_OD1.json",
            OD1_st,
            point_name="Off Design 1"
        )
        OD0_df.to_csv(
            "./Tests/PyCycle/OD1_validation.csv"
        )



    
    