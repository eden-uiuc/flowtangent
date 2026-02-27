import jax
jax.config.update('jax_disable_jit', True)

from RCAIDE.Framework import State, System, Settings, Process, ProcessStep
from RCAIDE.Framework.System import Aircraft, VehicleEnvelope, AircraftMassProperties
from RCAIDE.Framework.Missions.Segments import TestCSACruise
from RCAIDE.Framework.Analyses.Aerodynamics import TestAero


from RCAIDE.Library.Components import ComponentAreas, Airfoil, Airfoil_Data, MassProperties
from RCAIDE.Library.Components.Wings import *
from RCAIDE.Library.Components.Fuselages import *
from RCAIDE.Library.Components.Landing_Gear import LandingGear
from RCAIDE.Library.Components.Nacelles import Nacelle, NacelleDiameters
from RCAIDE.Library.Components.Energy.Propulsors import TurbofanEngine, DesignParameters
from RCAIDE.Library.Components.Energy.Stores import FuelTank, FuelTankMass
from RCAIDE.Library.Components.Energy.Lines.Jets import TurbofanEnergyLine

from RCAIDE.Library.Atmospheres import USStandard1976

import equinox as eqx
import jax.numpy as jnp

import tracemalloc
tracemalloc.start()

def vehicle_setup():

    # ------------------------------------------------------------------------------------------------------------------
    # Vehicle Level Parameters
    # ------------------------------------------------------------------------------------------------------------------

    mass_props = AircraftMassProperties(
        total               = 79015.8,   # kg
        max_takeoff         = 79015.8,   # kg
        takeoff             = 79015.8,   # kg
        operating_empty     = 62746.4,   # kg
        max_zero_fuel       = 62732.0,   # kg
        cargo               = 10000.0,   # kg
        center_of_gravity   = jnp.array([[15.30987849,   0.,             -0.48023939]]),  # Estimated
        moments_of_inertia  = jnp.array([[3173074.17,    0.,             28752.77565],
                                        [0.,             3019041.443,    0],
                                        [0.,             0.,             5730017.433]])
    )

    env = VehicleEnvelope(
        ultimate_load_factor=3.75,
        limit_load_factor=1.5
    )
    
    areas = ComponentAreas(
        reference=124.862
    )

    vehicle = Aircraft(
        tag='Boeing 737',
        passengers = 170,
        mass_properties=mass_props,
        areas=areas,
        envelope=env,
        design_mach_number=0.78,
        design_range=3582,
        design_cruise_alt=35000.0,
    )

    # ------------------------------------------------------------------------------------------------------------------
    # Main Wing
    # ------------------------------------------------------------------------------------------------------------------

    # Segments ---------------------------------------------------------------------------------------------------------

    # Root Segment

    root_sweeps = WingSweeps(
        quarter_chord=jnp.deg2rad(28.225)
    )
    root_segment = WingSegment(
        tag='Main Wing Root Segment',
        percent_span_location   = 0.0,
        twist                   = jnp.deg2rad(4.),
        root_chord_percent      = 1.,
        thickness_to_chord      = 0.1,
        dihedral_outboard       = jnp.deg2rad(2.5),
        sweeps                  = root_sweeps,
        airfoil                 = Airfoil.from_file(Airfoil_Data/'B737a.txt')
    )

    # Yehudi Segment
    yehudi_sweeps = WingSweeps(
        quarter_chord=jnp.deg2rad(25.)
    )
    yehudi_segment = WingSegment(
        tag='Main Wing Yehudi Segment',
        percent_span_location   = 0.324,
        twist                   = jnp.deg2rad(0.047193),
        root_chord_percent      = 0.5,
        thickness_to_chord      = 0.1,
        dihedral_outboard       = jnp.deg2rad(5.5),
        sweeps                  = yehudi_sweeps,
        airfoil                 = Airfoil.from_file(Airfoil_Data/'B737b.txt')
    )

    # Mid Segment

    mid_sweeps = WingSweeps(
        quarter_chord=jnp.deg2rad(56.75)
    )
    mid_segment = WingSegment(
        tag='Main Wing Mid Segment',
        percent_span_location   = 0.963,
        twist                   = jnp.deg2rad(0.00258),
        root_chord_percent      = 0.220,
        thickness_to_chord      = 0.1,
        dihedral_outboard       = jnp.deg2rad(5.5),
        sweeps                  = mid_sweeps,
        airfoil                 = Airfoil.from_file(Airfoil_Data/'B737c.txt'),
    )

    # Tip Segment

    tip_segment = WingSegment(
        tag='Main Wing Tip Segment',
        percent_span_location         = 1.,
        root_chord_percent            = 0.10077,
        thickness_to_chord            = 0.1,
        airfoil = Airfoil.from_file(Airfoil_Data/'B737d.txt'),
    )

    # Control Surfaces -------------------------------------------------------------------------------------------------

    slat = WingControlSurface(
        tag='Slat',
        span_fraction_start    = 0.2,
        span_fraction_end      = 0.963,
        deflection             = 0.0,
        chord_fraction         = 0.075,
        hinge_fraction         = 1.0,
    )

    flap = WingControlSurface(
        tag='Flap',
        span_fraction_start    = 0.2,
        span_fraction_end      = 0.7,
        deflection             = 0.0,
        configuration_type     = 'double_slotted',
        chord_fraction         = 0.30,
    )

    aileron = WingControlSurface(
        tag='Aileron',
        span_fraction_start = 0.7,
        span_fraction_end   = 0.963,
        deflection          = 0.0,
        chord_fraction      = 0.16,
        sign_duplicate      = -1.0,
    )

    main_controls = Component(
        tag="Main Wing Control Surfaces",
        subcomponents=(slat, flap, aileron)
    )

    # Wing Properties --------------------------------------------------------------------------------------------------

    main_sweeps = WingSweeps(
        quarter_chord=jnp.deg2rad(25.)
    )
    main_spans = WingDimensions(
        projected = 34.32
    )
    main_chords = WingChords(
        root             = 7.760,
        tip              = 0.782,
        mean_aerodynamic = 4.235,
    )
    main_areas = ComponentAreas(
        reference         = 124.862,
        wetted            = 225.08,
    )
    main_twists = WingDimensions(
        root = jnp.deg2rad(4.0),
        tip  = jnp.deg2rad(0.0),
    )
    
    main_wing = Wing(
        tag='Main Wing',
        aspect_ratio=10.18,
        thickness_to_chord=0.1,
        taper=0.1,
        origin=jnp.array([[13.61, 0., -0.93]]),
        aerodynamic_center=jnp.array([0, 0, 0]),
        vertical= False,
        symmetric= True,
        high_lift= True,
        dynamic_pressure_ratio = 1.0,
        sweeps=main_sweeps,
        spans=main_spans,
        chords=main_chords,
        areas=main_areas,
        twists=main_twists,
        segments=(root_segment, yehudi_segment, mid_segment, tip_segment),
        control_surfaces=main_controls
    )
    
    vehicle = vehicle.add_subcomponent(main_wing)

    # ------------------------------------------------------------------------------------------------------------------
    # Horizontal Stabilizer
    # ------------------------------------------------------------------------------------------------------------------

    # H-Stab Segments --------------------------------------------------------------------------------------------------
    
    h_root_sweeps = WingSweeps(
        quarter_chord=jnp.deg2rad(28.2250),
    )
    h_root_segment = WingSegment(
        tag = "Main Wing Root Segment",
        thickness_to_chord      = 0.1,
        percent_span_location   = 0.0,
        root_chord_percent      = 1.0,
        dihedral_outboard       = jnp.deg2rad(8.63),
    )

    h_tip_segment = WingSegment(
        tag='Horizontal Stabilizer Tip Segment',
        percent_span_location  = 1.,
        root_chord_percent     = 0.3333,
        thickness_to_chord     = .1,
    )

    # H-Stab Controls  -------------------------------------------------------------------------------------------------

    elevator = WingControlSurface(
        tag='Elevator',
        span_fraction_start   = 0.09,
        span_fraction_end     = 0.92,
        deflection            = 0.0,
        chord_fraction        = 0.3,

    )
    
    h_stab_controls = Component(
        "Horizontal Stabilizer Controls",
        subcomponents=(elevator,)
    )

    # H-Stab Properties  -----------------------------------------------------------------------------------------------

    h_sweeps = WingSweeps(
        quarter_chord=jnp.deg2rad(28.2250)
    )
    h_spans = WingDimensions(
        projected         = 14.4
    )
    h_chords = WingChords(
        root             = 4.2731,
        tip              = 1.4243,
        mean_aerodynamic = 8.0,
    )
    h_areas = ComponentAreas(
        reference         = 41.49,
        exposed           = 59.354,
        wetted            = 71.81,
    )
    h_twists = WingDimensions(
        root             = jnp.deg2rad(3.0),
        tip              = jnp.deg2rad(3.0),
    )
    
    h_stab = Wing(
        tag='Horizontal Stabilizer',
        aspect_ratio            = 4.99,
        thickness_to_chord      = 0.08,
        taper                   = 0.3333,
        dynamic_pressure_ratio  = 0.9,
        origin                  = jnp.array([[33.02, 0, 1.466]]),
        aerodynamic_center      = jnp.array([0, 0, 0]),
        vertical                = False,
        symmetric               = True,
        sweeps                  = h_sweeps,
        spans                   = h_spans,
        chords                  = h_chords,
        areas                   = h_areas,
        twists                  = h_twists,
        control_surfaces        = h_stab_controls,
        segments                = (h_root_segment, h_tip_segment)
    )

    # Add H-Stab to vehicle
    vehicle = vehicle.add_subcomponent(h_stab)

    # ------------------------------------------------------------------------------------------------------------------
    # Vertical Stabilizer
    # ------------------------------------------------------------------------------------------------------------------

    
    # V-Stab Segments --------------------------------------------------------------------------------------------------
    
    v_root_sweeps = WingSweeps(
        quarter_chord=61.485
    )
    root_segment = WingSegment(
        tag='Vertical Stabilizer Root Segment',
        percent_span_location   = 0.0,
        root_chord_percent      = 1.,
        thickness_to_chord      = .1,
        sweeps                  = v_root_sweeps
    )

    v_mid_sweeps = WingSweeps(
        quarter_chord=jnp.deg2rad(31.2)
    )
    mid_segment = WingSegment(
        tag='Vertical Stabilizer Mid Segment',
        percent_span_location   = 0.2962,
        root_chord_percent      = 0.45,
        sweeps                  = v_mid_sweeps,
        thickness_to_chord      = .1,
    )

    tip_segment = WingSegment(
        tag='Vertical Stabilizer Tip Segment',
        percent_span_location   = 1.0,
        root_chord_percent      = 0.1183,
        thickness_to_chord      = .1,
    )

    # V-Stab Properties ------------------------------------------------------------------------------------------------

    v_sweeps = WingSweeps(
        quarter_chord=jnp.deg2rad(31.2)
    )
    v_spans = WingDimensions(
        projected=8.33
    )
    v_chords = WingChords(
        root             = 10.1,
        tip              = 1.20,
        mean_aerodynamic = 4.0,
    )
    v_areas = ComponentAreas(
        reference=34.89,
        wetted=57.25
    )

    v_stab = Wing(
        tag='Vertical Stabilizer',
        aspect_ratio            = 1.98865,
        thickness_to_chord      = 0.08,
        taper                   = 0.1183,
        origin                  = jnp.array([[26.944, 0, 1.54]]),
        aerodynamic_center      = jnp.array([0, 0, 0]),
        vertical                = True,
        symmetric               = False,
        t_tail                  = False,
        dynamic_pressure_ratio  = 1.0,
        sweeps                  = v_sweeps,
        spans                   = v_spans,
        chords                  = v_chords,
        areas                   = v_areas
        
    )  

    vehicle = vehicle.add_subcomponent(v_stab)
    
    # ------------------------------------------------------------------------------------------------------------------
    # Fuselage
    # ------------------------------------------------------------------------------------------------------------------



    # Fuselage Segments ------------------------------------------------------------------------------------------------

    segment_specs = [
        # %X      %Z        Height   Width
        (0.00000, -0.00144, 0.01000, 0.01000),
        (0.00576, -0.00144, 0.75000, 0.65000),
        (0.02017,  0.00000, 1.52783, 1.20043),
        (0.03170,  0.00000, 1.96435, 1.52783),
        (0.04899,  0.00431, 2.72826, 1.96435),
        (0.07781,  0.00861, 3.49217, 2.61913),
        (0.10375,  0.01005, 3.70130, 3.05565),
        (0.16427,  0.01148, 3.92870, 3.92870),
        (0.69164,  0.01292, 3.81957, 3.81957),
        (0.71758,  0.01292, 3.81957, 3.81957),
        (0.78098,  0.01722, 3.49217, 3.71043),
        (0.85303,  0.02296, 3.05565, 3.16478),
        (0.91931,  0.03157, 2.40087, 1.96435),
        (1.00000,  0.04593, 1.09130, 0.21826)
    ]

    f_segments = []
    for idx, (x, z, h, w) in enumerate(segment_specs):
        s_h = ComponentDimensions(
            maximum=h
        )
        s_w = ComponentDimensions(
            maximum=w
        )
        segment = FuselageSegment(
            tag=f'Fuselage Segment {idx}',
            percent_x_location  = x,
            percent_z_location  = z,
            heights             = s_h,
            widths              = s_w,
        )
        f_segments.append(segment)

    # Fuselage Properties ----------------------------------------------------------------------------------------------    

    f_diameters = ComponentDimensions(
        effective=3.74
    )
    f_fineness = ComponentFineness(
        nose = 1.6,
        tail = 2.
    )
    f_lengths = FuselageLengths(
        nose       = 6.4,
        tail       = 8.0,
        cabin      = 28.85,
        total      = 38.02,
        fore_space = 6.,
        aft_space  = 5.,

    )
    f_widths = ComponentDimensions(
        maximum=3.74
    )
    f_heights = FuselageHeights(
        maximum = 3.74,
        at_quarter_length = 3.74,
        at_three_quarters_length = 3.65,
        at_wing_root_quarter_chord = 3.74,
    )
    f_areas = ComponentAreas(
        side_projected = 142.1948,
        wetted = 385.51,
        front_projected = 12.57,
    )

    fuse = Fuselage(
        tag='Fuselage',
        number_of_seats         = 170,
        seats_abreast           = 6,
        seat_pitch              = 0.7874,
        differential_pressure   = 5.0e4,
        diameters               = f_diameters,
        fineness                = f_fineness,
        lengths                 = f_lengths,
        widths                  = f_widths,
        heights                 = f_heights,
        areas                   = f_areas,
        segments                = tuple(f_segments)
    )

    vehicle = vehicle.add_subcomponent(fuse)

    # ------------------------------------------------------------------------------------------------------------------
    # Landing Gear
    # ------------------------------------------------------------------------------------------------------------------

    mlg = LandingGear(
        tag='Main Landing Gear',
        number_of_wheels    = 2,
        tire_diameter       = 1.12,
        strut_length        = 1.8,
    )

    vehicle.add_subcomponent(mlg)

    nlg = LandingGear(
        tag='Nose Landing Gear',
        number_of_wheels    = 2,
        tire_diameter       = 1.12,
        strut_length        = 1.3,
    )
    
    vehicle.add_subcomponent(nlg)

    # ------------------------------------------------------------------------------------------------------------------
    # Nacelles
    # ------------------------------------------------------------------------------------------------------------------
    
    n_lengths = ComponentDimensions(
        total=2.71
    )
    n_diams = NacelleDiameters(
        maximum=2.05,
        inlet=1.90
    )
    n_areas = ComponentAreas(
        wetted = 1.1 * jnp.pi * 2.05 * 2.71
    )

    nacelle = Nacelle(
        tag             ='Engine Nacelle 1',
        flow_through    = True,
        airfoil         = Airfoil.NACA_4_Series('2410'),
        origin          = jnp.array([[13.72, -4.86, -1.9]]),
        lengths         = n_lengths,
        diameters       = n_diams,
        areas           = n_areas
    )

    nacelle_2 = eqx.tree_at(lambda n: (n.tag, n.origin), nacelle, ("Engine Nacelle 2", jnp.array([[13.72, 4.86, -1.9]])))

    vehicle = vehicle.add_subcomponent(nacelle)
    vehicle = vehicle.add_subcomponent(nacelle_2)

    # ------------------------------------------------------------------------------------------------------------------
    # Turbofan Engines
    # ------------------------------------------------------------------------------------------------------------------

    lines = vehicle.energy.lines
    lines = lines.add_subcomponent(TurbofanEnergyLine())
    vehicle = eqx.tree_at(lambda v:v.energy.lines, vehicle, lines)

    tf_lengths = ComponentDimensions(
        total = 2.71
    )
    tf_des = DesignParameters(
        total_thrust            = 24000.,
        altitude                = 10668.,
        mach_number             = 0.78,
    )

    tf = TurbofanEngine(
        tag = "Engine 1",
        origin=jnp.array([[13.72, -4.86, -1.9]]),
        bypass_ratio= 5.4,
        plug_diameter= 0.1,
        lengths=tf_lengths,
        design_thrust_parameters=tf_des,
    )

    # Converters -------------------------------------------------------------------------------------------------------
    
    cons = tf.converters

    # Inlet Nozzle
    cons = eqx.tree_at(
        lambda c: (c.inlet_nozzle.polytropic_efficiency, c.inlet_nozzle.pressure_ratio),
        cons, (0.98, 0.98)
    )
    
    # Fan
    cons = eqx.tree_at(
        lambda c: (c.fan.polytropic_efficiency, c.fan.pressure_ratio),
        cons, (0.93, 1.7)
    )

    # Low Pressure Compressor
    cons = eqx.tree_at(
        lambda c: (c.compressors[0].polytropic_efficiency, c.compressors[0].pressure_ratio),
        cons, (0.91, 1.14)
    )

    # High Pressure Compressor
    cons = eqx.tree_at(
        lambda c: (c.compressors[1].polytropic_efficiency, c.compressors[1].pressure_ratio),
        cons, (0.91, 13.415)
    )

    # Combustor
    cons = eqx.tree_at(
        lambda c: (c.combustor.efficiency, c.combustor.pressure_ratio),
        cons, (0.99, 0.95)
    )

    # High Pressure Turbine
    cons = eqx.tree_at(
        lambda c: (c.turbines[0].mechanical_efficiency, c.turbines[0].polytropic_efficiency),
        cons, (0.91, 0.93)
    )

    # Low Pressure Turbine
    cons = eqx.tree_at(
        lambda c: (c.turbines[1].mechanical_efficiency, c.turbines[1].polytropic_efficiency),
        cons, (0.99, 0.93)
    )

    # Core Nozzle
    cons = eqx.tree_at(
        lambda c: (c.core_nozzle.polytropic_efficiency, c.core_nozzle.pressure_ratio, c.diameters.reference),
        cons, (0.95, 0.99, 0.92)
    )

    # Fan Nozzle
    cons = eqx.tree_at(
        lambda c: (c.fan_nozzle.polytropic_efficiency, c.fan_nozzle.pressure_ratio, c.diameters.reference),
        cons, (0.95, 0.99, 1.659)
    )

    tf = eqx.tree_at(lambda t: t.converters, tf, cons)    

    tf2 = tf.tree_at(lambda t: (t.tag, t.origin), tf, ("Engine 2", jnp.array([[13.72, 4.86, -1.9]])))

    line_cons = vehicle.energy.lines[0].converters
    line_cons = line_cons.add_subcomponent(tf)
    line_cons = line_cons.add_subcomponent(tf2)

    vehicle = eqx.tree_at(lambda v: v.energy.lines[0].converters, vehicle, line_cons)

    # ------------------------------------------------------------------------------------------------------------------
    # Fuel Tanks
    # ------------------------------------------------------------------------------------------------------------------

    fuel_mass = FuelTankMass(
        full_fuel_mass=79015.8-62732.0, # Max Takeoff - Max Zero Fuel
        center_of_gravity=jnp.array([[13.61, 0., -0.93]]),
    )
    fuel = FuelTank(
        origin = jnp.array([[13.61, 0., -0.93]]),
        mass_properties=fuel_mass
    )

    line_stores = vehicle.energy.lines[0].stores
    line_stores = line_stores.add_subcomponent(fuel)

    vehicle = eqx.tree_at(lambda v: v.energy.lines[0].stores, vehicle, line_stores)

    # ------------------------------------------------------------------------------------------------------------------
    # Configurations
    # ------------------------------------------------------------------------------------------------------------------

    # Takeoff Configuration

    # takeoff_config = deepcopy(vehicle)
    # takeoff_config.tag = "Takeoff"
    # takeoff_config.wings.main_wing.control_surfaces.flap.deflection    = jnp.deg2rad(20)
    # takeoff_config.wings.main_wing.control_surfaces.slat.deflection    = jnp.deg2rad(25)

    # for tf in takeoff_config.energy.lines[0].converters:

    #     tf: rcl.Components.Energy.Propulsors.TurbofanEngine
    #     tf.converters.fan.rotation_speed        = 2780.
    #     tf.converters.fan_nozzle.noise_speed    = 315.
    #     tf.converters.core_nozzle.noise_speed   = 415.

    # vehicle.configurations.add_subcomponent(takeoff_config)

    # # Cutback Configuration

    # cutback_config = deepcopy(vehicle)
    # cutback_config.tag = "Cutback"
    # cutback_config.wings.main_wing.control_surfaces.flap.deflection    = jnp.deg2rad(20)
    # cutback_config.wings.main_wing.control_surfaces.slat.deflection    = jnp.deg2rad(20)

    # for tf in cutback_config.energy.lines[0].converters:

    #     tf: rcl.Components.Energy.Propulsors.TurbofanEngine
    #     tf.converters.fan.rotation_speed        = 2780.
    #     tf.converters.fan_nozzle.noise_speed    = 210.
    #     tf.converters.core_nozzle.noise_speed   = 360.

    # vehicle.configurations.add_subcomponent(cutback_config)

    # # Landing Configuration

    # landing_config = deepcopy(vehicle)
    # landing_config.tag = "Landing"
    # landing_config.wings.main_wing.control_surfaces.flap.deflection    = jnp.deg2rad(30)
    # landing_config.wings.main_wing.control_surfaces.slat.deflection    = jnp.deg2rad(25)

    # landing_config.landing_gear.main_landing_gear.deployed = True
    # landing_config.landing_gear.main_landing_gear.deployed = True

    # for tf in landing_config.energy.lines[0].converters:

    #     tf: rcl.Components.Energy.Converters.TurbofanEngine
    #     tf.converters.fan.rotation_speed        = 2030.
    #     tf.converters.fan_nozzle.noise_speed    = 109.3
    #     tf.converters.core_nozzle.noise_speed   = 92.

    # vehicle.configurations.add_subcomponent(landing_config)

    return vehicle


def mission_setup(settings: "Settings"):

    mission = Process(
        tag='Boeing 737 Mission',
        steps=(TestCSACruise(altitude=10000.0, air_speed=230.0, distance=5500. * 1000.),) #type: ignore
    )

    final_segments = []
    for segment in mission.steps:

        seg_w_analyis = eqx.tree_at(lambda s:s.analyze.aerodynamics, segment, TestAero())
        final_segments.append(seg_w_analyis)

    return eqx.tree_at(lambda m:m.steps, mission, tuple(final_segments))


def state_setup():

    state = State()

    state.freestream.atmosphere = USStandard1976()

    frozen_initials = eqx.tree_at(lambda s: s.initials, state, None)
    
    state = eqx.tree_at(lambda s: s.initials, state, frozen_initials)

    return state


def mission_b737(state, system, settings):


    mission = mission_setup(settings)
    mission = eqx.tree_at(lambda m: m.initial_state   , mission, state)
    mission = eqx.tree_at(lambda m: m.initial_system  , mission, system)
    mission = eqx.tree_at(lambda m: m.initial_settings, mission, settings)

    final_state, final_system, final_settings = mission.run(state, system, settings)

    return final_state, final_system, final_settings


if __name__ == '__main__':
  
    state = state_setup()
    system = vehicle_setup()
    settings = Settings()

    st, sy, se = mission_b737(state, system, settings)

    print("Done!")