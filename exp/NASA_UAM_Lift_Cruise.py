import flowtangent as ft

from flowtangent.components.wings import Wing, WingChords, WingDimensions, WingSegment, WingSweeps 
from flowtangent.data import units
from flowtangent.core._component import Areas

# from flowtangent.solve.aero._vorjax import VORJAXSettings, InitializeVORJAX
from flowtangent.plots import plot_airfoil, plot_panels
import jax.numpy as jnp 
# from flowtangent.solve.aero._vorjax import PanelSettings
from flowtangent.solve.aero._vorjax import calculate_macro_properties,morph_to_3d_mesh,LatticeData

def wing_setup():
    # Wing setup

    root_segment = WingSegment(
        name='Main Wing Root Segment',
        percent_span_location   = 0.0,
        twist                   = -0.5 * units.deg,
        root_chord_percent      = 1.,
        thickness_to_chord      = 0.17,
        dihedral_outboard       =  -2 * units.deg,
        sweeps                  =WingSweeps(quarter_chord= 2.29 * units.deg),
        airfoil=ft.airfoils.ls417)

    mid_segment_1 = WingSegment(
        name='Main Wing Mid Segment 1 ',
        percent_span_location=0.583,
        twist= -0.5 * units.deg,
        root_chord_percent= 0.7589,
        thickness_to_chord= 0.17,
        dihedral_outboard= -2 * units.deg,
        sweeps=WingSweeps(quarter_chord= 3.21 * units.deg),
        airfoil=ft.airfoils.ls417)
        
    mid_segment_2 = WingSegment(
        name = 'Main Wing Mid Segment 2',
        percent_span_location = 0.8456,
        twist = -0.5 * units.deg,
        root_chord_percent = 0.6071,
        thickness_to_chord=0.17,
        dihedral_outboard= -2 * units.deg,
        sweeps=WingSweeps(quarter_chord =9.272 * units.deg),
        airfoil=ft.airfoils.ls417)
    
    mid_segment_3 = WingSegment(
        name='Main Wing Mid Segment 3',
        percent_span_location=0.928,
        twist= -0.5 * units.deg,
        root_chord_percent=0.522,
        thickness_to_chord=0.17,
        dihedral_outboard= -2 * units.deg,
        sweeps=WingSweeps(quarter_chord=30.426 * units.deg),
        airfoil=ft.airfoils.ls417)

    tip_segment = WingSegment(
        name='Main Wing Tip Segment',
        percent_span_location=1.,
        root_chord_percent=0.192,
        thickness_to_chord=0.17,
        twist= -0.5 * units.deg,
        dihedral_outboard= -2 * units.deg,
        # sweeps=WingSweeps(quarter_chord= 30.4 * units.deg),
        airfoil=ft.airfoils.ls417)

    main_wing = Wing(
            name='Main Wing',
            aspect_ratio = 12.13,
            taper = 1.044/5.44,
            dihedral = -2 * units.deg,
            thickness_to_chord = 0.17,
            origin=jnp.array([[0, 0, 0]]),
            aerodynamic_center=jnp.array([0, 0, 0]),
            vertical=False,
            symmetric=True,
            high_lift=True,
            dynamic_pressure_ratio=1.0,
            sweeps=WingSweeps(quarter_chord=0. * units.deg),
            spans= WingDimensions(total = 50.53 *units.ft, projected=50.5*units.ft),
            chords=WingChords(root=5.4433*units.ft, tip=1.044*units.ft, mean_aerodynamic=4.37*units.ft),
            areas=Areas(reference=210.28 * units.ft**2,wetted=225.08 * units.ft**2),
            twists=WingDimensions(root=-0.5 * units.deg, tip=-0.5 * units.deg),
            # incidence_angle = -2 *units.deg,

            segments=(root_segment, mid_segment_1, mid_segment_2, mid_segment_3 ,tip_segment),
            # subcomponents=(slat, flap, aileron)
        ).update_geometry()

    return main_wing 

if __name__ == '__main__':
    airfoil=ft.airfoils.ls417
    plot_airfoil(airfoil, show_camber=True).show()
    
    wing = wing_setup()
    eta = jnp.array([s.percent_span_location for s in wing.segments])
    geometry = calculate_macro_properties(
        wing, eta, wing.spans.projected / 2
    )
    chord_grid = jnp.tile(jnp.linspace(0, 1, 9), (len(eta) - 1, 1))
    vertices = morph_to_3d_mesh(chord_grid, *geometry).reshape(-1, 4, 3)

    mirrored = vertices.at[:, :, 1].multiply(-1)
    vertices = jnp.concatenate([vertices, mirrored])
    vertices = vertices + jnp.asarray(wing.origin).reshape(3)

    plot_panels(LatticeData(panel_vertices=vertices)).show()