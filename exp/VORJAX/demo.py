from dataclasses import replace

import flowtangent as ft
import jax.numpy as jnp

from flowtangent import units
from flowtangent.components import Wing, WingSegment
from flowtangent.plots import plot_panels

from flowtangent.solve.aero._vorjax import update_mesh, VORJAXSettings, PanelSettings

if __name__ == "__main__":

    AR = 8.0
    n_seg = 10

    span = AR * units.m
    root_chord = 4.0 * span / (jnp.pi * AR)
    S_ref = span ** 2 / AR

    def chord_frac(eta):
        return jnp.sqrt(1.0 - jnp.clip(eta, 0.0, 0.99999) **2)

    segments = ()
    eta = jnp.cos(jnp.linspace(jnp.pi/2, 0, n_seg + 1))

    for i in range(n_seg):
        eta_start   = eta[i]
        eta_end     = eta[i+1]

        chord_start = chord_frac(eta_start)
        chord_end   = chord_frac(eta_end)

        delta_x_c4 = 0.25 * 1.0 * (chord_start - chord_end)
        delta_y = 0.5 * AR * (eta_end - eta_start)
        sweep_c4 = jnp.arctan2(delta_x_c4, delta_y)

        new_seg = WingSegment(
                        name=f"Elliptical Segment {i}",
                        percent_span_location=eta_start.item(),
                        root_chord_percent=chord_start.item(),
                    )
        new_seg = ft.update(new_seg, ("sweeps.quarter_chord", sweep_c4.item()))
        segments += (new_seg,)

    # Instantiate wing
    wing = Wing(
        name=f"Main Wing",
        symmetric=True,
        segments=segments,
        taper=0.01,
        aerodynamic_center=jnp.zeros(3)
    )

    # Update nested properties and geometry
    wing = ft.update(wing, (
        (("spans", "projected"), span),
        (("chords", "root"), root_chord),
        (("chords", "mean_aerodynamic"), root_chord * 8.0 / (3 * jnp.pi)),
        (("areas", "reference"), S_ref),
        (("areas", "wetted"), (2.0 * S_ref)),
    )).update_geometry()

    system = ft.update(ft.Aircraft(name="VORJAX Model", subcomponents=(wing,)), "areas", wing.areas)

    # initial_state = ft.State().freeze_initials()
    initial_state = ft.update(
        ft.State(),
        (
            ("stability.static.roll_rate", jnp.zeros((1, 1))),
            ("stability.static.pitch_rate", jnp.zeros((1, 1))),
            ("stability.static.yaw_rate", jnp.zeros((1, 1))),
        )
    )

    initial_state = ft.update(initial_state, "aerodynamics.angles.alpha", jnp.atleast_2d(3.0 * units.deg))
    initial_state = ft.update(initial_state, "aerodynamics.angles.beta", jnp.atleast_2d(0.0 * units.deg))
    initial_state = ft.update(initial_state, "freestream.mach_number", jnp.atleast_2d(0.0))

    # initial_state = update(initial_state, "freestream.speed", jnp.array([[100.0]]))
    initial_state = ft.update(initial_state, "freestream.density", jnp.array([[1.0]]))
    initial_state = ft.update(initial_state, "freestream.gamma", jnp.array([[1.4]]))
    initial_state = ft.update(initial_state, "freestream.temperature", jnp.array([[273.15]]))
    initial_state = ft.update(initial_state, "frames.inertial.velocity_vector", jnp.array([[100.0, 0., 0.]]))

    initialize = ft.solve.aero.InitializeVORJAX()

    span_path  = ft.TreePath(("wings", "main_wing", "spans", "projected"), name="b")
    vert_path   = ft.TreePath(path=("analysis_data", "vortex_distribution", "panel_vertices"), path_slice=(slice(100), slice(1), slice(1)), name="v")
    
    jac_map = ft.JacobianMap(system_inputs=(span_path,), system_outputs=(vert_path,))

    aero_settings = VORJAXSettings(panels=PanelSettings(n_spanwise=24, n_chordwise=8))
    numerical_settings = ft.solve.NumericalSettings(jacobian=ft.solve.JacobianSettings(mapping=jac_map, calculate=True))
    settings = ft.Settings(DEBUG_MODE=False, analysis=ft.solve.AnalysisSettings(aerodynamics=aero_settings), numerical=numerical_settings)
    ft.configure_environment(settings)

    init_state, init_system, init_settings = initialize.run(initial_state, system, settings)

    print(init_system.analysis_data['vortex_distribution'].panel_vertices[100,1,1])

    update_proc = ft.Process(steps=(update_mesh,))
    up_state, up_system, up_settings = update_proc.run(init_state, init_system, init_settings)

    print(up_system.analysis_data['vortex_distribution'].panel_vertices[100,1,1])

    # vd = init_system.analysis_data['vortex_distribution']
    # fig = plot_panels(vd)
    # fig.show()

    print(up_state.process_jacobian.shape)

    print("Done.")