import equinox as eqx
import jax.numpy as jnp

from flowtangent.framework.analyses.aero.VORJAX import InitializeVORJAX, VORJAX, Vortices, VORJAX_Settings
from flowtangent.library.components.wings import Wing, WingSegment, Sweeps, Chords, WingDimensions
from flowtangent.core._systems import Aircraft
from flowtangent.library.components import Areas
from flowtangent.data import units

from flowtangent.utils import configure_environment, DataPath

from flowtangent.framework import State, Settings, Process
from flowtangent.core._settings import JacobianMap, NumericalSettings

from flowtangent.framework.plotting.VLM import plot_vlm_panels

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

        sweeps = Sweeps(quarter_chord=sweep_c4.item())

        segments += (
            WingSegment(
                tag=f"Elliptical Segment {i}",
                percent_span_location=eta_start,
                root_chord_percent=chord_start,
                sweeps=sweeps,
            ),
        )

    segments += (WingSegment(
        tag="Tip",
        percent_span_location=1.0,
        root_chord_percent=0.01),)

    wing_areas = Areas(reference=S_ref, wetted=2.0 * S_ref)

    wing = Wing(
        tag=f"Elliptical {n_seg}",
        symmetric=True,
        spans=WingDimensions(projected=span),
        segments=segments,
        chords=Chords(
            root=root_chord,
            tip = 0.01 * root_chord,
            mean_aerodynamic=root_chord * 8.0 / (3 * jnp.pi)),
            areas = wing_areas,
            taper=0.01,
            aerodynamic_center=jnp.zeros(3)
        ).update_geometry()

    system = Aircraft(
        tag="VORJAX Model",
        areas=wing_areas,
        subcomponents=(wing,)
    )

    initial_state = State().freeze_initials()
    initial_state = eqx.tree_at(
        lambda s: (
            s.stability.static.roll_rate,
            s.stability.static.pitch_rate,
            s.stability.static.yaw_rate),
        initial_state,
        (
            jnp.zeros((1, 1)),
            jnp.zeros((1, 1)),
            jnp.zeros((1, 1))
        )
    )

    initial_state = eqx.tree_at(lambda s: s.aerodynamics.angles.alpha, initial_state, jnp.atleast_2d(3.0 * units.deg))
    initial_state = eqx.tree_at(lambda s: s.aerodynamics.angles.beta, initial_state, jnp.atleast_2d(0.0 * units.deg))
    initial_state = eqx.tree_at(lambda s: s.freestream.mach_number, initial_state, jnp.atleast_2d(0.0))

    # initial_state = eqx.tree_at(lambda s: s.freestream.speed, initial_state, jnp.array([[100.0]]))
    initial_state = eqx.tree_at(lambda s: s.freestream.density, initial_state, jnp.array([[1.0]]))
    initial_state = eqx.tree_at(lambda s: s.freestream.gamma, initial_state, jnp.array([[1.4]]))
    initial_state = eqx.tree_at(lambda s: s.freestream.temperature, initial_state, jnp.array([[273.15]]))
    initial_state = eqx.tree_at(lambda s: s.frames.inertial.velocity_vector, initial_state, jnp.array([[100.0, 0., 0.]]))

    analysis = VORJAX()

    panelization = Vortices(
        n_spanwise=24,
        n_chordwise=48
    )

    alpha_path  = DataPath(("aerodynamics", "angles", "alpha"), tag="a")
    lift_path   = DataPath(("aerodynamics", "coefficients", "lift", "total"), tag="CL")
    
    jac_map = JacobianMap(state_inputs=(alpha_path,), state_outputs=(lift_path,))
    num_sets = NumericalSettings(jacobian_map=jac_map)

    settings = eqx.tree_at(
        lambda s: (
            s.analysis.aerodynamics,
            s.numerical
        ),
        Settings(DEBUG_MODE=True,),
        (
            VORJAX_Settings(vortices=panelization),
            num_sets
        )
    )
    configure_environment(settings)

    init_state, init_system, init_settings = analysis.run(initial_state, system, settings)
    vd = init_system.analysis_data['vortex_distribution']
    fig = plot_vlm_panels(vd)
    fig.show()

    print("Done!")