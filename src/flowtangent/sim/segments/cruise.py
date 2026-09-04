# $NAME.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: Oct 2025, Flowtangent Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from typing import TYPE_CHECKING

# package import
import equinox as eqx
import jax.numpy as jnp

if TYPE_CHECKING:
    from flowtangent.framework import Settings, State, System

from flowtangent.core._state_data._controls import Control, NamedResidual
from flowtangent.framework import ProcessStep
from flowtangent.framework.simulation.segments import Segment
from flowtangent.utils import field

from .profiles import *

# ----------------------------------------------------------------------------------------------------------------------
#  Cruise
# ----------------------------------------------------------------------------------------------------------------------


class Cruise(Segment):
    tag: str = "Cruise"

    distance: float = 0.0


# ----------------------------------------------------------------------------------------------------------------------
#  Test CSA Cruise
# ----------------------------------------------------------------------------------------------------------------------


def _test_cruise_controls():
    return (
        Control(
            tag="Lift Coefficient",
            state_path=("aerodynamics", "coefficients", "lift", "total"),
            path_indices=(slice(None), 0),
            _active=True,
        ),
        Control(
            tag="Drag Coefficient",
            state_path=("aerodynamics", "coefficients", "drag", "total"),
            path_indices=(slice(None), 0),
            _active=True,
        ),
    )


def _build_dynamics(
    altitude: float,
    distance: float,
    air_speed: float,
    sideslip: float,
):

    def initialize_dynamics(state: "State", system: "System", settings: "Settings"):
        alt = altitude
        xf = distance
        av = air_speed
        beta = sideslip

        if not av:
            av = jnp.linalg.norm(state.frames.inertial.velocity_vector[-1])
        if not alt:
            alt = -1.0 * state.frames.inertial.position_vector[-1, 2]

        v_x = jnp.cos(beta) * av
        v_y = jnp.sin(beta) * av

        t_0 = state.frames.inertial.time[0, 0]
        t_f = t_0 + xf / av

        t_nondim = state.time.dimensionless.control_points
        time = t_nondim * (t_f - t_0) + t_0

        new_vel = state.frames.inertial.velocity_vector.at[:, 0].set(v_x).at[:, 1].set(v_y)
        new_pos = state.frames.inertial.position_vector.at[:, 2].set(-alt)
        new_alt = state.freestream.altitude.at[:, 0].set(alt)

        new_state = eqx.tree_at(
            lambda s: (
                s.freestream.altitude,
                s.frames.inertial.position_vector,
                s.frames.inertial.velocity_vector,
                s.frames.inertial.time,
            ),
            state,
            (new_alt, new_pos, new_vel, time),
        )

        return new_state, system, settings

    return initialize_dynamics


class TestCSACruise(Cruise):
    tag: str = "Constant Speed & Altitude Cruise"

    altitude: float = 1.0
    air_speed: float = 1.0

    active_controls: tuple[str | Control, ...] = field(_test_cruise_controls)
    active_residuals: tuple[NamedResidual, ...] = field(("force_x", "force_z"), static=True)
    controls_initial_guess: tuple[jnp.ndarray | float, ...] = (1.0, 0.05)

    def __post_init__(self):

        super().__post_init__()

        # 2. Build the pure, detached physics function
        initialize_dynamics = _build_dynamics(self.altitude, self.distance, self.air_speed, self.sideslip_angle)

        # 3. Functionally append to the InitializeSegment
        new_init = self.initialize.append(ProcessStep(tag="Dynamics and Controls", function=initialize_dynamics))

        # 4. Overwrite the underlying tuple, NOT the `@property`
        new_steps = (new_init, self.iterate, self.finalize)
        object.__setattr__(self, "steps", new_steps)
