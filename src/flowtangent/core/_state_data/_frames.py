# flowtangent/Framework/Missions/Conditions/Frames.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Jul 2024, FlowTangent Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------


# package imports
import jax.numpy as jnp

from flowtangent.core._state_data import StateData

# FlowTangent imports
from ...utils import field
from ...utils.typing import TimeScalar, TimeVector3, _

# ----------------------------------------------------------------------------------------------------------------------
#  Frames
# ----------------------------------------------------------------------------------------------------------------------


class Frame(StateData):
    transform_to_inertial: TimeVector3 = _

    total_force_vector: TimeVector3 = _
    total_moment_vector: TimeVector3 = _


class Inertial(Frame):
    position_vector: TimeVector3 = _

    velocity_vector: TimeVector3 = _
    acceleration_vector: TimeVector3 = _

    angular_velocity_vector: TimeVector3 = _
    angular_acceleration_vector: TimeVector3 = _

    gravity_force_vector: TimeVector3 = _

    time: TimeScalar = _
    system_range: TimeScalar = _


class Body(Frame):
    inertial_rotations: TimeVector3 = _
    thrust_force_vector: TimeVector3 = _
    moment_vector: TimeVector3 = _


class Wind(Frame):
    body_rotations: TimeVector3 = _
    transform_to_body: TimeVector3 = _

    velocity_vector: TimeVector3 = _
    force_vector: TimeVector3 = _
    moment_vector: TimeVector3 = _


class Planetary(Frame):
    start_time: TimeScalar = _

    # Default to takeoff at JFK
    latitude: TimeScalar = field(jnp.array([40.6446]))
    longitude: TimeScalar = field(jnp.array([73.7797]))

    true_course: TimeScalar = _


class FrameData(StateData):
    inertial: Inertial = field(Inertial)
    body: Body = field(Body)
    wind: Wind = field(Wind)
    planet: Planetary = field(Planetary)
