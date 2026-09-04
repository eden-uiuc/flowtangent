# flowtangent/Framework/Missions/Conditions/Frames.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Jul 2024, Flowtangent Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------


# package imports
import jax.numpy as jnp

# Flowtangent imports
from flowtangent.utils import empty_array, field, register

from flowtangent.framework.state_data import StateData

# ----------------------------------------------------------------------------------------------------------------------
#  Frames
# ----------------------------------------------------------------------------------------------------------------------

@register
class Frame(StateData):
    # Attribute             Type        Default Value
    tag: str = field("Frame", static=True)

    transform_to_inertial: jnp.ndarray = empty_array((0, 3))

    total_force_vector: jnp.ndarray = empty_array((0, 3))
    total_moment_vector: jnp.ndarray = empty_array((0, 3))

@register
class Inertial(Frame):
    # Attribute                     Type        Default Value
    tag: str = field("Inertial Frame", static=True)

    position_vector: jnp.ndarray = empty_array((0, 3))

    velocity_vector: jnp.ndarray = empty_array((0, 3))
    acceleration_vector: jnp.ndarray = empty_array((0, 3))

    angular_velocity_vector: jnp.ndarray = empty_array((0, 3))
    angular_acceleration_vector: jnp.ndarray = empty_array((0, 3))

    gravity_force_vector: jnp.ndarray = empty_array((0, 3))

    time: jnp.ndarray = empty_array((0))
    system_range: jnp.ndarray = empty_array((0))

@register
class Body(Frame):
    # Attribute             Type        Default Value
    tag: str = field("Body Frame", static=True)

    inertial_rotations: jnp.ndarray = empty_array((0, 3))
    thrust_force_vector: jnp.ndarray = empty_array((0, 3))
    moment_vector: jnp.ndarray = empty_array((0, 3))

@register
class Wind(Frame):
    # Attribute         Type            Default Value
    tag: str = field("Wind Frame", static=True)

    body_rotations: jnp.ndarray = empty_array((0, 3))
    transform_to_body: jnp.ndarray = empty_array((0, 3))

    velocity_vector: jnp.ndarray = empty_array((0, 3))
    force_vector: jnp.ndarray = empty_array((0, 3))
    moment_vector: jnp.ndarray = empty_array((0, 3))

@register
class Planet(Frame):
    # Attribute     Type            Default Value
    tag: str = field("Planet Frame", static=True)
    start_time: jnp.ndarray = empty_array()

    # Default to takeoff at JFK
    latitude: jnp.ndarray = field(lambda: jnp.array([40.6446]))
    longitude: jnp.ndarray = field(lambda: jnp.array([73.7797]))

    true_course: jnp.ndarray = empty_array()

@register
class FrameData(StateData):
    # Attribute     Type            Default Value
    tag: str = field("Dynamic Frames", static=True)

    inertial: Inertial = field(Inertial)
    body: Body = field(Body)
    wind: Wind = field(Wind)
    planet: Planet = field(Planet)
