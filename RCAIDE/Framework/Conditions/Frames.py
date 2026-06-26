# RCAIDE/Framework/Missions/Conditions/Frames.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Jul 2024, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------



# package imports
import jax.numpy as jnp

# RCAIDE imports
from RCAIDE.utils import empty_array, init_field

from RCAIDE.Framework.Conditions import Conditions

# ----------------------------------------------------------------------------------------------------------------------
#  Frames
# ----------------------------------------------------------------------------------------------------------------------

class Frame(Conditions):

    # Attribute             Type        Default Value
    tag:                    str         = init_field('Frame', static=True)

    transform_to_inertial:  jnp.ndarray = empty_array((0, 3))

    total_force_vector:     jnp.ndarray = empty_array((0, 3))
    total_moment_vector:    jnp.ndarray = empty_array((0, 3))


class InertialFrame(Frame):

    # Attribute                     Type        Default Value
    tag:                            str         = init_field('Inertial Frame', static=True)

    position_vector:                jnp.ndarray  = empty_array((0, 3))

    velocity_vector:                jnp.ndarray  = empty_array((0, 3))
    acceleration_vector:            jnp.ndarray  = empty_array((0, 3))

    angular_velocity_vector:        jnp.ndarray  = empty_array((0, 3))
    angular_acceleration_vector:    jnp.ndarray  = empty_array((0, 3))

    gravity_force_vector:           jnp.ndarray  = empty_array((0, 3))

    time:                           jnp.ndarray  = empty_array((0))
    system_range:                   jnp.ndarray  = empty_array((0))


class BodyFrame(Frame):

    # Attribute             Type        Default Value
    tag:                    str         = init_field('Body Frame', static=True)

    inertial_rotations:     jnp.ndarray  = empty_array((0, 3))
    thrust_force_vector:    jnp.ndarray  = empty_array((0, 3))
    moment_vector:          jnp.ndarray  = empty_array((0, 3))

class WindFrame(Frame):

    # Attribute         Type            Default Value
    tag:                str             = init_field('Wind Frame', static=True)

    body_rotations:     jnp.ndarray      = empty_array((0, 3))
    transform_to_body:  jnp.ndarray      = empty_array((0, 3))

    velocity_vector:    jnp.ndarray      = empty_array((0, 3))
    force_vector:       jnp.ndarray      = empty_array((0, 3))
    moment_vector:      jnp.ndarray      = empty_array((0, 3))


class PlanetFrame(Frame):

    # Attribute     Type            Default Value
    tag:            str             = init_field('Planet Frame', static=True)
    start_time:     jnp.ndarray     = empty_array(0)

    # Default to takeoff at JFK
    latitude:       jnp.ndarray     = init_field(lambda: jnp.array([40.6446]))
    longitude:      jnp.ndarray     = init_field(lambda: jnp.array([73.7797]))

    true_course:    jnp.ndarray     = empty_array(0)


class FrameConditions(Conditions):

    # Attribute     Type            Default Value
    tag:            str             = init_field('Dynamic Frames', static=True)

    inertial:       InertialFrame   = init_field(InertialFrame)
    body:           BodyFrame       = init_field(BodyFrame)
    wind:           WindFrame       = init_field(WindFrame)
    planet:         PlanetFrame     = init_field(PlanetFrame)
