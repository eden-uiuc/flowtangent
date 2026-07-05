# Trace/Framework/Missions/Update/acceleration.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Aug 2024, Trace Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

# package imports
import equinox as eqx
import jax.numpy as jnp

# Trace imports
import src.eden_trace.framework as rcf

# ----------------------------------------------------------------------------------------------------------------------
#  acceleration
# ----------------------------------------------------------------------------------------------------------------------


def update_acceleration(state: "rcf.state", system: "rcf.systems", settings: "rcf.settings"):

    v = state.frames.inertial.velocity_vector
    D = state.numerics.time.differentiate

    state = eqx.tree_at(lambda s: s.frames.inertial.acceleration_vector, state, jnp.dot(D, v))

    return state, system, settings
