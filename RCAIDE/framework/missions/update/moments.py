# RCAIDE/Framework/Missions/Update/moments.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Aug, 2024, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
# IMPORT
# ----------------------------------------------------------------------------------------------------------------------

# package imports
import equinox as eqx
import jax.numpy as jnp

# RCAIDE Imports
import RCAIDE.framework as rcf

# ----------------------------------------------------------------------------------------------------------------------
# Update Moments
# ----------------------------------------------------------------------------------------------------------------------


def update_moments(
    state: "rcf.state",
    system: "rcf.systems",
    settings: "rcf.settings",
):

    wind = state.frames.wind.total_moment_vector
    thrust = state.energy.total_moment_vector

    TW2I = state.frames.wind.transform_to_inertial

    M = jnp.einsum("nij,nj->ni", TW2I, wind)

    state = eqx.tree_at(lambda s: s.frames.inertial.total_moment_vector, state, M + thrust)

    return state, system, settings
