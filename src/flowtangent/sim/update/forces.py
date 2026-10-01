# flowtangent/Framework/Missions/Update/forces.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Aug, 2024, FlowTangent Team

# ----------------------------------------------------------------------------------------------------------------------
# IMPORT
# ----------------------------------------------------------------------------------------------------------------------

# package imports
# FlowTangent imports
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ... import Settings, State, System

import jax.numpy as jnp

from ...utils import update

# ----------------------------------------------------------------------------------------------------------------------
# Update Forces
# ----------------------------------------------------------------------------------------------------------------------


def update_forces(
    state: State,
    system: System,
    settings: Settings,
):

    wind = state.frames.wind.total_force_vector
    thrust = state.frames.body.thrust_force_vector
    weight = state.frames.inertial.gravity_force_vector

    TB2I = state.frames.body.transform_to_inertial
    TW2I = state.frames.wind.transform_to_inertial

    wind_force = jnp.einsum("nij,nj->ni", TW2I, wind)
    thrust_force = jnp.einsum("nij,nj->ni", TB2I, thrust)

    total_force = weight + wind_force + thrust_force

    state = update(state, "frames.inertial.total_force_vector", total_force)

    return state, system, settings
