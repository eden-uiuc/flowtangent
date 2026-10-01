# flowtangent/Framework/Missions/Update/moments.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Aug, 2024, FlowTangent Team

# ----------------------------------------------------------------------------------------------------------------------
# IMPORT
# ----------------------------------------------------------------------------------------------------------------------

# package imports
# FlowTangent Imports
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ... import Settings, State, System

import jax.numpy as jnp

from ...utils import update

# ----------------------------------------------------------------------------------------------------------------------
# Update Moments
# ----------------------------------------------------------------------------------------------------------------------


def update_moments(
    state: State,
    system: System,
    settings: Settings,
):

    wind = state.frames.wind.total_moment_vector
    thrust = state.energy.total_moment_vector

    TW2I = state.frames.wind.transform_to_inertial

    M = jnp.einsum("nij,nj->ni", TW2I, wind)

    state = update(state, "frames.inertial.total_moment_vector", M + thrust)

    return state, system, settings
