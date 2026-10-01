# flowtangent/Framework/Missions/Update/angular_acceleration.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Aug 2024, FlowTangent Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
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
#  Update Angular Acceleration
# ----------------------------------------------------------------------------------------------------------------------


def update_angular_acceleration(
    state: State,
    system: System,
    settings: Settings,
):

    w = state.frames.inertial.angular_velocity_vector
    D = state.numerics.time.differentiate

    state = update(state, "frames.inertial.angular_acceleration_vector", jnp.dot(D, w))

    return state, system, settings
