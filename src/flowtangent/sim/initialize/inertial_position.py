# flowtangent/Framework/Missions/Initialization/inertial_position.py
# (c) Copyright 2024 Aerospace Research Community LLC
# Created: Aug 2024, FlowTangent Team

# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------

# package imports
# ----------------------------------------------------------------------------------------------------------------------
# Initialize Inertial Position
# ----------------------------------------------------------------------------------------------------------------------
# FlowTangent Imports
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ... import Settings, State, System

from ...utils import update


def initialize_inertial_position(
    state: State,
    system: System,
    settings: Settings,
):

    # Extract current arrays
    p_initial = state.initials.frames.inertial.position_vector
    p_current = state.frames.inertial.position_vector

    # Calculate deltas
    p_initial = p_initial.at[-1, None, -1].set(-state.initials.freestream.altitude[-1, 0])
    delta_p = p_initial[-1, None, :] - p_current[0, None, :]

    R_initial = state.initials.frames.inertial.system_range
    R_current = state.frames.inertial.system_range
    delta_R = R_initial[-1, None, :] - R_current[0, None, :]

    # Calculate the new values
    new_position_vector = p_current + delta_p
    new_system_range = R_current + delta_R

    state = update(
        state,
        (
            ("frames.inertial.position_vector", new_position_vector),
            ("frames.inertial.system_range", new_system_range),
        ),
    )

    return state, system, settings
