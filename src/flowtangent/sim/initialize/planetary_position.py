# flowtangent/Framework/Missions/Initialization/planetary_position.py
# (c) Copyright 2024 Aerospace Research Community LLC
# Created: Aug 2024, FlowTangent Team

# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------

# package imports
# FlowTangent Imports
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ... import Settings, State, System
from ...utils import update

# ----------------------------------------------------------------------------------------------------------------------
# Initialize Planetary Position
# ----------------------------------------------------------------------------------------------------------------------


def initialize_planetary_position(
    state: State,
    system: System,
    settings: Settings,
):

    state = update(
        state,
        lambda s: (s.frames.planet.longitude, s.frames.planet.latitude),
        (
            state.frames.planet.longitude.at[:, 0].set(state.initials.frames.planet.longitude[-1, 0]),
            state.frames.planet.latitude.at[:, 0].set(state.initials.frames.planet.latitude[-1, 0]),
        ),
    )

    return state, system, settings
