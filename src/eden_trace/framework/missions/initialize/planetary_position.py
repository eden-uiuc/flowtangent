# Trace/Framework/Missions/Initialization/planetary_position.py
# (c) Copyright 2024 Aerospace Research Community LLC
# Created: Aug 2024, Trace Team

# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------

# package imports
import equinox as eqx

# Trace Imports
import src.eden_trace.framework as rcf

# ----------------------------------------------------------------------------------------------------------------------
# Initialize Planetary Position
# ----------------------------------------------------------------------------------------------------------------------


def initialize_planetary_position(
    state: "rcf.state",
    system: "rcf.systems",
    settings: "rcf.settings",
):

    state = eqx.tree_at(
        lambda s: (s.frames.planet.longitude, s.frames.planet.latitude),
        state,
        (
            state.frames.planet.longitude.at[:, 0].set(state.initials.frames.planet.longitude[-1, 0]),
            state.frames.planet.latitude.at[:, 0].set(state.initials.frames.planet.latitude[-1, 0]),
        ),
    )

    return state, system, settings
