# flowtangent/Framework/Missions/Initialization/time.py
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

import jax.numpy as jnp

from ...utils import update

# ----------------------------------------------------------------------------------------------------------------------
# Initialize Time
# ----------------------------------------------------------------------------------------------------------------------


def initialize_time(state: State, system: System, settings: Settings):

    t_initial = state.initials.frames.inertial.time
    if t_initial is None:
        t_initial = jnp.atleast_2d(state.frames.planet.start_time)

    t_current = state.frames.inertial.time

    # Use explicit positive indexing to avoid JAX dynamic shape issues with -1
    last_idx = int(state.time.dimensionless.number_of_control_points) - 1
    delta_t = t_initial[last_idx, 0] - t_current[0, 0]
    offset_time = t_current + delta_t

    state = update(state, lambda s: (s.frames.planet.start_time, s.frames.inertial.time), (t_initial, offset_time))

    return state, system, settings
