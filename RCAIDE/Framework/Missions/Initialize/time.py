# RCAIDE/Framework/Missions/Initialization/time.py
# (c) Copyright 2024 Aerospace Research Community LLC
# Created: Aug 2024, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------

# package imports
#import numpy as np
import jax.numpy as np

# RCAIDE Imports
import RCAIDE.Framework as rcf

# ----------------------------------------------------------------------------------------------------------------------
# Initialize Time
# ----------------------------------------------------------------------------------------------------------------------


def initialize_time(state: "rcf.State",
                    system: "rcf.System",
                    settings: "rcf.Settings"
                    ):

    t_initial = state.initials.frames.inertial.time
    if t_initial is None:
        t_initial = np.atleast_2d(state.frames.planet.start_time)

    t_current = state.frames.inertial.time

    # Use explicit positive indexing to avoid JAX dynamic shape issues with -1
    last_idx = int(state.numerics.number_of_control_points) - 1
    delta_t     = t_initial[last_idx, 0] - t_current[0, 0]
    offset_time = t_current + delta_t

    state.frames.planet.start_time  = t_initial
    state.frames.inertial.time      = offset_time

    return state, system, settings


