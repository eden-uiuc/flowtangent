# flowtangent/Framework/Missions/Initialize/altitude_differential.py
# (c) Copyright 2025 Aerospace Research Community LLC#
# Created:  Sep 2025, J. Smart
# Modified:
# -------------------------------------------------------------------------------
#  Imports
# -------------------------------------------------------------------------------

# Package Imports

# FlowTangent Imports
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ... import Settings, State, System

import jax.numpy as jnp

# -------------------------------------------------------------------------------
#  Stateful/Framework Version
# -------------------------------------------------------------------------------


def initialize_altitude_differential(state: State, settings: Settings, system: System):
    """
    Framework version of initialize_altitude_differential

    See Also
    --------
    func_altitude_differential:
        Functional implementation which this method calls.
    """

    # Unpack state inputs
    t = state.time.dimensionless.control_points
    I = state.time.dimensionless.integrate
    r = state.frames.inertial.position_vector
    v = state.frames.inertial.velocity_vector

    # Get altitude and time step
    dz = r[-1, 2] - r[0, 2]
    dt = jnp.dot(I[-1, :] * dz, 1 / v[:, 2])

    # Rescale operator
    t = t * dt

    # Pack state outputes
    t_initial = state.frames.inertial.time[0, 0]
    state.frames.inertial.time[:, 0] = t_initial + t[:, 0]

    return state, settings, system
