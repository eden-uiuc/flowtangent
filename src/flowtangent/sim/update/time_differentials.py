# flowtangent/Framework/Missions/Update/time_differentials.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Aug 2024, FlowTangent Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ... import Settings, State, System

# package imports
# FlowTangent imports
from ...utils import update

# ----------------------------------------------------------------------------------------------------------------------
#  Update Time Differentials
# ----------------------------------------------------------------------------------------------------------------------


def update_time_differentials(
    state: State,
    system: System,
    settings: Settings,
):

    x = state.time.dimensionless.control_points
    D = state.time.dimensionless.differentiate
    I = state.time.dimensionless.integrate

    time = state.frames.inertial.time
    T = time[-1] - time[0]
    t_scaled = x * T
    D_scaled = D / T
    I_scaled = I * T

    state = update(
        state,
        (
            ("numerics.time.control_points", t_scaled),
            ("numerics.time.differentiate", D_scaled),
            ("numerics.time.integrate", I_scaled),
        ),
        is_leaf=lambda x: x is None,
    )

    return state, system, settings
