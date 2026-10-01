# flowtangent/Framework/Missions/residuals.py
# (c) Copyright 2025 Aerospace Research Community LLC#
# Created:  Sep 2025, J. Smart
# Modified:
# -------------------------------------------------------------------------------
#  Imports
# -------------------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flowtangent.framework import Settings, State, System

# package imports
import equinox as eqx

from ...utils import update

# FlowTangent Imports

# -------------------------------------------------------------------------------
#  Stateful/Framework Version
# -------------------------------------------------------------------------------


def flight_dynamics_residuals(
    state: State,
    system: System,
    settings: Settings,
):
    """
    Calculates the residuals from the flight dynamics equations.
    """

    active_residuals = state.dynamics.get_active_residuals()
    force_residuals = [res for res in active_residuals if "force" in res.name]
    moment_residuals = [res for res in active_residuals if "moment" in res.name]

    FT = state.frames.inertial.total_force_vector
    MT = state.frames.inertial.total_moment_vector
    a = state.frames.inertial.acceleration_vector
    wdot = state.frames.inertial.angular_acceleration_vector

    m = state.mass.total
    I = state.mass.moments_of_inertia

    for force_res in force_residuals:
        force_res = update(force_res, "value", FT[:, force_res.index] / m[:, 0] - a[:, force_res.index])
        state = update(state, lambda s: getattr(s.dynamics, force_res.name), force_res)
    for moment_res in moment_residuals:
        if I[moment_res.index, moment_res.index] == 0:
            raise ValueError(
                f"Moment of Inertia Matrix must be defined for residual: {moment_res.name} at "
                f"I[{moment_res.index}, {moment_res.index}]"
            )
        moment_res = eqx.tree_at(
            lambda m: m.value,
            moment_res,
            MT[:, moment_res.index] / I[moment_res.index, moment_res.index] - wdot[:, moment_res.index],
        )
        state = update(state, lambda s: getattr(s.dynamics, moment_res.name), moment_res)

    return state, system, settings
