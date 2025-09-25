# RCAIDE/Framework/Missions/residuals.py
# (c) Copyright 2025 Aerospace Research Community LLC#
# Created:  Sep 2025, J. Smart
# Modified: 
# -------------------------------------------------------------------------------
#  Imports
# -------------------------------------------------------------------------------

# Package Imports

import inspect
import numpy as np

# RCAIDE Imports

import RCAIDE.Library as rcl
import RCAIDE.Framework as rcf


# -------------------------------------------------------------------------------
#  Stateful/Framework Version
# -------------------------------------------------------------------------------

def get_active_residuals(
    state: "rcf.State"
):
    dyn = state.controls.dynamics

    possible_residuals = [v for v in vars(dyn).values() if isinstance(v, rcf.Missions.Residual)]
    active_residuals = [res for res in possible_residuals if res.active]

    return active_residuals


def flight_dynamics_residuals(
    unknowns,
    state: "rcf.State",
    settings: "rcf.Settings",
    system: "rcf.System"
):
    """
    Calculates the residuals from the flight dynamics equations.
    """
    state.unknowns.unpack_array(unknowns)

    active_residuals = get_active_residuals(state)
    force_residuals = [res for res in active_residuals if res.type == 'force']
    moment_residuals = [res for res in active_residuals if res.type == 'moment']

    FT      = state.frames.inertial.total_force_vector
    MT      = state.frames.inertial.total_moment_vector
    a       = state.frames.inertial.acceleration_vector
    wdot    = state.frames.inertial.angular_acceleration_vector

    m   = state.mass.total
    I   = state.mass.moments_of_inertia

    for force_res in force_residuals:
        force_res.value = FT[:, force_res.index] / m[:, 0] - a[:, force_res.index]
    for moment_res in moment_residuals:
        if I[moment_res.index, moment_res.index] == 0:
            raise ValueError(f"Moment of Inertia Matrix must be defined for residual: {moment_res.name} at "
                             f"I[{moment_res.index}, {moment_res.index}]")
        moment_res.value = MT[:, moment_res.index] / I[moment_res.index, moment_res.index] - wdot[:, moment_res.index]

    return state, settings, system