# flowtangent/Library/Methods/Propulsors/moment.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: Apr 2025, FlowTangent Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

# package imports

# FlowTangent imports
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

import jax.numpy as np

from ... import Settings, State, System

# ----------------------------------------------------------------------------------------------------------------------
#  Turbofan Moment
# ----------------------------------------------------------------------------------------------------------------------


def func_propulsor_moment(
    propulsor_thrust: np.ndarray, propulsor_origin: np.ndarray, vehicle_center_of_gravity: np.ndarray
):

    moment_arm = propulsor_origin - vehicle_center_of_gravity
    moment = np.cross(moment_arm, propulsor_thrust)

    return moment


def propulsor_moment(
    state: State,
    system: System,
    settings: Settings,
):

    vehicle_center_of_gravity = system.mass_properties.center_of_gravity

    for idx, propulsor in enumerate(system.energy.propulsors):
        propulsor_thrust = state.energy.propulsors[idx].thrust
        propulsor_origin = system.energy.propulsors[idx].origin

        moment = func_propulsor_moment(propulsor_thrust, propulsor_origin, vehicle_center_of_gravity)

        state.energy.propulsors[idx].moment = moment

    return state, system, settings
