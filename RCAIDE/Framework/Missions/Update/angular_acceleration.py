# RCAIDE/Framework/Missions/Update/angular_acceleration.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Aug 2024, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from dataclasses import dataclass, field

# package imports
import numpy as np

# RCAIDE imports
import RCAIDE.Framework as rcf

# ----------------------------------------------------------------------------------------------------------------------
#  Update Angular Acceleration
# ----------------------------------------------------------------------------------------------------------------------


def update_angular_acceleration(state: "rcf.State",
                                system: "rcf.System",
                                settings: "rcf.Settings",
                                ):

    w = state.frames.inertial.angular_velocity_vector
    D = state.numerics.time.differentiate

    aa = np.dot(D, w)

    state.frames.inertial.angular_acceleration = aa
                   
    return state, system, settings