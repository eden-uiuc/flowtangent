# RCAIDE/Framework/Missions/Update/altitude.py
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
#  Update Altitude
# ----------------------------------------------------------------------------------------------------------------------


def update_altitude(state: "rcf.State",
                    system: "rcf.System",
                    settings: "rcf.Settings",
                    ):

    state.freestream.altitude = state.frames.inertial.position_vector[:, 2]
                   
    return state, system, settings