# RCAIDE/Framework/Missions/Update/altitude.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Aug 2024, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

# package imports
import equinox as eqx

# RCAIDE imports
import RCAIDE.Framework as rcf

# ----------------------------------------------------------------------------------------------------------------------
#  Update Altitude
# ----------------------------------------------------------------------------------------------------------------------


def update_altitude(state: "rcf.State",
                    system: "rcf.System",
                    settings: "rcf.Settings",
                    ):

    state = eqx.tree_at(lambda s:s.freestream.altitude, state, state.frames.inertial.position_vector[:, 2])
                   
    return state, system, settings