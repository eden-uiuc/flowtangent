# RCAIDE/Framework/Missions/Initialization/expand_state.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Aug 2024, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

# RCAIDE imports
import RCAIDE.Framework as rcf

# ----------------------------------------------------------------------------------------------------------------------
#  Expand State
# ----------------------------------------------------------------------------------------------------------------------


def expand_state(state: "rcf.State",
                 system: "rcf.System",
                 settings: "rcf.Settings",
                 ):

    state.expand_rows(rows=state.numerics.number_of_control_points)

    return state, system, settings
