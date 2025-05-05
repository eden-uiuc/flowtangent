# RCAIDE/Framework/Missions/Update/moments.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Aug, 2024, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
# IMPORT 
# ----------------------------------------------------------------------------------------------------------------------

import RCAIDE.Framework as rcf

# ----------------------------------------------------------------------------------------------------------------------
# Update Moments
# ----------------------------------------------------------------------------------------------------------------------


def update_moments(
        state: "rcf.State",
        system: "rcf.System",
        settings: "rcf.Settings",
        ):

        wind    = state.frames.wind.total_moment_vector
        thrust  = state.energy.total_moment_vector

        TW2I    = state.frames.wind.transform_to_inertial

        M = TW2I.apply(wind)

        state.frames.inertial.total_moment = M + thrust
        
        return state, system, settings