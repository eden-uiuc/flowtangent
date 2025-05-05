# RCAIDE/Framework/Missions/Update/forces.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Aug, 2024, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
# IMPORT 
# ----------------------------------------------------------------------------------------------------------------------

import RCAIDE.Framework as rcf

# ----------------------------------------------------------------------------------------------------------------------
# Update Forces
# ----------------------------------------------------------------------------------------------------------------------


def update_forces(
        state: "rcf.State",
        system: "rcf.System",
        settings: "rcf.Settings",
        ):
        
        wind    = state.frames.wind.total_force_vector
        thrust  = state.frames.body.thrust_force_vector
        gravity = state.frames.inertial.gravity_force_vector

        TB2I = state.frames.body.transform_to_inertial
        TW2I = state.frames.wind.transform_to_inertial

        F = TW2I.apply(wind)
        T = TB2I.apply(thrust)

        state.frames.inertial.total_force_vector = F + T + gravity
        
        return state, system, settings