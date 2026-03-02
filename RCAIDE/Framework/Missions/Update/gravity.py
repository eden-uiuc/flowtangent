# RCAIDE/Framework/Missions/gravity.py
# (c) Copyright 2026 Aerospace Research Community LLC#
# Created:  Jan 2026, E. Botero
# Modified: 
# -------------------------------------------------------------------------------
#  Imports
# -------------------------------------------------------------------------------

# package imports
import equinox as eqx

# RCAIDE Imports

import RCAIDE.Framework as rcf


# -------------------------------------------------------------------------------
#  Stateful/Framework Version
# -------------------------------------------------------------------------------


def update_gravity(
    state: "rcf.State",
    system: "rcf.System",
    settings: "rcf.Settings",
):
    """
    Updates the current gravity as applied to the system
    """

    
    state = eqx.tree_at(lambda s:s.freestream.gravity, state, state.freestream.gravity.at[:,0].set(-9.81))


    return state, system, settings