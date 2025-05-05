# RCAIDE/Library/Methods/Mass/Correlation/Transport/landing_gear.py
# (c) Copyright 2024 Aerospace Research Community LLC#
# Created:  May 2024, J. Smart
# Modified: 
# -------------------------------------------------------------------------------
#  Imports
# -------------------------------------------------------------------------------

import RCAIDE.Framework as rcf

# -------------------------------------------------------------------------------
#  Functional/Library Version
# -------------------------------------------------------------------------------

def func_landing_gear(
    MTOW: float,
    lg_wt_factor: float = 0.04):

    return MTOW * lg_wt_factor


# -------------------------------------------------------------------------------
#  Stateful/Framework Version
# -------------------------------------------------------------------------------

def landing_gear(state: "rcf.State",
                 system: "rcf.System",
                 settings: "rcf.Settings"):
    """
    Framework version of landing_gear
    
    See Also
    --------
    func_landing_gear: 
        Functional implementation which this method calls.
    """

    # TODO: Unpack functional arguments

    results = func_landing_gear(*args,
                                **kwargs
                                )

    # TODO: Unpack results

    return state, system, settings
