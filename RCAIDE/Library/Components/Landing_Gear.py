# RCAIDE/Library/Components/Landing_Gear.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: May, 2025, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
# IMPORT 
# ----------------------------------------------------------------------------------------------------------------------

from dataclassess import dataclass, field, make_dataclass

import RCAIDE.Framework as rcf
import RCAIDE.Library as rcl

# ----------------------------------------------------------------------------------------------------------------------
# Landing_Gear
# ----------------------------------------------------------------------------------------------------------------------

@dataclass(kw_only=True)
class Landing_Gear():

    name: str = 'Landing_Gear'

def Landing_Gear(
        state: rcf.State,
        system: rcf.System
        settings: rcf.Settings,
        ):
        
        
        
        return state, system, settings