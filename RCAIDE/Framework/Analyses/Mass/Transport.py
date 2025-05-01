# RCAIDE/Framework/Analyses/Mass/Transport.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: May, 2025, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
# IMPORT 
# ----------------------------------------------------------------------------------------------------------------------

from dataclasses import dataclass, field, make_dataclass

import RCAIDE.Framework as rcf
import RCAIDE.Library as rcl

# ----------------------------------------------------------------------------------------------------------------------
# Transport
# ----------------------------------------------------------------------------------------------------------------------

@dataclass(kw_only=True)
class Transport():

    name: str = 'Transport'

def Transport(
        state: rcf.State,
        system: rcf.System
        settings: rcf.Settings,
        ):
        
        
        
        return state, system, settings