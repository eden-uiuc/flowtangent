# RCAIDE/Framework/Analyses/Propulsion/Turbofan_Performance.py
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
# Turbofan_Performance
# ----------------------------------------------------------------------------------------------------------------------

@dataclass(kw_only=True)
class Turbofan_Performance():

    name: str = 'Turbofan_Performance'

def Turbofan_Performance(
        state: rcf.State,
        system: rcf.System,
        settings: rcf.Settings
        ):
        
        
        
        return state, system, settings