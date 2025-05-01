# RCAIDE/Library/Propellants.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: Apr 2025, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from dataclasses import dataclass, field

# package imports
import numpy as np

# RCAIDE imports
import RCAIDE.Framework as rcf

# ----------------------------------------------------------------------------------------------------------------------
#  Propellants
# ----------------------------------------------------------------------------------------------------------------------


def Propellants(state: rcf.State,
            settings: rcf.Settings,
            system: rcf.System):


                   
    return state, settings, system