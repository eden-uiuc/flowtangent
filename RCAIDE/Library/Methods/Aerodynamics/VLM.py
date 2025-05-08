# RCAIDE/Library/Methods/Aerodynamics/VLM.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: May 2025, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from dataclasses import dataclass, field

# package imports
import numpy as np

# RCAIDE imports
import RCAIDE.Framework as rcf

# ----------------------------------------------------------------------------------------------------------------------
#  VLM
# ----------------------------------------------------------------------------------------------------------------------


def VLM(state: rcf.State,
            settings: rcf.Settings,
            system: rcf.System):


                   
    return state, settings, system