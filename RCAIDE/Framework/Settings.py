# RCAIDE/Framework/Settings.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Jul 2024, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from typing import Callable
import chex
from dataclasses import field

# package imports
import numpy as np
from scipy.optimize import fsolve

# RCAIDE imports

# ----------------------------------------------------------------------------------------------------------------------
#  Settings
# ----------------------------------------------------------------------------------------------------------------------


@chex.dataclass(kw_only=True)
class Settings:

    name: str = 'Settings'

    # Mission Settings
    root_finder: Callable = fsolve