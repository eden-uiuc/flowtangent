# $NAME.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: May 2025, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from dataclasses import dataclass, field
from typing import Callable

# package imports
import numpy as np

# RCAIDE imports
import RCAIDE.Framework as rcf

from .Segments import ConvergedSegment, OptimalSegment
# ----------------------------------------------------------------------------------------------------------------------
#  Cruise
# ----------------------------------------------------------------------------------------------------------------------


def energy_use(
        state: "rcf.State",
        system: "rcf.System",
        settings: "rcf.Settings"):

    energy_start    = state.energy.total_energy[0]
    energy_end      = state.energy.total_energy[-1]
    energy_used     = energy_end - energy_start

    return energy_used[0]


@dataclass(kw_only=True)
class EnergyOptimalCruise(OptimalSegment):

    name: str = 'Energy Optimal Cruise'

    altitude: float = 0.0
    distance: float = 0.0

    calculate_objective: Callable = energy_use

