# RCAIDE/Framework/Missions/Segments/Climb.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: Sep, 2025, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
# IMPORT 
# ----------------------------------------------------------------------------------------------------------------------

import chex

from dataclasses import field, make_dataclass

import RCAIDE.Framework as rcf
import RCAIDE.Library as rcl

# ----------------------------------------------------------------------------------------------------------------------
# Climb
# ----------------------------------------------------------------------------------------------------------------------


@chex.dataclass(kw_only=True)
class ConvergedClimb(rcf.Missions.ConvergedSegment):

    name: str = 'Climb'

    altitude_start: float = 0.0
    altitude_end:   float = 0.0

    def __post_init__(self):
        self.state.controls.throttle.active = True
