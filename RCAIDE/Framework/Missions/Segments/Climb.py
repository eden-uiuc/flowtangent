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
        super().__post_init__()
        self.state.controls.throttle.active = True

        self._initialize.append(rcf.ProcessStep(name='Altitude Differential'))


@chex.dataclass(kw_only=True)
class SpeedRateClimb(ConvergedClimb):

    name: str = 'Constant Speed & Rate Climb'

    climb_rate:     float = 0.0
    air_speed:      float = 0.0
    true_course:    float = 0.0

    def __post_init__(self):
        super().__post_init__()

        self._initialize.