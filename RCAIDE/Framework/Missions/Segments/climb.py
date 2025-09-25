# RCAIDE/Framework/Missions/Segments/Climb.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: Sep, 2025, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
# IMPORT 
# ----------------------------------------------------------------------------------------------------------------------

import chex
import numpy as np

from dataclasses import field, make_dataclass

import RCAIDE.Framework as rcf

from RCAIDE.Framework import ProcessStep
from RCAIDE.Framework.Missions.Segments import ConvergedSegment
from RCAIDE.Framework.Missions.Initialize import altitude_differential

# ----------------------------------------------------------------------------------------------------------------------
# Climb
# ----------------------------------------------------------------------------------------------------------------------


@chex.dataclass(kw_only=True)
class ConvergedClimb(ConvergedSegment):

    name: str = 'Climb'

    altitude_start: float = None
    altitude_end:   float = 0.0

    def __post_init__(self):
        super(ConvergedClimb, self).__post_init__()

        self._initialize.append(ProcessStep(name='Altitude Differential',
                                            function=altitude_differential))


@chex.dataclass(kw_only=True)
class SpeedRateClimb(ConvergedClimb):

    name: str = 'Constant Speed & Rate Climb'

    climb_rate:     float = 0.0
    air_speed:      float = None
    true_course:    float = 0.0

    def initialize_dynamics_and_controls(
            self,
            state: "rcf.State",
            system: "rcf.System",
            settings: "rcf.Settings",
    ):
        # Unpack inputs from segment parameters and state

        cr      = self.climb_rate
        av      = self.air_speed

        alt0    = self.altitude_start
        altf    = self.altitude_end

        beta    = self.sideslip_angle

        t_nondim = state.numerics.dimensionless.control_points

        # If air speed and altitude are not provided, inherit from previous segment

        if not self.air_speed:
            av = state.frames.inertial.velocity_vector[-1]
        if not self.altitude_start:
            alt0 = -1.0 * state.frames.inertial.position_vector[-1, 2]

        # Calculate velocity vector in inertial frame
        v_xy    = np.sqrt(av ** 2 - (-cr) ** 2)
        v_x     = np.cos(beta) * v_xy
        v_y     = np.sin(beta) * v_xy

        state.frames.inertial.velocity_vector[:, 0] = v_x
        state.frames.inertial.velocity_vector[:, 1] = v_y
        state.frames.inertial.velocity_vector[:, 2] = -cr

        # Calculate altitude using time discretization
        alt = t_nondim * (altf - alt0) + alt0
        state.frames.inertial.position_vector[:, 2] = -alt[:, 0]
        state.freestream.altitude[:, 0] = alt[:, 0]

        # Set active controls
        state.controls.throttle.active = True
        state.controls.body_angle = True

        return state, system, settings



    def __post_init__(self):
        super(SpeedRateClimb, self).__post_init__()
        self._initialize.append(ProcessStep(name='Dynamics and Controls',
                                            function=self.initialize_dynamics_and_controls))
