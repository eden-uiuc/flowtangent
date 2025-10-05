# $NAME.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: Oct 2025, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

import chex
import numpy as np

from dataclasses import field, make_dataclass

import RCAIDE.Framework as rcf

from RCAIDE.Framework import ProcessStep
from RCAIDE.Framework.Missions.Segments import Segment

# ----------------------------------------------------------------------------------------------------------------------
#  Cruise
# ----------------------------------------------------------------------------------------------------------------------


@chex.dataclass(kw_only=True)
class Cruise(Segment):

    tag: str = 'Cruise'

    distance: float = 0.0
    true_course: float = 0.0

    def __post_init__(self):
        super(Cruise, self).__post_init__()


@chex.dataclass(kw_only=True)
class CSACruise(Cruise):

    tag: str = 'Constant Speed & Altitude Cruise'

    altitude: float = None
    air_speed: float = None

    def initialize_dynamics_and_controls(
            self,
            state: "rcf.State",
            system: "rcf.System",
            settings: "rcf.Settings",
    ):

        alt = self.altitude
        xf = self.distance
        av = self.air_speed
        beta = self.sideslip_angle

        if not self.air_speed:
            av = np.linalg.norm(state.frames.inertial.velocity_vector[-1])
        if not self.altitude:
            alt = -1.0 * state.frames.inertial.position_vector[-1, 2]

        v_x = np.cos(beta) * av
        v_y = np.sin(beta) * av
        t_0 = state.frames.inertia.time[0, 0]
        t_f = t_0 + xf / av

        t_nondim = state.numerics.dimensionless.control_points
        time = t_nondim * (t_f - t_0) + t_0

        state.freesteram.altitude[:, 0] = alt
        state.frames.inertial.position_vector[:, 2] = -alt
        state.frames.inertial.velocity_vector[:, 0] = v_x
        state.frames.inertial.velocity_vector[:, 1] = v_y
        state.frames.inertial.time[:, 0] = time

        # Set active controls and dynamics
        state.controls.throttle.active = True
        state.controls.body_angle = True

        state.controls.residuals.force_x.active = True
        state.controls.residuals.force_z.active = True

        return state, system, settings

    def __post_init__(self):
        super(CSACruise, self).__post_init__()
        self._initialize.append(ProcessStep(tag='Dynamics and Controls',
                                            function=self.initialize_dynamics_and_controls))