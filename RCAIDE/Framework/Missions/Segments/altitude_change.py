# RCAIDE/Framework/Missions/Segments/Climb.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: Sep, 2025, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
# IMPORT 
# ----------------------------------------------------------------------------------------------------------------------

# package imports
import chex #TODO: Switch to Equinox
import equinox as eqx
import jax.numpy as jnp


# RCAIDE imports
import RCAIDE.Framework as rcf

from RCAIDE.Framework import ProcessStep
from RCAIDE.Framework.Missions.Segments import Segment
from RCAIDE.Framework.Missions.Initialize import initialize_altitude_differential

# ----------------------------------------------------------------------------------------------------------------------
# Climb
# ----------------------------------------------------------------------------------------------------------------------

# @chex.dataclass
# class AltitudeChange(Segment):

#     tag: str = eqx.field(static=True, default='Altitude Change')

#     altitude_start: float = 0.0
#     altitude_end:   float = 0.0

#     def __post_init__(self):
#         super(AltitudeChange, self).__post_init__()

#         self.initialize.append(
#             ProcessStep(tag='Altitude Differential',
#                         function=initialize_altitude_differential)
#         )

# @chex.dataclass
# class CSRAltitudeChange(AltitudeChange):

#     tag: str = eqx.field(static=True, default='Constant Speed & Rate Altitude Change')

#     rate:           float = 0.0
#     air_speed:      float = 0.0
#     true_course:    float = 0.0

#     active_controls = ("body_angle", "throttle")
#     active_residuals = ("force_x", "force_z")

#     def initialize_conditions(
#             self,
#             state: "rcf.State",
#             system: "rcf.System",
#             settings: "rcf.Settings",
#     ):
#         # Unpack inputs from segment parameters and state

#         rate    = self.rate
#         av      = self.air_speed

#         alt0    = self.altitude_start
#         altf    = self.altitude_end

#         beta    = self.sideslip_angle

#         t_nondim = state.numerics.dimensionless.control_points

#         # If air speed and altitude are not provided, inherit from previous segment

#         if not self.air_speed:
#             av = jnp.linalg.norm(state.frames.inertial.velocity_vector[-1])
#         if not self.altitude_start:
#             alt0 = -1.0 * state.frames.inertial.position_vector[-1, 2]

#         # Calculate velocity vector in inertial frame
#         v_xy    = jnp.sqrt(av ** 2 - rate ** 2)
#         v_x     = jnp.cos(beta) * v_xy
#         v_y     = jnp.sin(beta) * v_xy

#         state.frames.inertial.velocity_vector[:, 0] = v_x
#         state.frames.inertial.velocity_vector[:, 1] = v_y
#         state.frames.inertial.velocity_vector[:, 2] = -rate

#         # Calculate altitude using time discretization
#         alt = t_nondim * (altf - alt0) + alt0
#         state.frames.inertial.position_vector[:, 2] = -alt[:, 0]
#         state.freestream.altitude[:, 0] = alt[:, 0]

#         return state, system, settings


#     def __post_init__(self):
#         super(CSRAltitudeChange, self).__post_init__()
#         self.initialize.append(
#             ProcessStep(tag='Conditions',
#                         function=self.initialize_conditions)
#         )
