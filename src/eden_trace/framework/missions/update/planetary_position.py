# RCAIDE/Framework/Missions/Update/planetary_position.py
# (c) Copyright 2026 Aerospace Research Community LLC
#
# Created: Mar 2026, J. Smart
# Modified: Mar 2026, J. Smart

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp

# --- Framework Imports (Strictly for Type Hinting to avoid Circular Imports) ---
if TYPE_CHECKING:
    from src.eden_trace.framework.settings import Settings
    from src.eden_trace.framework.state import State
    from src.eden_trace.framework.systems import System

from src.eden_trace.library import units

# ----------------------------------------------------------------------------------------------------------------------
#  Update Planetary Position
# ----------------------------------------------------------------------------------------------------------------------


def update_planetary_position(state: "State", system: "System", settings: "Settings"):

    # Unpack state
    v = state.frames.inertial.velocity_vector[:, 0]  # Velocity over ground along true course
    alt = state.freestream.altitude

    theta = state.frames.body.inertial_rotations[:, 1]
    psi = state.frames.planet.true_course
    Re = state.freestream.planet.mean_radius

    alpha = state.aerodynamics.angles.alpha

    I = state.time.dimensional.integrate

    #  Calculate flight path and radius
    gamma = theta - alpha
    R = alt + Re

    # Find local velocities and integrate position
    lamdadot = (v / R) * jnp.cos(gamma) * jnp.cos(psi)
    lamda = jnp.dot(I, lamdadot) / units.deg  # Latitude

    mudot = (v / R) * jnp.cos(gamma) * jnp.sin(psi) / jnp.cos(lamda)
    mu = jnp.dot(I, mudot) / units.deg  # Longitude

    lat_0 = state.frames.planet.latitude[0, 0]
    lon_0 = state.frames.planet.longitude[0, 0]

    updated_state = eqx.tree_at(
        lambda s: (s.frames.planet.latitude, s.frames.planet.longitude), state, (lat_0 + lamda, lon_0 + mu)
    )

    return updated_state, system, settings
