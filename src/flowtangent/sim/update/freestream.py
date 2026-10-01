# flowtangent/Framework/Missions/Update/freestream.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Aug 2024, Flowtangent Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flowtangent import Settings, State, System

# package imports
import jax.numpy as jnp

from ...utils import update

# ----------------------------------------------------------------------------------------------------------------------
#  Update Freestream
# ----------------------------------------------------------------------------------------------------------------------


def update_freestream(
    state: State,
    system: System,
    settings: Settings,
):

    # Update Altitude
    alt = -state.frames.inertial.position_vector[:, 2][:, None]  # Z is negative by right hand rule convention
    state = update(state, "freestream.altitude", alt)

    # Update gravity
    G = state.freestream.planet.compute_gravity()
    state = update(state, "freestream.gravity", jnp.atleast_2d(G))

    # Update Atmospheric Properties
    atmo = state.freestream.atmosphere
    r = atmo.compute_density(alt)
    P = atmo.compute_pressure(alt)
    T = atmo.compute_temperature(alt)
    a = atmo.compute_speed_of_sound(alt)
    m = atmo.compute_dynamic_viscosity(alt)
    Cp = atmo.compute_Cp(alt)
    gamma = atmo.compute_gamma(alt)

    updated_fs = update(
        state.freestream,
        (
            ("density", r),
            ("pressure", P),
            ("temperature", T),
            ("speed_of_sound", a),
            ("dynamic_viscosity", m),
            ("gamma", gamma),
            ("Cp", Cp),
        ),
    )
    state = update(state, "freestream", updated_fs)

    # Speed
    v = state.frames.inertial.velocity_vector
    v_mag_sq = jnp.sum(v**2, axis=1)[:, None]
    v_mag = jnp.sqrt(v_mag_sq)

    # Dynamic Pressure
    q = 0.5 * r * v_mag_sq

    # Mach Number
    M = v_mag / a

    # Stagnation
    P_t = P * (1 + (gamma - 1) / 2 * M**2) ** (gamma / (gamma - 1))  # Stagnation Pressure
    T_t = T * (1 + (gamma - 1) / 2 * M**2)  # Stagnation Temperature

    # Reynolds Number (per meter)
    Re = r * v_mag / m

    state = update(
        state,
        (
            ("freestream.speed", v_mag),
            ("freestream.mach_number", M),
            ("freestream.reynolds_number", Re),
            ("freestream.dynamic_pressure", q),
            ("freestream.stagnation_pressure", P_t),
            ("freestream.stagnation_temperature", T_t),
        ),
    )

    return state, system, settings
