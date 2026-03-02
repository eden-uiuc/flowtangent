# RCAIDE/Framework/Missions/Update/freestream.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Aug 2024, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

# package imports
import equinox as eqx
import jax.numpy as jnp

# RCAIDE imports
import RCAIDE.Framework as rcf

# ----------------------------------------------------------------------------------------------------------------------
#  Update Freestream
# ----------------------------------------------------------------------------------------------------------------------


def update_freestream(state: "rcf.State",
                      system: "rcf.System",
                      settings: "rcf.Settings",
                      ):

    v = state.frames.inertial.velocity_vector
    r = state.freestream.density
    a = state.freestream.speed_of_sound
    m = state.freestream.dynamic_viscosity
    P = state.freestream.pressure
    T = state.freestream.temperature

    gamma   = jnp.polyval(jnp.array(state.freestream.atmosphere.fluid.gamma_coefficients), T)
    Cp      = jnp.polyval(jnp.array(state.freestream.atmosphere.fluid.cp_coefficients), T)

    # Speed
    v_mag_sq = jnp.sum(v ** 2, axis=1)[:, None]
    v_mag    = jnp.sqrt(v_mag_sq)

    # Dynamic Pressure
    q = 0.5 * r * v_mag_sq

    # Mach Number
    M = v_mag / a

    # Stagnation
    P_t = P * (1 + (gamma - 1)/2 * M**2) ** (gamma / (gamma - 1))   # Stagnation Pressure
    T_t = T * (1 + (gamma - 1)/2 * M**2)                            # Stagnation Temperature

    # Reynolds Number (per meter)
    Re = r * v_mag / m

    state = eqx.tree_at(lambda s:(
        s.freestream.gamma,
        s.freestream.Cp,
        s.freestream.speed,
        s.freestream.mach_number,
        s.freestream.reynolds_number,
        s.freestream.dynamic_pressure,
        s.freestream.stagnation_pressure,
        s.freestream.stagnation_temperature,
        ), state, (
        gamma,
        Cp,
        v_mag,
        M,
        Re,
        q,
        P_t,
        T_t,
        )
    )
                   
    return state, system, settings