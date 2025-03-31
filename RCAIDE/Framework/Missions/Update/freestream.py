# RCAIDE/Framework/Missions/Update/freestream.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Aug 2024, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from dataclasses import dataclass, field

# package imports
import numpy as np

# RCAIDE imports
import RCAIDE.Framework as rcf

# ----------------------------------------------------------------------------------------------------------------------
#  Update Freestream
# ----------------------------------------------------------------------------------------------------------------------


def update_freestream(State: rcf.State,
                      Settings: rcf.Settings,
                      System: rcf.System):

    v = State.frames.inertial.velocity
    r = State.freestream.density
    a = State.freestream.speed_of_sound
    m = State.freestream.dynamic_viscosity
    P = State.freestream.pressure
    T = State.freestream.temperature

    gamma = np.polyval(State.freestream.atmosphere.fluid.gamma_coefficients, T)
    Cp = np.polyval(State.freestream.atmosphere.fluid.cp_coefficients, T)

    # Speed
    v_mag_sq = np.sum(v ** 2, axis=1)[:, None]
    v_mag    = np.sqrt(v_mag_sq)

    # Dynamic Pressure
    q = 0.5 * r * v_mag_sq

    # Mach Number
    M = v_mag / a

    # Stagnation
    P_t = P * (1 + (g - 1)/2 * M**2) ** (g / (g-1))   # Stagnation Pressure
    T_t = T * (1 + (g - 1)/2 * M**2)                  # Stagnation Temperature

    # Reynolds Number (per meter)
    Re = r * v_mag / m

    State.freestream.gamma                  = gamma
    State.freestream.cp                     = Cp
    State.freestream.velocity               = v_mag
    State.freestream.mach_number            = M
    State.freestream.reynolds_number        = Re
    State.freestream.dynamic_pressure       = q
    State.freestream.stagnation_pressure    = P_t
    State.freestream.stagnation_temperature = T_t
                   
    return State, Settings, System