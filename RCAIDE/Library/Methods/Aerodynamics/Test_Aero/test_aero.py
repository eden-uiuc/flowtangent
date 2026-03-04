# RCAIDE/Library/Methods/Aerodynamics/Test_Aero/test_aero.py
# (c) Copyright 2026 Aerospace Research Community LLC#
# Created:  Jan 2026, J. Smart
# Modified: 
# -------------------------------------------------------------------------------
#  Imports
# -------------------------------------------------------------------------------
from typing import TYPE_CHECKING

# package imports
import equinox as eqx
import jax.numpy as np

# RCAIDE Imports

# --- Framework Imports (Strictly for Type Hinting to avoid Circular Imports) ---
if TYPE_CHECKING:
    from RCAIDE.Framework.State import State
    from RCAIDE.Framework.System import System
    from RCAIDE.Framework.Settings import Settings


# -------------------------------------------------------------------------------
#  Functional/Library Version
# -------------------------------------------------------------------------------

def func_aero_from_mass(
    air_density: float,
    flight_speed: float,
    projected_wing_area: float,
    wing_aspect_ratio: float,
    total_mass: float,
    Oswald_efficiency_factor: float = 0.85,
    CL_max: float = 1.5,
    parasitic_drag: float = 0.06,
):
    # TODO: Implement functional version of aero_from_mass

    q = 0.5 * air_density * flight_speed**2
    C_L = min(total_mass * 9.81 / (q * projected_wing_area), CL_max)

    induced_drag_factor = 1.0 / (np.pi * Oswald_efficiency_factor * wing_aspect_ratio)
    induced_drag = induced_drag_factor * C_L**2

    C_D = parasitic_drag + induced_drag

    return C_L, C_D


# -------------------------------------------------------------------------------
#  Stateful/Framework Version
# -------------------------------------------------------------------------------

def direct_aero(
    state: "State",
    system: "System",
    settings: "Settings",
):

    C_L = state.aerodynamics.coefficients.lift.total
    C_D = state.aerodynamics.coefficients.drag.total

    rho = 0.4
    flight_speed = state.freestream.speed
    S = system.areas.reference
    qS = 0.5 * rho * flight_speed**2 * S

    F_Z = qS * C_L
    F_X = qS * (C_D + 0.06)

    state = eqx.tree_at(lambda s: s.frames.wind.total_force_vector, state, state.frames.wind.total_force_vector.at[:, 2].set(F_Z.flatten()))
    state = eqx.tree_at(lambda s: s.frames.wind.total_force_vector, state, state.frames.wind.total_force_vector.at[:, 0].set(F_X.flatten()))

    return state, system, settings,




