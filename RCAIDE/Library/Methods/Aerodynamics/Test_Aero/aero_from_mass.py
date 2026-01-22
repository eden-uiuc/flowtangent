# RCAIDE/Library/Methods/Aerodynamics/Test_Aero/aero_from_mass.py
# (c) Copyright 2026 Aerospace Research Community LLC#
# Created:  Jan 2026, J. Smart
# Modified: 
# -------------------------------------------------------------------------------
#  Imports
# -------------------------------------------------------------------------------

# Package Imports

import numpy as np

# RCAIDE Imports

import RCAIDE.Library as rcl
import RCAIDE.Framework as rcf


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

def aero_from_mass(
    state: rcf.State,
    settings: rcf.Settings,
    system: rcf.System):
    """
    Framework version of aero_from_mass
    
    See Also
    --------
    func_aero_from_mass: 
        Functional implementation which this method calls.
    """

    air_density     = state.freestream.density
    flight_speed    = state.freestream.speed

    projected_wing_area = system.main_wing.areas.projected
    wing_aspect_ratio   = system.main_wing.aspect_ratio
    total_mass          = system.mass_properties.total

    C_L, C_D = func_aero_from_mass(
        air_density,
        flight_speed,
        projected_wing_area,
        wing_aspect_ratio,
        total_mass,
    )

    state.aerodynamics.coefficients.lift.total  = C_L
    state.aerodynamics.coefficients.drag.total = C_D

    return state, settings, system
