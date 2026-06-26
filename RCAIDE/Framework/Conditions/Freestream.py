# RCAIDE/Framework/Missions/Conditions/Freestream.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Aug 2024, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

# package imports
import jax.numpy as jnp

# RCAIDE imports
from RCAIDE.utils import empty_array, init_field

from RCAIDE.Library.Atmospheres import Atmosphere, USStandard1976
from RCAIDE.Library.Planets import Earth, Planet

from RCAIDE.Framework.Conditions import Conditions

# ----------------------------------------------------------------------------------------------------------------------
#  Freestream
# ----------------------------------------------------------------------------------------------------------------------


class FreestreamConditions(Conditions):
    """
    Represents the freestream conditions in a flight environment.

    This class encapsulates various atmospheric and flight parameters that define
    the freestream conditions for aerodynamic analysis or simulation.

    Attributes
    ----------
    name : str, optional
        Name of the freestream condition. Default is 'Freestream'.

    velocity : jnp.ndarray, optional
        Velocity (speed) of the freestream. Default is empty(0).
    u : jnp.ndarray, optional
        X-component of velocity. Default is empty(0).
    v : jnp.ndarray, optional
        Y-component of velocity. Default is empty(0).
    w : jnp.ndarray, optional
        Z-component of velocity. Default is empty(0).

    altitude : jnp.ndarray, optional
        Altitude of the freestream condition. Default is empty(0).

    gravity : jnp.ndarray, optional
        Gravitational acceleration. Default is empty(0).

    pressure : jnp.ndarray, optional
        Atmospheric pressure. Default is empty(0).
    temperature : jnp.ndarray, optional
        Atmospheric temperature. Default is empty(0).
    density : jnp.ndarray, optional
        Air density. Default is empty(0).

    speed_of_sound : jnp.ndarray, optional
        Speed of sound in the atmosphere. Default is empty(0).

    dynamic_viscosity : jnp.ndarray, optional
        Dynamic viscosity of the air. Default is empty(0).
    dynamic_pressure : jnp.ndarray, optional
        Dynamic pressure of the freestream. Default is empty(0).

    mach_number : jnp.ndarray, optional
        Mach number of the freestream. Default is empty(0).
    reynolds_number : jnp.ndarray, optional
        Reynolds number of the flow. Default is empty(0).

    delta_ISA : jnp.ndarray, optional
        Deviation from International Standard Atmosphere. Default is empty(0).

    Notes
    -----
    All attributes are initialized as zero arrays of shape (1, 1) by default.
    """

    tag: str = init_field("Freestream", static=True)
    atmosphere: Atmosphere = init_field(USStandard1976)
    planet: Planet = init_field(Earth)

    speed: jnp.ndarray = empty_array(0)
    altitude: jnp.ndarray = empty_array(0)
    gravity: jnp.ndarray = empty_array(0)

    speed_of_sound: jnp.ndarray = empty_array(0)
    pressure: jnp.ndarray = empty_array(0)
    temperature: jnp.ndarray = empty_array(0)
    density: jnp.ndarray = empty_array(0)

    dynamic_viscosity: jnp.ndarray = empty_array(0)
    dynamic_pressure: jnp.ndarray = empty_array(0)

    stagnation_pressure: jnp.ndarray = empty_array(0)
    stagnation_temperature: jnp.ndarray = empty_array(0)

    mach_number: jnp.ndarray = empty_array(0)
    reynolds_number: jnp.ndarray = empty_array(0)

    delta_ISA: jnp.ndarray = empty_array(0)
    gamma: jnp.ndarray = empty_array(0)
    Cp: jnp.ndarray = empty_array(0)
    R: jnp.ndarray = empty_array(0)
