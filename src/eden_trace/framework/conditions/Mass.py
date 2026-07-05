# Trace/Framework/Missions/Conditions/Mass.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Jul 2024, Trace Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------


# package imports
import jax.numpy as jnp

# Trace imports
from src.eden_trace.utils import empty_array, init_field

from src.eden_trace.framework.conditions import Condition

# ----------------------------------------------------------------------------------------------------------------------
#  Mass
# ----------------------------------------------------------------------------------------------------------------------


class MassConditions(Condition):
    """
    Represents the mass conditions for a vehicle or system.

    This class extends the Conditions base class to specifically handle mass-related
    parameters and calculations.

    Attributes
    ----------
    name : str
        The name of the mass conditions.

    total : np.ndarray
        The total mass of the system.
    rate_of_change : np.ndarray
        The rate of change of mass.

    total_moment_of_inertia : np.ndarray
        The total moment of inertia.

    breakdown : Conditions
        A nested Conditions object representing the breakdown of mass components.

    Notes
    -----
    All attributes are initialized using default factories to ensure each instance
    has its own copy of mutable objects.
    """

    # Attribute             Type        Default Value
    tag: str = init_field("Mass Conditions", static=True)

    total: jnp.ndarray = empty_array((0))
    rate_of_change: jnp.ndarray = empty_array((0))
    volume: jnp.ndarray = empty_array((0))
    density: jnp.ndarray = empty_array((0))
    center_of_gravity: jnp.ndarray = empty_array((0, 3))
    moments_of_inertia: jnp.ndarray = empty_array((0, 3, 3))

    breakdown: Condition = init_field(lambda: Condition(tag="Mass Breakdown"))
