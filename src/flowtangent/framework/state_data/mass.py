# flowtangent/Framework/Missions/Conditions/Mass.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Jul 2024, Flowtangent Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------


# package imports
import jax.numpy as jnp

# Flowtangent imports
from flowtangent.utils import empty_array, field, register

from flowtangent.framework.state_data import StateData

# ----------------------------------------------------------------------------------------------------------------------
#  Mass
# ----------------------------------------------------------------------------------------------------------------------

@register
class Mass(StateData):
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
    tag: str = field("Mass Conditions", static=True)

    total: jnp.ndarray = empty_array(())
    rate_of_change: jnp.ndarray = empty_array(())
    volume: jnp.ndarray = empty_array(())
    density: jnp.ndarray = empty_array(())
    center_of_gravity: jnp.ndarray = empty_array((0, 3))
    moments_of_inertia: jnp.ndarray = empty_array((0, 3, 3))

    breakdown: StateData = field(lambda: StateData(tag="Mass Breakdown"))
