# flowtangent/Framework/Missions/Conditions/Mass.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Jul 2024, FlowTangent Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------


# package imports


from flowtangent.core._state_data import StateData

# FlowTangent imports
from ...utils import field
from ...utils.typing import TimeScalar, TimeVector3, _

# ----------------------------------------------------------------------------------------------------------------------
#  Mass
# ----------------------------------------------------------------------------------------------------------------------


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
    name: str = field("Mass Conditions", static=True)

    total: TimeScalar = _
    rate_of_change: TimeScalar = _
    volume: TimeScalar = _
    density: TimeScalar = _
    center_of_gravity: TimeVector3 = _

    breakdown: StateData = field(lambda: StateData(name="Mass Breakdown"))
