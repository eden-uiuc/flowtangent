# flowtangent/Library/Components/Landing_Gear.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: May, 2025, FlowTangent Team

# ----------------------------------------------------------------------------------------------------------------------
# IMPORT
# ----------------------------------------------------------------------------------------------------------------------

# package imports

# FlowTangent imports
from ..core._component import Component
from ..utils import static_field

# ----------------------------------------------------------------------------------------------------------------------
# Landing_Gear
# ----------------------------------------------------------------------------------------------------------------------


class LandingGear(Component):
    deployed: bool = False

    number_of_units: int = static_field(1)
    number_of_wheels: int = static_field(0)

    strut_length: float = 0.0
    tire_diameter: float = 0.0
