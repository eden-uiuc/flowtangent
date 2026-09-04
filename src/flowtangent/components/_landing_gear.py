# flowtangent/Library/Components/Landing_Gear.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: May, 2025, Flowtangent Team

# ----------------------------------------------------------------------------------------------------------------------
# IMPORT
# ----------------------------------------------------------------------------------------------------------------------

# package imports

# Flowtangent imports
from flowtangent.library import Component

from flowtangent.utils import field

# ----------------------------------------------------------------------------------------------------------------------
# Landing_Gear
# ----------------------------------------------------------------------------------------------------------------------


class LandingGear(Component):
    tag: str = field("Landing Gear", static=True)

    deployed: bool = False

    number_of_units: int = field(1, static=True)
    number_of_wheels: int = field(0, static=True)

    strut_length: float = 0.0
    tire_diameter: float = 0.0
