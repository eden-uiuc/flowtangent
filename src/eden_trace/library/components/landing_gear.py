# Trace/Library/Components/Landing_Gear.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: May, 2025, Trace Team

# ----------------------------------------------------------------------------------------------------------------------
# IMPORT
# ----------------------------------------------------------------------------------------------------------------------

# package imports

# Trace imports
from src.eden_trace.utils import init_field

from src.eden_trace.library import Component

# ----------------------------------------------------------------------------------------------------------------------
# Landing_Gear
# ----------------------------------------------------------------------------------------------------------------------


class LandingGear(Component):
    tag: str = init_field("Landing Gear", static=True)

    deployed: bool = False

    number_of_units: int = init_field(1, static=True)
    number_of_wheels: int = init_field(0, static=True)

    strut_length: float = 0.0
    tire_diameter: float = 0.0
