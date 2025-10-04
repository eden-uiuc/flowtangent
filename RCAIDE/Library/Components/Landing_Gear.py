# RCAIDE/Library/Components/Landing_Gear.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: May, 2025, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
# IMPORT 
# ----------------------------------------------------------------------------------------------------------------------

import chex
from dataclasses import field, make_dataclass


import RCAIDE.Library as rcl

# ----------------------------------------------------------------------------------------------------------------------
# Landing_Gear
# ----------------------------------------------------------------------------------------------------------------------


@chex.dataclass(kw_only=True)
class LandingGear(rcl.Component):

    tag: str = 'Landing Gear'

    deployed: bool = False

    number_of_units:    int = 1
    number_of_wheels:   int = 0

    strut_length:       float = 0.
    tire_diameter:      float = 0.
