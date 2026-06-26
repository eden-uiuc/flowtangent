# RCAIDE/Library/Components/Landing_Gear.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: May, 2025, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
# IMPORT
# ----------------------------------------------------------------------------------------------------------------------

# package imports
import equinox as eqx

# RCAIDE imports
from RCAIDE.utils import init_field

from RCAIDE.Library import Component

# ----------------------------------------------------------------------------------------------------------------------
# Landing_Gear
# ----------------------------------------------------------------------------------------------------------------------


class LandingGear(Component):

    tag:                str     = init_field('Landing Gear', static=True)

    deployed:           bool    = False

    number_of_units:    int     = init_field(1, static=True)
    number_of_wheels:   int     = init_field(0, static=True)

    strut_length:       float   = 0.
    tire_diameter:      float   = 0.
