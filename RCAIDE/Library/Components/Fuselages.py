# RCAIDE/Compoments/Fuselages/Fuselage.py
# (c) Copyright 2023 Aerospace Research Community LLC
#
# Created: Jul 2024, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

# package imports
import jax.numpy as jnp

# RCAIDE imports
from RCAIDE.utils import empty_array, init_field

from RCAIDE.Library import Component, ComponentAreas, ComponentDimensions, ComponentFineness

# ----------------------------------------------------------------------------------------------------------------------
#  Fuselage
# ----------------------------------------------------------------------------------------------------------------------


class FuselageHeights(ComponentDimensions):
    quarter_length: float = 0.0
    three_quarters_length: float = 0.0
    wing_root_quarter_chord: float = 0.0
    vertical_root_quarter_chord: float = 0.0


class FuselageLengths(ComponentDimensions):
    nose: float = 0.0
    tail: float = 0.0
    cabin: float = 0.0
    fore_space: float = 0.0
    aft_space: float = 0.0
    ordinal_direction: bool = init_field(True, static=True)


class FuselageSegment(Component):
    percent_x_location: float = 0.0
    percent_z_location: float = 0.0


class Fuselage(Component):
    aerodynamic_center: jnp.ndarray = empty_array((0, 3))

    number_of_seats: int = init_field(1, static=True)
    seats_abreast: int = init_field(0, static=True)
    seat_pitch: float = init_field(0.0, static=True)
    differential_pressure: float = init_field(0.0, static=True)

    heights: ComponentDimensions = init_field(FuselageHeights)
    lengths: ComponentDimensions = init_field(FuselageLengths)

    diameters: ComponentDimensions = init_field(ComponentDimensions)
    fineness: ComponentFineness = init_field(ComponentFineness)


# ----------------------------------------------------------------------------------------------------------------------
#  BWB Fuselage
# ----------------------------------------------------------------------------------------------------------------------


class BWBAreas(ComponentAreas):
    aft_centerbody: float = 0.0


class BWBFuselage(Fuselage):
    tag: str = init_field("BWB Fuselage", static=True)

    aft_centerbody_taper: float = init_field(0.0, static=True)

    areas: BWBAreas = init_field(BWBAreas)  # type: ignore
