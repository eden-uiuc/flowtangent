# Trace/Compoments/Fuselages/Fuselage.py
# (c) Copyright 2023 Aerospace Research Community LLC
#
# Created: Jul 2024, Trace Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

# package imports
import jax.numpy as jnp

# Trace imports
from eden_trace.utils import empty_array, field

from eden_trace.library import Component, Areas, Dimensions, Fineness

# ----------------------------------------------------------------------------------------------------------------------
#  Fuselage
# ----------------------------------------------------------------------------------------------------------------------


class FuselageHeights(Dimensions):
    quarter_length: float = 0.0
    three_quarters_length: float = 0.0
    wing_root_quarter_chord: float = 0.0
    vertical_root_quarter_chord: float = 0.0


class FuselageLengths(Dimensions):
    nose: float = 0.0
    tail: float = 0.0
    cabin: float = 0.0
    fore_space: float = 0.0
    aft_space: float = 0.0
    ordinal_direction: bool = field(True, static=True)


class FuselageSegment(Component):
    percent_x_location: float = 0.0
    percent_z_location: float = 0.0


class Fuselage(Component):
    aerodynamic_center: jnp.ndarray = empty_array((0, 3))

    number_of_seats: int = field(1, static=True)
    seats_abreast: int = field(0, static=True)
    seat_pitch: float = field(0.0, static=True)
    differential_pressure: float = field(0.0, static=True)

    heights: Dimensions = field(FuselageHeights)
    lengths: Dimensions = field(FuselageLengths)

    diameters: Dimensions = field(Dimensions)
    fineness: Fineness = field(Fineness)


# ----------------------------------------------------------------------------------------------------------------------
#  BWB Fuselage
# ----------------------------------------------------------------------------------------------------------------------


class BWBAreas(Areas):
    aft_centerbody: float = 0.0


class BWBFuselage(Fuselage):
    tag: str = field("BWB Fuselage", static=True)

    aft_centerbody_taper: float = field(0.0, static=True)

    areas: BWBAreas = field(BWBAreas)  # type: ignore
