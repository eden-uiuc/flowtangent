# $NAME.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: Apr 2025, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

# package imports
import jax.numpy as jnp

# RCAIDE imports
from src.eden_trace.utils import empty_array, init_field

from src.eden_trace.library import Component, ComponentDimensions

# ----------------------------------------------------------------------------------------------------------------------
#  Nacelle
# ----------------------------------------------------------------------------------------------------------------------


class NacelleDiameters(ComponentDimensions):
    inlet: float = 0.0


class Nacelle(Component):
    tag: str = init_field("Nacelle", static=True)
    flow_through: bool = init_field(False, static=True)
    fuselage_integrated: bool = init_field(False, static=True)
    has_pylon: bool = init_field(True)

    aerodynamic_center: jnp.ndarray = empty_array((0, 3))
    orientation_euler_angles: jnp.ndarray = empty_array((0, 3))

    airfoil: Component | None = None
    cowling_airfoil_angle: float = 0.0

    diameters: NacelleDiameters = init_field(NacelleDiameters)  # type: ignore
