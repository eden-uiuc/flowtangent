# $NAME.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: Apr 2025, Trace Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

# package imports
import jax.numpy as jnp

# Trace imports
from eden_trace.utils import empty_array, field

from eden_trace.library import Component, Dimensions

# ----------------------------------------------------------------------------------------------------------------------
#  Nacelle
# ----------------------------------------------------------------------------------------------------------------------


class NacelleDiameters(Dimensions):
    inlet: float = 0.0


class Nacelle(Component):
    tag: str = field("Nacelle", static=True)
    flow_through: bool = field(False, static=True)
    fuselage_integrated: bool = field(False, static=True)
    has_pylon: bool = field(True)

    aerodynamic_center: jnp.ndarray = empty_array((0, 3))
    orientation_euler_angles: jnp.ndarray = empty_array((0, 3))

    airfoil: Component | None = None
    cowling_airfoil_angle: float = 0.0

    diameters: NacelleDiameters = field(NacelleDiameters)  # type: ignore
