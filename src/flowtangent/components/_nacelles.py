# $NAME.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: Apr 2025, Flowtangent Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

# package imports
import jax.numpy as jnp
from flowtangent.library import Component, Dimensions

# Flowtangent imports
from flowtangent.utils import empty_array, field

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
