# $NAME.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: Apr 2025, FlowTangent Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

# package imports
import jax

from flowtangent.core._component import Component, Dimensions

# FlowTangent imports
from flowtangent.utils import empty_array, field

# ----------------------------------------------------------------------------------------------------------------------
#  Nacelle
# ----------------------------------------------------------------------------------------------------------------------


class NacelleDiameters(Dimensions):
    inlet: float = 0.0


class Nacelle(Component):
    name: str = field("Nacelle", static=True)
    flow_through: bool = field(False, static=True)
    fuselage_integrated: bool = field(False, static=True)
    has_pylon: bool = field(True)

    aerodynamic_center: jax.Array = empty_array((0, 3))
    orientation_euler_angles: jax.Array = empty_array((0, 3))

    airfoil: Component | None = None
    cowling_airfoil_angle: float = 0.0

    diameters: NacelleDiameters = field(NacelleDiameters)  # type: ignore
