# flowtangent/Framework/Missions/Conditions/Aerodynamics.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Aug 2024, Flowtangent Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

# package imports
import jax.numpy as jnp

from flowtangent.core._state_data import StateData

# Flowtangent imports
from flowtangent.utils import empty_array, field, register

# ----------------------------------------------------------------------------------------------------------------------
#  Aerodynamics
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------
#  Coefficients
# ----------------------------------------------------------

# Component-Level Bookkeeping ------------------------------


@register
class ComponentCoeffs(StateData):
    tag: str = field("Component Coefficients", static=True)

    total: jnp.ndarray = empty_array()

    # Component Arrays: (n_time, n_components)
    wings: jnp.ndarray = empty_array((0, 0))
    fuselages: jnp.ndarray = empty_array((0, 0))
    nacelles: jnp.ndarray = empty_array((0, 0))


# Lift Coefficients ----------------------------------------


@register
class LiftCoeffs(StateData):
    # Attribute     Type            Default Value
    tag: str = field("Lift Coefficients", static=True)

    total: jnp.ndarray = empty_array((0,))

    inviscid: ComponentCoeffs = field(lambda: ComponentCoeffs(tag="Inviscid Lift"))
    compressible: ComponentCoeffs = field(lambda: ComponentCoeffs(tag="Compressible Lift"))


# Drag Coefficients ----------------------------------------


@register
class InducedDrag(StateData):
    # Attribute   Type            Default Value
    tag: str = field("Induced Drag", static=True)

    total: jnp.ndarray = empty_array()

    inviscid: ComponentCoeffs = field(lambda: ComponentCoeffs(tag="Inviscid Induced Drag"))
    viscous: ComponentCoeffs = field(lambda: ComponentCoeffs(tag="Viscous Induced Drag"))
    near_field: ComponentCoeffs = field(lambda: ComponentCoeffs(tag="Near-Field Induced Drag"))
    far_field: ComponentCoeffs = field(lambda: ComponentCoeffs(tag="Far-Field Induced Drag"))


@register
class DragCoeffs(StateData):
    # Attribute     Type            Default Value
    tag: str = field("Drag Coefficients", static=True)

    total: jnp.ndarray = empty_array()

    parasite: ComponentCoeffs = field(lambda: ComponentCoeffs(tag="Parasite Drag"))
    compressible: ComponentCoeffs = field(lambda: ComponentCoeffs(tag="Compressible Drag"))
    miscellaneous: ComponentCoeffs = field(lambda: ComponentCoeffs(tag="Miscellaneous Drag"))
    spoiler: ComponentCoeffs = field(lambda: ComponentCoeffs(tag="Spoiler Drag"))

    induced: InducedDrag = field(InducedDrag)


# Moment Coefficients --------------------------------------


@register
class MomentCoeffs(StateData):
    # Attribute         Type            Default Value
    tag: str = field("Moment Coefficients", static=True)

    pitch: jnp.ndarray = empty_array()
    roll: jnp.ndarray = empty_array()
    yaw: jnp.ndarray = empty_array()


# All Coefficients -----------------------------------------


@register
class Coefficients(StateData):
    # Attribute         Type                Default Value
    tag: str = field("Aerodynamic Coefficients", static=True)

    lift: LiftCoeffs = field(LiftCoeffs)
    drag: DragCoeffs = field(DragCoeffs)

    moments: MomentCoeffs = field(MomentCoeffs)

    X: jnp.ndarray = empty_array()
    Y: jnp.ndarray = empty_array()
    Z: jnp.ndarray = empty_array()


# ----------------------------------------------------------
#  Aerodynamic Angles
# ----------------------------------------------------------


@register
class Angles(StateData):
    # Attribute         Type        Default Value
    tag: str = field("Aerodynamic Angles", static=True)

    alpha: jnp.ndarray = empty_array()  # Y-axis / angle of attack
    beta: jnp.ndarray = empty_array()  # Z-axis / sideslip angle
    phi: jnp.ndarray = empty_array()  # X-axis / roll angle


# ----------------------------------------------------------
#  Full Aerodynamic Conditions
# ----------------------------------------------------------


@register
class Aerodynamics(StateData):
    # Attribute     Type                    Default Value
    tag: str = field("Aerodynamics", static=True)

    angles: Angles = field(Angles)

    coefficients: Coefficients = field(Coefficients)
