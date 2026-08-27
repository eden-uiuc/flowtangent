# Trace/Framework/Missions/Conditions/Aerodynamics.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Aug 2024, Trace Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

# package imports
import jax.numpy as jnp

# Trace imports
from eden_trace.utils import empty_array, init_field, register

from eden_trace.framework.state_data import StateData

# ----------------------------------------------------------------------------------------------------------------------
#  Aerodynamics
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------
#  Coefficients
# ----------------------------------------------------------

# Component-Level Bookkeeping ------------------------------


@register
class ComponentCoeffs(StateData):
    tag: str = init_field("Component Coefficients", static=True)

    total: jnp.ndarray = empty_array()

    # Component Arrays: (n_time, n_components)
    wings: jnp.ndarray = empty_array((0, 0))
    fuselages: jnp.ndarray = empty_array((0, 0))
    nacelles: jnp.ndarray = empty_array((0, 0))


# Lift Coefficients ----------------------------------------


@register
class LiftCoeffs(StateData):
    # Attribute     Type            Default Value
    tag: str = init_field("Lift Coefficients", static=True)

    total: jnp.ndarray = empty_array((0,))

    inviscid: ComponentCoeffs = init_field(lambda: ComponentCoeffs(tag="Inviscid Lift"))
    compressible: ComponentCoeffs = init_field(lambda: ComponentCoeffs(tag="Compressible Lift"))


# Drag Coefficients ----------------------------------------


@register
class InducedDrag(StateData):
    # Attribute   Type            Default Value
    tag: str = init_field("Induced Drag", static=True)

    total: jnp.ndarray = empty_array()

    inviscid: ComponentCoeffs = init_field(lambda: ComponentCoeffs(tag="Inviscid Induced Drag"))
    viscous: ComponentCoeffs = init_field(lambda: ComponentCoeffs(tag="Viscous Induced Drag"))
    near_field: ComponentCoeffs = init_field(lambda: ComponentCoeffs(tag="Near-Field Induced Drag"))
    far_field: ComponentCoeffs = init_field(lambda: ComponentCoeffs(tag="Far-Field Induced Drag"))


@register
class DragCoeffs(StateData):
    # Attribute     Type            Default Value
    tag: str = init_field("Drag Coefficients", static=True)

    total: jnp.ndarray = empty_array()

    parasite: ComponentCoeffs = init_field(lambda: ComponentCoeffs(tag="Parasite Drag"))
    compressible: ComponentCoeffs = init_field(lambda: ComponentCoeffs(tag="Compressible Drag"))
    miscellaneous: ComponentCoeffs = init_field(lambda: ComponentCoeffs(tag="Miscellaneous Drag"))
    spoiler: ComponentCoeffs = init_field(lambda: ComponentCoeffs(tag="Spoiler Drag"))

    induced: InducedDrag = init_field(InducedDrag)


# Moment Coefficients --------------------------------------


@register
class MomentCoeffs(StateData):
    # Attribute         Type            Default Value
    tag: str = init_field("Moment Coefficients", static=True)

    pitch: jnp.ndarray = empty_array()
    roll: jnp.ndarray = empty_array()
    yaw: jnp.ndarray = empty_array()


# All Coefficients -----------------------------------------


@register
class Coefficients(StateData):
    # Attribute         Type                Default Value
    tag: str = init_field("Aerodynamic Coefficients", static=True)

    lift: LiftCoeffs = init_field(LiftCoeffs)
    drag: DragCoeffs = init_field(DragCoeffs)

    moments: MomentCoeffs = init_field(MomentCoeffs)

    X: jnp.ndarray = empty_array()
    Y: jnp.ndarray = empty_array()
    Z: jnp.ndarray = empty_array()


# ----------------------------------------------------------
#  Aerodynamic Angles
# ----------------------------------------------------------


@register
class Angles(StateData):
    # Attribute         Type        Default Value
    tag: str = init_field("Aerodynamic Angles", static=True)

    alpha: jnp.ndarray = empty_array()  # Y-axis / angle of attack
    beta: jnp.ndarray = empty_array()  # Z-axis / sideslip angle
    phi: jnp.ndarray = empty_array()  # X-axis / roll angle


# ----------------------------------------------------------
#  Full Aerodynamic Conditions
# ----------------------------------------------------------


@register
class Aerodynamics(StateData):
    # Attribute     Type                    Default Value
    tag: str = init_field("Aerodynamics", static=True)

    angles: Angles = init_field(Angles)

    coefficients: Coefficients = init_field(Coefficients)
