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
from eden_trace.utils import empty_array, init_field

from eden_trace.framework.conditions import Condition

# ----------------------------------------------------------------------------------------------------------------------
#  Aerodynamics
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------
#  Coefficients
# ----------------------------------------------------------

# Component-Level Bookkeeping ------------------------------


class ComponentCoefficients(Condition):
    tag: str = init_field("Component Coefficients", static=True)

    total: jnp.ndarray = empty_array()

    # Component Arrays: (n_time, n_components)
    wings: jnp.ndarray = empty_array((0, 0))
    fuselages: jnp.ndarray = empty_array((0, 0))
    nacelles: jnp.ndarray = empty_array((0, 0))


# Lift Coefficients ----------------------------------------


class LiftCoefficients(Condition):
    # Attribute     Type            Default Value
    tag: str = init_field("Lift Coefficients", static=True)

    total: jnp.ndarray = empty_array((0,))

    inviscid: ComponentCoefficients = init_field(lambda: ComponentCoefficients(tag="Inviscid Lift"))
    compressible: ComponentCoefficients = init_field(lambda: ComponentCoefficients(tag="Compressible Lift"))


# Drag Coefficients ----------------------------------------


class InducedDrag(Condition):
    # Attribute   Type            Default Value
    tag: str = init_field("Induced Drag", static=True)

    total: jnp.ndarray = empty_array()

    inviscid: ComponentCoefficients = init_field(lambda: ComponentCoefficients(tag="Inviscid Induced Drag"))
    viscous: ComponentCoefficients = init_field(lambda: ComponentCoefficients(tag="Viscous Induced Drag"))
    near_field: ComponentCoefficients = init_field(lambda: ComponentCoefficients(tag="Near-Field Induced Drag"))
    far_field: ComponentCoefficients = init_field(lambda: ComponentCoefficients(tag="Far-Field Induced Drag"))


class DragCoefficients(Condition):
    # Attribute     Type            Default Value
    tag: str = init_field("Drag Coefficients", static=True)

    total: jnp.ndarray = empty_array()

    parasite: ComponentCoefficients = init_field(lambda: ComponentCoefficients(tag="Parasite Drag"))
    compressible: ComponentCoefficients = init_field(lambda: ComponentCoefficients(tag="Compressible Drag"))
    miscellaneous: ComponentCoefficients = init_field(lambda: ComponentCoefficients(tag="Miscellaneous Drag"))
    spoiler: ComponentCoefficients = init_field(lambda: ComponentCoefficients(tag="Spoiler Drag"))

    induced: InducedDrag = init_field(InducedDrag)


# Moment Coefficients --------------------------------------


class MomentCoefficients(Condition):
    # Attribute         Type            Default Value
    tag: str = init_field("Moment Coefficients", static=True)

    pitch: jnp.ndarray = empty_array()
    roll: jnp.ndarray = empty_array()
    yaw: jnp.ndarray = empty_array()


# All Coefficients -----------------------------------------


class AerodynamicCoefficients(Condition):
    # Attribute         Type                Default Value
    tag: str = init_field("Aerodynamic Coefficients", static=True)

    lift: LiftCoefficients = init_field(LiftCoefficients)
    drag: DragCoefficients = init_field(DragCoefficients)

    moments: MomentCoefficients = init_field(MomentCoefficients)

    X: jnp.ndarray = empty_array()
    Y: jnp.ndarray = empty_array()
    Z: jnp.ndarray = empty_array()


# ----------------------------------------------------------
#  Aerodynamic Angles
# ----------------------------------------------------------


class AerodynamicAngles(Condition):
    # Attribute         Type        Default Value
    tag: str = init_field("Aerodynamic Angles", static=True)

    alpha: jnp.ndarray = empty_array()  # Y-axis / angle of attack
    beta: jnp.ndarray = empty_array()  # Z-axis / sideslip angle
    phi: jnp.ndarray = empty_array()  # X-axis / roll angle


# ----------------------------------------------------------
#  Full Aerodynamic Conditions
# ----------------------------------------------------------


class AerodynamicsConditions(Condition):
    # Attribute     Type                    Default Value
    tag: str = init_field("Aerodynamics", static=True)

    angles: AerodynamicAngles = init_field(AerodynamicAngles)

    coefficients: AerodynamicCoefficients = init_field(AerodynamicCoefficients)
