# RCAIDE/Framework/Missions/Conditions/Aerodynamics.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Aug 2024, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

# package imports
import jax.numpy as jnp

# RCAIDE imports
from RCAIDE.utils import empty_array, init_field

from RCAIDE.Framework.Conditions import Conditions

# ----------------------------------------------------------------------------------------------------------------------
#  Aerodynamics
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------
#  Coefficients
# ----------------------------------------------------------

# Component-Level Bookkeeping ------------------------------

class ComponentCoefficients(Conditions):
    tag:            str = init_field('Component Coefficients', static=True)

    total:          jnp.ndarray = empty_array(0)

    # Component Arrays: (n_time, n_components)
    wings:          jnp.ndarray = empty_array((0, 0))
    fuselages:      jnp.ndarray = empty_array((0, 0))
    nacelles:       jnp.ndarray = empty_array((0, 0))

# Lift Coefficients ----------------------------------------

class LiftCoefficients(Conditions):

    # Attribute     Type            Default Value
    tag:            str             = init_field('Lift Coefficients', static=True)

    total:          jnp.ndarray     = empty_array((0,))

    inviscid:       ComponentCoefficients   = init_field(lambda: ComponentCoefficients(tag='Inviscid Lift'))
    compressible:   ComponentCoefficients   = init_field(lambda: ComponentCoefficients(tag='Compressible Lift'))

# Drag Coefficients ----------------------------------------

class InducedDrag(Conditions):

    # Attribute   Type            Default Value
    tag:          str             = init_field('Induced Drag', static=True)

    total:        jnp.ndarray     = empty_array(0)

    inviscid:     ComponentCoefficients = init_field(lambda: ComponentCoefficients(tag='Inviscid Induced Drag'))
    viscous:      ComponentCoefficients = init_field(lambda: ComponentCoefficients(tag='Viscous Induced Drag'))
    near_field:   ComponentCoefficients = init_field(lambda: ComponentCoefficients(tag='Near-Field Induced Drag'))
    far_field:    ComponentCoefficients = init_field(lambda: ComponentCoefficients(tag='Far-Field Induced Drag'))


class DragCoefficients(Conditions):

    # Attribute     Type            Default Value
    tag:            str             = init_field('Drag Coefficients', static=True)

    total:          jnp.ndarray     = empty_array(0)

    parasite:       ComponentCoefficients = init_field(lambda: ComponentCoefficients(tag='Parasite Drag'))
    compressible:   ComponentCoefficients = init_field(lambda: ComponentCoefficients(tag='Compressible Drag'))
    miscellaneous:  ComponentCoefficients = init_field(lambda: ComponentCoefficients(tag='Miscellaneous Drag'))
    spoiler:        ComponentCoefficients = init_field(lambda: ComponentCoefficients(tag='Spoiler Drag'))

    induced:        InducedDrag = init_field(InducedDrag)

# Moment Coefficients --------------------------------------

class MomentCoefficients(Conditions):

    # Attribute         Type            Default Value
    tag:                str             = init_field('Moment Coefficients', static=True)

    pitch:              jnp.ndarray     = empty_array(0)
    roll:               jnp.ndarray     = empty_array(0)
    yaw:                jnp.ndarray     = empty_array(0)

# All Coefficients -----------------------------------------

class AerodynamicCoefficients(Conditions):

    # Attribute         Type                Default Value
    tag:                str                 = init_field('Aerodynamic Coefficients', static=True)

    lift:               LiftCoefficients    = init_field(LiftCoefficients)
    drag:               DragCoefficients    = init_field(DragCoefficients)

    moments:            MomentCoefficients  = init_field(MomentCoefficients)

    X:                  jnp.ndarray         = empty_array(0)
    Y:                  jnp.ndarray         = empty_array(0)
    Z:                  jnp.ndarray         = empty_array(0)

# ----------------------------------------------------------
#  Aerodynamic Angles
# ----------------------------------------------------------

class AerodynamicAngles(Conditions):

    # Attribute         Type        Default Value
    tag:                str         = init_field('Aerodynamic Angles', static=True)

    alpha:              jnp.ndarray      = empty_array(0) # Y-axis / angle of attack
    beta:               jnp.ndarray      = empty_array(0) # Z-axis / sideslip angle
    phi:                jnp.ndarray      = empty_array(0) # X-axis / roll angle

# ----------------------------------------------------------
#  Full Aerodynamic Conditions
# ----------------------------------------------------------


class AerodynamicsConditions(Conditions):

    # Attribute     Type                    Default Value
    tag:            str                     = init_field('Aerodynamics', static=True)

    angles:         AerodynamicAngles       = init_field(AerodynamicAngles)

    coefficients:   AerodynamicCoefficients = init_field(AerodynamicCoefficients)
