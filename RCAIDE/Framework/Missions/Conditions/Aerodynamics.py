# RCAIDE/Framework/Missions/Conditions/Aerodynamics.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Aug 2024, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

# package imports
import equinox as eqx
import jax.numpy as jnp

# RCAIDE imports
from RCAIDE.Framework.Missions.Conditions import Conditions

# ----------------------------------------------------------------------------------------------------------------------
#  Aerodynamics
# ----------------------------------------------------------------------------------------------------------------------


class AerodynamicAngles(Conditions):

    # Attribute         Type        Default Value
    tag:                str         = eqx.field(static=True, default='Aerodynamic Angles')

    alpha:              jnp.ndarray      = eqx.field(default_factory=lambda: jnp.empty(0))
    beta:               jnp.ndarray      = eqx.field(default_factory=lambda: jnp.empty(0))
    phi:                jnp.ndarray      = eqx.field(default_factory=lambda: jnp.empty(0))


class ComponentCoefficients(Conditions):
    tag:            str = eqx.field(static=True, default='Component Coefficients')

    total:          jnp.ndarray = eqx.field(default_factory=lambda: jnp.empty(0))

    # Component Arrays: (n_time, n_components)
    wings:          jnp.ndarray = eqx.field(default_factory=lambda: jnp.empty((0, 0)))
    fuselages:      jnp.ndarray = eqx.field(default_factory=lambda: jnp.empty((0, 0)))
    nacelles:       jnp.ndarray = eqx.field(default_factory=lambda: jnp.empty((0, 0)))


class LiftCoefficients(Conditions):

    # Attribute     Type            Default Value
    tag:            str             = eqx.field(static=True, default='Lift Coefficients')

    total:          jnp.ndarray     = eqx.field(default_factory=lambda: jnp.empty(0))

    inviscid:       ComponentCoefficients   = eqx.field(default_factory=lambda: Conditions(tag='Inviscid Bodies'))
    compressible:   ComponentCoefficients   = eqx.field(default_factory=lambda: Conditions(tag='Compressible Bodies'))


class InducedDrag(Conditions):

    # Attribute   Type            Default Value
    tag:          str             = eqx.field(static=True, default='Induced Drag')

    total:        jnp.ndarray     = eqx.field(default_factory=lambda: jnp.empty(0))

    inviscid:     ComponentCoefficients = eqx.field(default_factory=lambda: ComponentCoefficients(tag='Inviscid Wings'))


class DragCoefficients(Conditions):

    # Attribute     Type            Default Value
    tag:            str             = eqx.field(static=True, default='Drag Coefficients')

    total:          jnp.ndarray     = eqx.field(default_factory=lambda: jnp.empty(0))

    parasite:       ComponentCoefficients = eqx.field(default_factory=lambda: ComponentCoefficients(tag='Parasite Drag'))
    compressible:   ComponentCoefficients = eqx.field(default_factory=lambda: ComponentCoefficients(tag='Compressible Drag'))
    miscellaneous:  ComponentCoefficients = eqx.field(default_factory=lambda: ComponentCoefficients(tag='Miscellaneous Drag'))
    spoiler:        ComponentCoefficients = eqx.field(default_factory=lambda: ComponentCoefficients(tag='Spoiler Drag'))

    induced:        ComponentCoefficients = eqx.field(default_factory=lambda: ComponentCoefficients(tag='Induced Drag'))

class MomentCoefficients(Conditions):

    # Attribute         Type            Default Value
    tag:                str             = eqx.field(static=True, default='Moment Coefficients')

    pitch:              jnp.ndarray     = eqx.field(default_factory=lambda: jnp.empty(0))
    roll:               jnp.ndarray     = eqx.field(default_factory=lambda: jnp.empty(0))
    yaw:                jnp.ndarray     = eqx.field(default_factory=lambda: jnp.empty(0))

class AerodynamicCoefficients(Conditions):

    # Attribute         Type                Default Value
    tag:                str                 = eqx.field(static=True, default='Aerodynamic Coefficients')

    lift:               LiftCoefficients    = eqx.field(default_factory=LiftCoefficients)
    drag:               DragCoefficients    = eqx.field(default_factory=DragCoefficients)
    moments:            MomentCoefficients  = eqx.field(default_factory=MomentCoefficients)


class AerodynamicsConditions(Conditions):

    # Attribute     Type                    Default Value
    tag:            str                     = eqx.field(static=True, default='Aerodynamics')

    angles:         AerodynamicAngles       = eqx.field(default_factory=AerodynamicAngles)

    coefficients:   AerodynamicCoefficients = eqx.field(default_factory=AerodynamicCoefficients)
