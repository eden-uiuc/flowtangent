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
    tag:                str         = 'Aerodynamic Angles'

    alpha:              jnp.ndarray      = eqx.field(default_factory=lambda: jnp.empty(0))
    beta:               jnp.ndarray      = eqx.field(default_factory=lambda: jnp.empty(0))
    phi:                jnp.ndarray      = eqx.field(default_factory=lambda: jnp.empty(0))



class LiftCoefficients(Conditions):

    # Attribute     Type            Default Value
    tag:            str             = 'Lift Coefficients'

    total:          jnp.ndarray     = eqx.field(default_factory=lambda: jnp.empty(0))

    inviscid:       Conditions      = eqx.field(default_factory=lambda: Conditions(tag='Inviscid Bodies'))
    compressible:   Conditions      = eqx.field(default_factory=lambda: Conditions(tag='Compressible Bodies'))


class InducedDrag(Conditions):

    # Attribute         Type            Default Value
    tag:                str             = 'Induced Drag'

    total:              jnp.ndarray     = eqx.field(default_factory=lambda: jnp.empty(0))

    inviscid_wings:     Conditions      = eqx.field(default_factory=lambda: Conditions(tag='Inviscid Wings'))



class DragCoefficients(Conditions):

    # Attribute         Type            Default Value
    tag:                str             = 'Drag Coefficients'

    total:              jnp.ndarray      = eqx.field(default_factory=lambda: jnp.empty(0))

    parasite:           Conditions      = eqx.field(default_factory=lambda: Conditions(tag='Parasite Drag'))
    compressible:       Conditions      = eqx.field(default_factory=lambda: Conditions(tag='Compressible Drag'))
    miscellaneous:      Conditions      = eqx.field(default_factory=lambda: Conditions(tag='Miscellaneous Drag'))
    spoiler:            Conditions      = eqx.field(default_factory=lambda: Conditions(tag='Spoiler Drag'))

    induced:            InducedDrag     = eqx.field(default_factory=InducedDrag)


class AerodynamicCoefficients(Conditions):

    # Attribute         Type                Default Value
    tag:                str                 = 'Aerodynamic Coefficients'

    lift:               LiftCoefficients    = eqx.field(default_factory=LiftCoefficients)
    drag:               DragCoefficients    = eqx.field(default_factory=DragCoefficients)

class AerodynamicsConditions(Conditions):

    # Attribute     Type                    Default Value
    tag:            str                     = 'Aerodynamics'

    angles:         AerodynamicAngles       = eqx.field(default_factory=AerodynamicAngles)

    coefficients:   AerodynamicCoefficients = eqx.field(default_factory=AerodynamicCoefficients)