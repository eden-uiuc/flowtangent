# flowtangent/Framework/Missions/Conditions/Aerodynamics.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Aug 2024, FlowTangent Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

# package imports

from flowtangent.core._state_data import StateData

# FlowTangent imports
from ...utils import field
from ...utils.typing import TimeScalar, _

# ----------------------------------------------------------------------------------------------------------------------
#  Aerodynamics
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------
#  Coefficients
# ----------------------------------------------------------

# Component-Level Bookkeeping ------------------------------


class ComponentCoeffs(StateData):
    total: TimeScalar = _

    wings: TimeScalar = _
    fuselages: TimeScalar = _
    nacelles: TimeScalar = _


# Lift Coefficients ----------------------------------------


class LiftCoeffs(StateData):
    total: TimeScalar = _

    inviscid: ComponentCoeffs = field(lambda: ComponentCoeffs(name="Inviscid Lift"))
    compressible: ComponentCoeffs = field(lambda: ComponentCoeffs(name="Compressible Lift"))


# Drag Coefficients ----------------------------------------


class InducedDrag(StateData):
    total: TimeScalar = _

    inviscid: ComponentCoeffs = field(lambda: ComponentCoeffs(name="Inviscid Induced Drag"))
    viscous: ComponentCoeffs = field(lambda: ComponentCoeffs(name="Viscous Induced Drag"))
    near_field: ComponentCoeffs = field(lambda: ComponentCoeffs(name="Near-Field Induced Drag"))
    far_field: ComponentCoeffs = field(lambda: ComponentCoeffs(name="Far-Field Induced Drag"))


class DragCoeffs(StateData):
    total: TimeScalar = _

    parasite: ComponentCoeffs = field(lambda: ComponentCoeffs(name="Parasite Drag"))
    compressible: ComponentCoeffs = field(lambda: ComponentCoeffs(name="Compressible Drag"))
    miscellaneous: ComponentCoeffs = field(lambda: ComponentCoeffs(name="Miscellaneous Drag"))
    spoiler: ComponentCoeffs = field(lambda: ComponentCoeffs(name="Spoiler Drag"))

    induced: InducedDrag = field(InducedDrag)


# Moment Coefficients --------------------------------------


class MomentCoeffs(StateData):
    pitch: TimeScalar = _
    roll: TimeScalar = _
    yaw: TimeScalar = _


# All Coefficients -----------------------------------------


class AeroCoefficients(StateData):
    lift: LiftCoeffs = field(LiftCoeffs)
    drag: DragCoeffs = field(DragCoeffs)

    moments: MomentCoeffs = field(MomentCoeffs)

    X: TimeScalar = _
    Y: TimeScalar = _
    Z: TimeScalar = _


# ----------------------------------------------------------
#  Aerodynamic Angles
# ----------------------------------------------------------


class AeroAngles(StateData):
    alpha: TimeScalar = _  # Y-axis / angle of attack
    beta: TimeScalar = _  # Z-axis / sideslip angle
    phi: TimeScalar = _  # X-axis / roll angle


# ----------------------------------------------------------
#  Full Aerodynamic Conditions
# ----------------------------------------------------------


class Aerodynamics(StateData):
    # Attribute     Type                    Default Value

    angles: AeroAngles = field(AeroAngles)
    coefficients: AeroCoefficients = field(AeroCoefficients)
