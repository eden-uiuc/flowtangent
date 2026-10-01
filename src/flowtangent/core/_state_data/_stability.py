# flowtangent/Framework/Missions/Conditions/Stability.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Aug 2024, FlowTangent Team

# ----------------------------------------------------------------------------------------------------------------------
#  Import
# ----------------------------------------------------------------------------------------------------------------------


# package imports

from flowtangent.core._state_data import StateData

# FlowTangent imports
from ...utils import field
from ...utils.typing import TimeScalar, _

# ----------------------------------------------------------------------------------------------------------------------
#  Stability
# ----------------------------------------------------------------------------------------------------------------------


class StaticCoeffs(StateData):
    """
    Static stability coefficients for an aircraft.

    This class encapsulates various aerodynamic coefficients and forces
    relevant to the static stability analysis of an aircraft.

    Attributes
    ----------
    name : str
        The name of the coefficient set.
    lift : jax.Array
        The lift coefficient. Shape: (1, 1)
    drag : jax.Array
        The drag coefficient. Shape: (1, 1)
    X : jax.Array
        The X-axis force coefficient. Shape: (1, 1)
    Y : jax.Array
        The Y-axis force coefficient. Shape: (1, 1)
    Z : jax.Array
        The Z-axis force coefficient. Shape: (1, 1)
    L : jax.Array
        The rolling moment coefficient. Shape: (1, 1)
    M : jax.Array
        The pitching moment coefficient. Shape: (1, 1)
    N : jax.Array
        The yawing moment coefficient. Shape: (1, 1)
    e : jax.Array
        The Oswald efficiency factor. Shape: (1, 1)

    Notes
    -----
    All coefficient arrays are initialized as 1x1 numpy arrays with zero values.
    These can be updated with actual coefficient values during analysis.
    """

    # Attribute     Type        Default Value
    name: str = field("Static Stability Coefficients", static=True)

    lift: TimeScalar = _
    drag: TimeScalar = _

    X: TimeScalar = _
    Y: TimeScalar = _
    Z: TimeScalar = _

    L: TimeScalar = _
    M: TimeScalar = _
    N: TimeScalar = _

    e: TimeScalar = _


class StaticForces(StateData):
    """
    Static forces acting on an aircraft.

    This class encapsulates the static forces that are relevant to aircraft
    stability analysis, including lift, drag, and forces in the X, Y, and Z directions.

    Attributes
    ----------
    name : str
        The name of the static forces set. Default is 'Static Stability Forces'.
    lift : jax.Array
        The lift force. Shape: (1, 1)
    drag : jax.Array
        The drag force. Shape: (1, 1)
    X : jax.Array
        The force in the X-direction. Shape: (1, 1)
    Y : jax.Array
        The force in the Y-direction. Shape: (1, 1)
    Z : jax.Array
        The force in the Z-direction. Shape: (1, 1)

    Notes
    -----
    All force arrays are initialized as 1x1 numpy arrays with zero values.
    These can be updated with actual force values during analysis.
    """

    # Attribute     Type        Default Value
    name: str = field("Static Stability Forces", static=True)

    lift: TimeScalar = _
    drag: TimeScalar = _

    X: TimeScalar = _
    Y: TimeScalar = _
    Z: TimeScalar = _


class StaticMoments(StateData):
    """
    Represents the static moments acting on an aircraft.

    This class encapsulates the static moments that are relevant to aircraft
    stability analysis, including rolling, pitching, and yawing moments.

    Attributes
    ----------
    name : str
        The name of the static moments set. Default is 'Static Stability Moments'.
    L : jax.Array
        The rolling moment. Shape: (1, 1)
    M : jax.Array
        The pitching moment. Shape: (1, 1)
    N : jax.Array
        The yawing moment. Shape: (1, 1)

    Notes
    -----
    All moment arrays are initialized as 1x1 numpy arrays with zero values.
    These can be updated with actual moment values during analysis.
    """

    # Attribute     Type        Default Value
    name: str = field("Static Stability Moments", static=True)

    L: TimeScalar = _
    M: TimeScalar = _
    N: TimeScalar = _


class Sensitivities(StateData):
    """
    Represents the coefficient derivatives for static stability analysis of an aircraft.

    This class encapsulates various coefficient derivatives related to stability axis
    and body axis, which are crucial for analyzing the static stability characteristics
    of an aircraft.

    Attributes:
    ----------
    name : str
        The name of the coefficient derivatives set. Default is 'Coefficient Static Stability Derivatives'.

    alpha : jax.Array
        Derivative with respect to angle of attack. Shape: (1, 1)
    beta : jax.Array
        Derivative with respect to sideslip angle. Shape: (1, 1)

    delta_a : jax.Array
        Derivative with respect to aileron deflection. Shape: (1, 1)
    delta_e : jax.Array
        Derivative with respect to elevator deflection. Shape: (1, 1)
    delta_r : jax.Array
        Derivative with respect to rudder deflection. Shape: (1, 1)
    delta_f : jax.Array
        Derivative with respect to flap deflection. Shape: (1, 1)
    delta_s : jax.Array
        Derivative with respect to spoiler deflection. Shape: (1, 1)

    u : jax.Array
        Derivative with respect to forward velocity. Shape: (1, 1)
    v : jax.Array
        Derivative with respect to lateral velocity. Shape: (1, 1)
    w : jax.Array
        Derivative with respect to vertical velocity. Shape: (1, 1)

    p : jax.Array
        Derivative with respect to roll rate. Shape: (1, 1)
    q : jax.Array
        Derivative with respect to pitch rate. Shape: (1, 1)
    r : jax.Array
        Derivative with respect to yaw rate. Shape: (1, 1)

    Notes:
    -----
    All derivative arrays are initialized as 1x1 numpy arrays with zero values.
    These can be updated with actual derivative values during analysis.
    """

    # Attribute     Type        Default Value
    name: str = field("Coefficient Static Stability Derivatives", static=True)

    # Throttle Derivative
    throttle: TimeScalar = _

    # Stability Axis Derivatives
    beta: TimeScalar = _
    alpha: TimeScalar = _

    delta_a: TimeScalar = _
    delta_e: TimeScalar = _
    delta_r: TimeScalar = _
    delta_f: TimeScalar = _
    delta_s: TimeScalar = _

    # Body Axis Derivatives

    u: TimeScalar = _
    v: TimeScalar = _
    w: TimeScalar = _

    p: TimeScalar = _
    q: TimeScalar = _
    r: TimeScalar = _


class StaticDerivatives(StateData):
    """
    Represents the static stability coefficient derivatives for an aircraft.

    This class encapsulates various coefficient derivatives related to static stability
    analysis, including lift, drag, and force/moment coefficients in different axes.

    Attributes:
    ----------
    name : str
        The name of the static derivatives set. Default is 'Static Stability Coefficients Derivatives'.
    Clift : CoefficientDerivatives
        Lift coefficient static stability derivatives.
    Cdrag : CoefficientDerivatives
        Drag coefficient static stability derivatives.
    CX : CoefficientDerivatives
        X-axis force coefficient static stability derivatives.
    CY : CoefficientDerivatives
        Y-axis force coefficient static stability derivatives.
    CZ : CoefficientDerivatives
        Z-axis force coefficient static stability derivatives.
    CL : CoefficientDerivatives
        Rolling moment coefficient static stability derivatives.
    CM : CoefficientDerivatives
        Pitching moment coefficient static stability derivatives.
    CN : CoefficientDerivatives
        Yawing moment coefficient static stability derivatives.

    Notes:
    -----
    All coefficient derivatives are instances of the CoefficientDerivatives class,
    allowing for detailed representation of stability characteristics in various axes and conditions.
    """

    # Attribute     Type            Default Value
    name: str = field("Static Stability Coefficients Derivatives", static=True)

    Clift: Sensitivities = field(lambda: Sensitivities(name="Lift Coefficient Static Stability Derivatives"))
    Cdrag: Sensitivities = field(lambda: Sensitivities(name="Drag Coefficient Static Stability Derivatives"))

    CX: Sensitivities = field(lambda: Sensitivities(name="X Coefficient Static Stability Derivatives"))
    CY: Sensitivities = field(lambda: Sensitivities(name="Y Coefficient Static Stability Derivatives"))
    CZ: Sensitivities = field(lambda: Sensitivities(name="Z Coefficient Static Stability Derivatives"))

    CL: Sensitivities = field(lambda: Sensitivities(name="L Coefficient Static Stability Derivatives"))
    CM: Sensitivities = field(lambda: Sensitivities(name="M Coefficient Static Stability Derivatives"))
    CN: Sensitivities = field(lambda: Sensitivities(name="N Coefficient Static Stability Derivatives"))


class Static(StateData):
    name: str = field("Static Stability", static=True)

    forces: StaticForces = field(StaticForces)
    moments: StaticMoments = field(StaticMoments)

    coefficients: StaticCoeffs = field(StaticCoeffs)
    derivatives: StaticDerivatives = field(StaticDerivatives)

    static_margin: TimeScalar = _
    neutral_point: TimeScalar = _
    spiral_criteria: TimeScalar = _

    pitch_rate: TimeScalar = _
    roll_rate: TimeScalar = _
    yaw_rate: TimeScalar = _


class Dynamic(StateData):
    # Attribute      Type        Default Value
    name: str = field("Dynamic Stability", static=True)

    LongModes: StateData = field(lambda: StateData(name="Longitudinal Modes"))
    LatModes: StateData = field(lambda: StateData(name="Lateral Modes"))


class StabilityData(StateData):
    # Attribute     Type                Default Value
    name: str = field("Stability", static=True)

    static: Static = field(Static)
    dynamic: Dynamic = field(Dynamic)
