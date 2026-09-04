# flowtangent/Framework/Missions/Conditions/Stability.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Aug 2024, Flowtangent Team

# ----------------------------------------------------------------------------------------------------------------------
#  Import
# ----------------------------------------------------------------------------------------------------------------------


# package imports
import jax.numpy as jnp

# Flowtangent imports
from flowtangent.utils import empty_array, field, register
from flowtangent.framework.state_data import StateData

# ----------------------------------------------------------------------------------------------------------------------
#  Stability
# ----------------------------------------------------------------------------------------------------------------------


@register
class StaticCoeffs(StateData):
    """
    Static stability coefficients for an aircraft.

    This class encapsulates various aerodynamic coefficients and forces
    relevant to the static stability analysis of an aircraft.

    Attributes
    ----------
    name : str
        The name of the coefficient set.
    lift : jnp.ndarray
        The lift coefficient. Shape: (1, 1)
    drag : jnp.ndarray
        The drag coefficient. Shape: (1, 1)
    X : jnp.ndarray
        The X-axis force coefficient. Shape: (1, 1)
    Y : jnp.ndarray
        The Y-axis force coefficient. Shape: (1, 1)
    Z : jnp.ndarray
        The Z-axis force coefficient. Shape: (1, 1)
    L : jnp.ndarray
        The rolling moment coefficient. Shape: (1, 1)
    M : jnp.ndarray
        The pitching moment coefficient. Shape: (1, 1)
    N : jnp.ndarray
        The yawing moment coefficient. Shape: (1, 1)
    e : jnp.ndarray
        The Oswald efficiency factor. Shape: (1, 1)

    Notes
    -----
    All coefficient arrays are initialized as 1x1 numpy arrays with zero values.
    These can be updated with actual coefficient values during analysis.
    """

    # Attribute     Type        Default Value
    tag: str = field("Static Stability Coefficients", static=True)

    lift: jnp.ndarray = empty_array()
    drag: jnp.ndarray = empty_array()

    X: jnp.ndarray = empty_array()
    Y: jnp.ndarray = empty_array()
    Z: jnp.ndarray = empty_array()

    L: jnp.ndarray = empty_array()
    M: jnp.ndarray = empty_array()
    N: jnp.ndarray = empty_array()

    e: jnp.ndarray = empty_array()


@register
class StaticForces(StateData):
    """
    Static forces acting on an aircraft.

    This class encapsulates the static forces that are relevant to aircraft
    stability analysis, including lift, drag, and forces in the X, Y, and Z directions.

    Attributes
    ----------
    name : str
        The name of the static forces set. Default is 'Static Stability Forces'.
    lift : jnp.ndarray
        The lift force. Shape: (1, 1)
    drag : jnp.ndarray
        The drag force. Shape: (1, 1)
    X : jnp.ndarray
        The force in the X-direction. Shape: (1, 1)
    Y : jnp.ndarray
        The force in the Y-direction. Shape: (1, 1)
    Z : jnp.ndarray
        The force in the Z-direction. Shape: (1, 1)

    Notes
    -----
    All force arrays are initialized as 1x1 numpy arrays with zero values.
    These can be updated with actual force values during analysis.
    """

    # Attribute     Type        Default Value
    tag: str = field("Static Stability Forces", static=True)

    lift: jnp.ndarray = empty_array()
    drag: jnp.ndarray = empty_array()

    X: jnp.ndarray = empty_array()
    Y: jnp.ndarray = empty_array()
    Z: jnp.ndarray = empty_array()


@register
class StaticMoments(StateData):
    """
    Represents the static moments acting on an aircraft.

    This class encapsulates the static moments that are relevant to aircraft
    stability analysis, including rolling, pitching, and yawing moments.

    Attributes
    ----------
    name : str
        The name of the static moments set. Default is 'Static Stability Moments'.
    L : jnp.ndarray
        The rolling moment. Shape: (1, 1)
    M : jnp.ndarray
        The pitching moment. Shape: (1, 1)
    N : jnp.ndarray
        The yawing moment. Shape: (1, 1)

    Notes
    -----
    All moment arrays are initialized as 1x1 numpy arrays with zero values.
    These can be updated with actual moment values during analysis.
    """

    # Attribute     Type        Default Value
    tag: str = field("Static Stability Moments", static=True)

    L: jnp.ndarray = empty_array()
    M: jnp.ndarray = empty_array()
    N: jnp.ndarray = empty_array()


@register
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

    alpha : jnp.ndarray
        Derivative with respect to angle of attack. Shape: (1, 1)
    beta : jnp.ndarray
        Derivative with respect to sideslip angle. Shape: (1, 1)

    delta_a : jnp.ndarray
        Derivative with respect to aileron deflection. Shape: (1, 1)
    delta_e : jnp.ndarray
        Derivative with respect to elevator deflection. Shape: (1, 1)
    delta_r : jnp.ndarray
        Derivative with respect to rudder deflection. Shape: (1, 1)
    delta_f : jnp.ndarray
        Derivative with respect to flap deflection. Shape: (1, 1)
    delta_s : jnp.ndarray
        Derivative with respect to spoiler deflection. Shape: (1, 1)

    u : jnp.ndarray
        Derivative with respect to forward velocity. Shape: (1, 1)
    v : jnp.ndarray
        Derivative with respect to lateral velocity. Shape: (1, 1)
    w : jnp.ndarray
        Derivative with respect to vertical velocity. Shape: (1, 1)

    p : jnp.ndarray
        Derivative with respect to roll rate. Shape: (1, 1)
    q : jnp.ndarray
        Derivative with respect to pitch rate. Shape: (1, 1)
    r : jnp.ndarray
        Derivative with respect to yaw rate. Shape: (1, 1)

    Notes:
    -----
    All derivative arrays are initialized as 1x1 numpy arrays with zero values.
    These can be updated with actual derivative values during analysis.
    """

    # Attribute     Type        Default Value
    tag: str = field("Coefficient Static Stability Derivatives", static=True)

    # Throttle Derivative
    throttle: jnp.ndarray = empty_array()

    # Stability Axis Derivatives
    beta: jnp.ndarray = empty_array()
    alpha: jnp.ndarray = empty_array()

    delta_a: jnp.ndarray = empty_array()
    delta_e: jnp.ndarray = empty_array()
    delta_r: jnp.ndarray = empty_array()
    delta_f: jnp.ndarray = empty_array()
    delta_s: jnp.ndarray = empty_array()

    # Body Axis Derivatives

    u: jnp.ndarray = empty_array()
    v: jnp.ndarray = empty_array()
    w: jnp.ndarray = empty_array()

    p: jnp.ndarray = empty_array()
    q: jnp.ndarray = empty_array()
    r: jnp.ndarray = empty_array()


@register
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
    tag: str = field("Static Stability Coefficients Derivatives", static=True)

    Clift: Sensitivities = field(
        lambda: Sensitivities(tag="Lift Coefficient Static Stability Derivatives")
    )
    Cdrag: Sensitivities = field(
        lambda: Sensitivities(tag="Drag Coefficient Static Stability Derivatives")
    )

    CX: Sensitivities = field(
        lambda: Sensitivities(tag="X Coefficient Static Stability Derivatives")
    )
    CY: Sensitivities = field(
        lambda: Sensitivities(tag="Y Coefficient Static Stability Derivatives")
    )
    CZ: Sensitivities = field(
        lambda: Sensitivities(tag="Z Coefficient Static Stability Derivatives")
    )

    CL: Sensitivities = field(
        lambda: Sensitivities(tag="L Coefficient Static Stability Derivatives")
    )
    CM: Sensitivities = field(
        lambda: Sensitivities(tag="M Coefficient Static Stability Derivatives")
    )
    CN: Sensitivities = field(
        lambda: Sensitivities(tag="N Coefficient Static Stability Derivatives")
    )


@register
class Static(StateData):
    tag: str = field("Static Stability", static=True)

    forces: StaticForces = field(StaticForces)
    moments: StaticMoments = field(StaticMoments)

    coefficients: StaticCoeffs = field(StaticCoeffs)
    derivatives: StaticDerivatives = field(StaticDerivatives)

    static_margin: jnp.ndarray = empty_array()
    neutral_point: jnp.ndarray = empty_array()
    spiral_criteria: jnp.ndarray = empty_array()

    pitch_rate: jnp.ndarray = empty_array()
    roll_rate: jnp.ndarray = empty_array()
    yaw_rate: jnp.ndarray = empty_array()


@register
class Dynamic(StateData):
    # Attribute      Type        Default Value
    tag: str = field("Dynamic Stability", static=True)

    LongModes: StateData = field(lambda: StateData(tag="Longitudinal Modes"))
    LatModes: StateData = field(lambda: StateData(tag="Lateral Modes"))


@register
class StabilityData(StateData):
    # Attribute     Type                Default Value
    tag: str = field("Stability", static=True)

    static: Static = field(Static)
    dynamic: Dynamic = field(Dynamic)
