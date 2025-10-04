# RCAIDE/Framework/Missions/Conditions/Controls.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Aug 2024, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

import unittest
import chex
from dataclasses import field

# package imports
import numpy as np

# RCAIDE imports
from RCAIDE.Framework.Missions.Conditions import Conditions
from RCAIDE.Framework.Missions.Conditions.Stability import StaticCoefficients

# ----------------------------------------------------------------------------------------------------------------------
#  Controls
# ----------------------------------------------------------------------------------------------------------------------


@chex.dataclass(kw_only=True)
class Residual(Conditions):

    tag:   str     = 'residual'
    type:   str     = None
    active: bool    = False
    index:  int     = None


@chex.dataclass(kw_only=True)
class DynamicsResiduals(Conditions):
    """
    Represents the dynamics variables for a simulation.

    This class defines the forces and moments acting on an object in three-dimensional space.
    It inherits from the Conditions class and uses keyword-only arguments.

    Attributes
    ----------
    name : str
        The name of the dynamics object.
    force_x : bool
        Indicates if there's a force acting along the x-axis.
    force_y : bool
        Indicates if there's a force acting along the y-axis.
    force_z : bool
        Indicates if there's a force acting along the z-axis.
    moment_x : bool
        Indicates if there's a moment acting around the x-axis.
    moment_y : bool
        Indicates if there's a moment acting around the y-axis.
    moment_z : bool
        Indicates if there's a moment acting around the z-axis.
    """

    tag:       str         = 'Dynamics'

    force_x:    Residual    = field(default_factory=lambda: Residual(tag='F_x', type='force', index=0))
    force_y:    Residual    = field(default_factory=lambda: Residual(tag='F_y', type='force', index=1))
    force_z:    Residual    = field(default_factory=lambda: Residual(tag='F_z', type='force', index=2))
    moment_x:   Residual    = field(default_factory=lambda: Residual(tag='M_x', type='moment', index=0))
    moment_y:   Residual    = field(default_factory=lambda: Residual(tag='M_y', type='moment', index=1))
    moment_z:   Residual    = field(default_factory=lambda: Residual(tag='M_z', type='moment', index=2))


@chex.dataclass(kw_only=True)
class ControlVariable(Conditions):
    """
    Represents a control variable in a simulation or control system.

    This class defines the properties of a control variable, including its name,
    activation status, initial guess, and current value. It inherits from the
    Conditions class and uses keyword-only arguments.

    Attributes
    ----------
    name : str
        The name of the control variable. Defaults to 'Control Variable'.
    active : bool
        Indicates whether the control variable is active or not. Defaults to False.
    initial_guess : float
        An initial guess for the control variable's value. Defaults to None.
    value : np.ndarray
        The current value of the control variable. Initialized as a 1x1 zero array.

    """

    # Attribute     Type        Default Value
    tag:           str         = 'Control Variable'
    path:           tuple[str]  = None
    path_indices:   tuple       = None
    active:         bool        = False
    initial_guess:  float       = None

    value:          np.ndarray  = field(default_factory=lambda: np.zeros((1, 1)))


@chex.dataclass(kw_only=True)
class SurfaceControlVariable(Conditions):
    """
    Represents a control variable for a surface in an aircraft or vehicle.

    This class defines the properties of a surface control variable, including its name,
    associated surfaces, deflection, and static stability characteristics. It inherits from
    the Conditions class and uses keyword-only arguments.

    Attributes
    ----------
    name : str
        The name of the surface control variable. Defaults to 'Surface Control Variable'.
    deflection : np.ndarray
        An array representing the deflection of the surface. Initialized as a 1x1 zero array.
    static_stability : StaticCoefficients
        An object representing the static stability characteristics of the surface.

    Returns
    -------
    SurfaceControlVariable
        An instance of the SurfaceControlVariable class with the specified attributes.
    """

    #Attribute          Type                Default Value
    tag:               str                 = 'Surface Control Variable'

    deflection:         np.ndarray          = field(default_factory=lambda: np.zeros((1, 1)))

    static_stability:   StaticCoefficients  = field(default_factory=lambda: StaticCoefficients())


@chex.dataclass(kw_only=True)
class PropulsionControlVariable(Conditions):
    """
    Represents a control variable for propulsion systems in a vehicle or aircraft.

    This class defines the properties of a propulsion control variable, including its name
    and associated propulsors. It inherits from the Conditions class and uses keyword-only arguments.

    Attributes
    ----------
    name : str
        The name of the propulsion control variable. Defaults to 'Propulsion Control Variable'.
    """

    #Attribute  Type            Default Value
    tag:       str             = 'Propulsion Control Variable'

    value:      np.ndarray      = field(default_factory=lambda: np.zeros((1, 1)))


@chex.dataclass(kw_only=True)
class ControlsConditions(Conditions):
    """
    Represents the control conditions for an aircraft or vehicle simulation.

    This class encapsulates various control variables and dynamics for a comprehensive
    simulation environment. It includes controls for aircraft orientation, propulsion,
    and surface controls.

    Attributes
    ----------
    name : str
        The name of the control conditions, default is 'Controls'.
    dynamics : DynamicsResiduals
        Object representing the dynamic variables of the simulation.

    body_angle : ControlVariable
        Control variable for the body angle.
    bank_angle : ControlVariable
        Control variable for the bank angle.
    wind_angle : ControlVariable
        Control variable for the wind angle.

    elapsed_time : ControlVariable
        Control variable for the elapsed time.
    velocity : ControlVariable
        Control variable for velocity.
    acceleration : ControlVariable
        Control variable for acceleration (Note: named 'Velocity' in initialization).
    altitude : ControlVariable
        Control variable for altitude.

    thrust : PropulsionControlVariable
        Control variable for thrust vector.
    throttle : PropulsionControlVariable
        Control variable for throttle.

    elevator : SurfaceControlVariable
        Control variable for elevator surfaces.
    rudder : SurfaceControlVariable
        Control variable for rudder surfaces.
    flaps : SurfaceControlVariable
        Control variable for flap surfaces.
    slats : SurfaceControlVariable
        Control variable for slat surfaces.
    ailerons : SurfaceControlVariable
        Control variable for aileron surfaces.
    """

    tag:           str                         = 'Controls'

    dynamics:       DynamicsResiduals           = field(default_factory=lambda: DynamicsResiduals())

    bank_angle:     ControlVariable             = field(default_factory=lambda: ControlVariable(tag='Bank Angle'))
    body_angle:     ControlVariable             = field(default_factory=lambda: ControlVariable(tag='Body Angle'))
    wind_angle:     ControlVariable             = field(default_factory=lambda: ControlVariable(tag='Wind Angle'))

    elapsed_time:   ControlVariable             = field(default_factory=lambda: ControlVariable(tag='Elapsed Time'))
    velocity:       ControlVariable             = field(default_factory=lambda: ControlVariable(tag='Velocity'))
    acceleration:   ControlVariable             = field(default_factory=lambda: ControlVariable(tag='Velocity'))
    altitude:       ControlVariable             = field(default_factory=lambda: ControlVariable(tag='Altitude'))

    thrust:         PropulsionControlVariable   = field(default_factory=lambda: PropulsionControlVariable(tag='Thrust'))
    throttle:       PropulsionControlVariable   = field(default_factory=lambda: PropulsionControlVariable(tag='Throttle'))

    elevator:       SurfaceControlVariable      = field(default_factory=lambda: SurfaceControlVariable(tag='Elevator Controls'))
    rudder:         SurfaceControlVariable      = field(default_factory=lambda: SurfaceControlVariable(tag='Rudder Controls'))
    flaps:          SurfaceControlVariable      = field(default_factory=lambda: SurfaceControlVariable(tag='Flap Controls'))
    slats:          SurfaceControlVariable      = field(default_factory=lambda: SurfaceControlVariable(tag='Slat Controls'))
    ailerons:       SurfaceControlVariable      = field(default_factory=lambda: SurfaceControlVariable(tag='Aileron Controls'))

    def __post_init__(self):

        # Map body angle to the Y-axis of the body frame
        self.body_angle.path = ('frames', 'body', 'inertial_rotations')
        self.body_angle.path_indices = (slice(None), 1)

        # Map bank angle to the X-axis of the body frame
        self.bank_angle.path = ('frames', 'body', 'inertial_rotations')
        self.bank_angle.path_indices = (slice(None), 0)

        # Map velocity to the X-axis of the inertial frame
        self.velocity.path = ('frames', 'inertial', 'velocity_vector')
        self.velocity.path_indices = (slice(None), 0)

        # Map altitude to the Z-axis of the inertial frame
        self.altitude.path = ('frames', 'inertial', 'position_vector')
        self.altitude.path_indices = (slice(None), 2)


class TestDynamicsVariables(unittest.TestCase):
    def test_default_values(self):
        dv = DynamicsResiduals()
        self.assertEqual(dv.tag, 'Dynamics')
        self.assertFalse(dv.force_x)
        self.assertFalse(dv.force_y)
        self.assertFalse(dv.force_z)
        self.assertFalse(dv.moment_x)
        self.assertFalse(dv.moment_y)
        self.assertFalse(dv.moment_z)

    def test_custom_values(self):
        dv = DynamicsResiduals(force_x=True, moment_z=True)
        self.assertTrue(dv.force_x)
        self.assertFalse(dv.force_y)
        self.assertTrue(dv.moment_z)


class TestControlVariable(unittest.TestCase):
    def test_default_values(self):
        cv = ControlVariable()
        self.assertEqual(cv.tag, 'Control Variable')
        self.assertFalse(cv.active)
        self.assertIsNone(cv.initial_guess)
        np.testing.assert_array_equal(cv.value, np.zeros((1, 1)))

    def test_custom_values(self):
        cv = ControlVariable(tag='Test', active=True, initial_guess=5.0)
        self.assertEqual(cv.tag, 'Test')
        self.assertTrue(cv.active)
        self.assertEqual(cv.initial_guess, 5.0)


class TestSurfaceControlVariable(unittest.TestCase):
    def test_default_values(self):
        scv = SurfaceControlVariable()
        self.assertEqual(scv.tag, 'Surface Control Variable')
        np.testing.assert_array_equal(scv.deflection, np.zeros((1, 1)))
        self.assertIsInstance(scv.static_stability, StaticCoefficients)


class TestPropulsionControlVariable(unittest.TestCase):
    def test_default_values(self):
        pcv = PropulsionControlVariable()
        self.assertEqual(pcv.tag, 'Propulsion Control Variable')
        np.testing.assert_array_equal(pcv.value, np.zeros((1, 1)))


class TestControlsConditions(unittest.TestCase):
    def test_default_values(self):
        cc = ControlsConditions()
        self.assertEqual(cc.tag, 'Controls')
        self.assertIsInstance(cc.dynamics, DynamicsResiduals)
        self.assertIsInstance(cc.body_angle, ControlVariable)
        self.assertIsInstance(cc.thrust, PropulsionControlVariable)
        self.assertIsInstance(cc.elevator, SurfaceControlVariable)

    def test_custom_values(self):
        cc = ControlsConditions(tag='Custom Controls')
        self.assertEqual(cc.tag, 'Custom Controls')
        self.assertEqual(cc.body_angle.tag, 'Body Angle')
        self.assertEqual(cc.thrust.tag, 'Thrust')
        self.assertEqual(cc.elevator.tag, 'Elevator Controls')


if __name__ == '__main__':
    unittest.main()
