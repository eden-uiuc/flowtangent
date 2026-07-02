# RCAIDE/Framework/Missions/Conditions/Controls.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Aug 2024, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

import warnings
from typing import Optional, Callable, Literal

# package imports
import jax
import jax.numpy as jnp

# RCAIDE imports
from RCAIDE.utils import DataPath, init_field

from RCAIDE.library import Component

from RCAIDE.framework.conditions import Condition
from RCAIDE.framework.conditions.Stability import StabilityConditions

# ----------------------------------------------------------------------------------------------------------------------
#  Controls
# ----------------------------------------------------------------------------------------------------------------------


def get_active(cond: Condition) -> tuple[Condition, ...]:
    """
    Returns the active controls/residuals
    """

    actives = []

    for c in cond.subconditions:
        if hasattr(c, "_active") and c._active:
            actives.append(c)
        elif hasattr(c, "subconditions") and len(c.subconditions) > 0:
            actives.append(get_active(c))

    return tuple(actives)


class Residual(Condition):
    tag: str = init_field("Dynamic Residual", static=True)
    
    get_value: Callable = init_field(lambda state: jnp.empty(0), as_value=True, static=True)

    _active: bool = init_field(False, static=True)

class DynamicsConditions(Condition):

    tag: str = init_field("Dynamics", static=True)

    @property
    def active_residuals(self) -> tuple[Residual, ...]:
        return get_active(self)


class Control(Condition):
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
    path: DataPath
        Location of the control variable in the overall state data structure.
    initial_guess : float
        An initial guess for the control variable's value. Defaults to None.
    value : np.ndarray
        The current value of the control variable. Initialized as a 1x1 zero array.

    """
    tag: str = init_field("Control", static=True)

    state_path: DataPath = init_field(DataPath, static=True)
    
    # Inital values aren't actually optional, but an unset one will be flagged in State.initialize_controls
    initial_value: Optional[float | jnp.ndarray] = None
    bounds: tuple[float, ...] = tuple((-1e6, 1e6))
    scaling: Literal["linear", "logistic"] = init_field("linear", static=True)

    _active: bool = init_field(False, static=True)

    def get_field_name(self):
        return self.tag.replace(" ", "_").lower()
    
    def scale(self, val):
        if self.scaling == "logistic":
            lb = self.bounds[0]
            ub = self.bounds[1]
            return lb + (ub - lb) * jax.nn.sigmoid(val)
        else:
            return val * self.initial_value
    
    def normalize(self, val):
        if self.scaling == "logistic":
            lb = self.bounds[0]
            ub = self.bounds[1]
            norm = jnp.clip((val - lb) / (ub - lb), 1e-6, 1.0 - 1e-6)
            return jnp.log(norm / (1.0 - norm))
        else:
            return val / self.initial_value
    
    def __post_init__(self):
        if self.bounds[0] > self.bounds[1]:
            warnings.warn(f"Control '{self.tag}' initialized with out-of-order bounds: {self.bounds}. Reversing...")
            rev_bnds = (self.bounds[1], self.bounds[0])
            object.__setattr__(self, "bounds", rev_bnds)

        if self.initial_value is not None:
            clip_value = jnp.clip(self.initial_value, self.bounds[0] * 1.10, self.bounds[1] * 0.90)
            if self.initial_value != clip_value:
                warnings.warn(f"Control '{self.tag}' initialized with out of bounds value {self.initial_value}. Clipping to {clip_value}...")
                object.__setattr__(self, "initial_value", clip_value)

        
        return super().__post_init__()


class SurfaceControl(Control):
    """
    Represents a control variable for a surface in an aircraft or vehicle.

    This class defines the properties of a surface control variable, including its name,
    associated surfaces, deflection, and static stability characteristics. It inherits from
    the Conditions class and uses keyword-only arguments.

    Attributes
    ----------
    tag : str
        The name of the aerodunamic control variable. Defaults to 'Surface Control Variable'.
    surfaces : list[Component]
        A list of surfaces associated with the control variable. Defaults to None.
    deflection : np.ndarray
        An array representing the deflection of the surface. Initialized as a 1x1 zero array.
    static_stability : StaticCoefficients
        An object representing the static stability characteristics of the surface.

    Returns
    -------
    SurfaceControlVariable
        An instance of the SurfaceControlVariable class with the specified attributes.
    """

    # Attribute          Type                        Default Value
    tag: str = init_field("Surface Control Variable", static=True)
    surfaces: tuple[Component] | None = None

    stability: StabilityConditions = init_field(StabilityConditions)


class ControlsConditions(Condition):
    tag: str = init_field("Controls", static=True)

    _default_paths: dict = init_field(lambda: {
        "bank_angle": ("frames", "body", "inertial_rotations"),
        "body_angle": ("frames", "body", "inertial_rotations"),
        "velocity": ("frames", "inertial", "velocity_vector"),
        "altitude": ("frames", "inertial", "position_vector", slice(None, 2)),
    }, static=True)

    def __post_init__(self):
        for ctrl, ctrl_path in self._default_paths.items():
            object.__setattr__(
                self,
                ctrl,
                Control(
                    tag=ctrl.replace('_', " ").title(),
                    state_path=DataPath(ctrl_path),
                ),
            )
        return super().__post_init__()

    @property
    def active_controls(self) -> tuple[Control, ...]:
        return get_active(self)


