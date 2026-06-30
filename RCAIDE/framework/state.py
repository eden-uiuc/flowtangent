# RCAIDE/Framework/State.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Jul 2024, RCAIDE Team
# Modified: Mar 2026, J.Smart

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from functools import reduce

import equinox as eqx

# package imports
import jax
import jax.numpy as jnp

from RCAIDE.utils import init_field, get_target

from RCAIDE.framework.conditions import (
    AerodynamicsConditions,
    ControlsConditions,
    DynamicsConditions,
    EnergyNetworkConditions,
    FrameConditions,
    FreestreamConditions,
    MassConditions,
    Numerics,
    StabilityConditions,
    Condition,
)

# ----------------------------------------------------------------------------------------------------------------------
#  State
# ----------------------------------------------------------------------------------------------------------------------

class State(Condition):

    tag: str = init_field("State", static=True)

    initials: eqx.Module | None = None
    numerics: Numerics = init_field(Numerics)

    frames: FrameConditions = init_field(FrameConditions)
    freestream: FreestreamConditions = init_field(FreestreamConditions)

    mass: MassConditions = init_field(MassConditions)
    energy: EnergyNetworkConditions = init_field(EnergyNetworkConditions)
    aerodynamics: AerodynamicsConditions = init_field(AerodynamicsConditions)
    stability: StabilityConditions = init_field(StabilityConditions)

    controls: ControlsConditions = init_field(ControlsConditions)
    dynamics: DynamicsConditions = init_field(DynamicsConditions)

    def __post_init__(self):
        frozen_initials = eqx.tree_at(lambda s: s.initials, self, None, is_leaf=lambda x: x is None)
        object.__setattr__(self, "initials", frozen_initials)
    
    def initialize_controls(self):
        
        control_values = []

        for ctrl in self.controls.active_controls:
            n_cp = int(self.numerics.number_of_control_points)

            # All control values are normalized by their initial value, so set initial control value to 1.0
            # Values are rescaled in update_controls when actually added to state
            if ctrl.initial_value is not None:
                    control_values.append(jnp.full((n_cp, 1), 1.0))
            else:
                raise ValueError(f"Control {ctrl.tag} has no initial value: {ctrl.initial_value}."
                                "Initial value must be a float or an array of size matching the number of analysis control points.")
        
        return self.update_controls(jnp.concatenate(control_values, axis=0))

    def update_controls(self, control_values: jnp.ndarray):
        
        def apply_bounds(val, lb, ub):
            return lb + (ub - lb) * jax.nn.sigmoid(val)

        updated_state = self
        n_points = int(self.numerics.number_of_control_points)
        control_idx = 0

        # Slice and set (Step through by n_cp to accomodate solvers which return 1D arrays)
        for ctrl in self.controls.active_controls:
            # The solver's guess is already in dimensionless logit space
            solver_logit = control_values[control_idx : control_idx + n_points]
            
            new_values = apply_bounds(solver_logit, ctrl.bounds[0], ctrl.bounds[1])
            
            updated_state = eqx.tree_at(lambda s: get_target(s, ctrl.state_path), updated_state, new_values)
            control_idx += n_points

        return updated_state
    
    def get_control_array(self) -> jnp.ndarray:
        
        def invert_bounds(val, lb, ub):
            # Fixed the 1e6 typo to 1e-6
            norm = jnp.clip((val - lb) / (ub - lb), 1e-6, 1.0 - 1e-6)
            return jnp.log(norm / (1.0 - norm))
        
        control_values = []
        for ctrl in self.controls.active_controls:
            current_physical_val = get_target(self, ctrl.state_path)
            
            # Map the physical value directly to logit space based on its bounds
            logit_val = invert_bounds(current_physical_val, ctrl.bounds[0], ctrl.bounds[1])
            
            control_values.append(logit_val)
                                
        control_array = jnp.concatenate(control_values, axis=0)
        return control_array.flatten()
    
    def get_residual_array(self) -> jnp.ndarray:
        residual_values = [r.get_value(self) for r in self.dynamics.active_residuals]
        residual_array = jnp.concatenate(residual_values, axis=0)
        return residual_array.flatten()
