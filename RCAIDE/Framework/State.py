# RCAIDE/Framework/State.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Jul 2024, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------


from functools import reduce

# package imports
import jax

import equinox as eqx
import numpy as np
import jax.numpy as jnp

import RCAIDE.Library.Components
# RCAIDE imports
from RCAIDE.Framework.Missions.Conditions import (
    Conditions, Numerics, FrameConditions, FreestreamConditions, MassConditions, EnergyNetworkConditions,
    AerodynamicsConditions, ControlsConditions, DynamicsConditions)

# ----------------------------------------------------------------------------------------------------------------------
#  State
# ----------------------------------------------------------------------------------------------------------------------

class State(Conditions):

    # Attribute         Type                        Default Value
    tag:                str                         = eqx.field(static=True, default='State')
    
    initials:           eqx.Module | None           = None
    numerics:           Numerics                    = eqx.field(default_factory=Numerics)

    frames:             FrameConditions             = eqx.field(default_factory=FrameConditions)
    freestream:         FreestreamConditions        = eqx.field(default_factory=FreestreamConditions)

    mass:               MassConditions              = eqx.field(default_factory=MassConditions)
    energy:             EnergyNetworkConditions     = eqx.field(default_factory=EnergyNetworkConditions)
    aerodynamics:       AerodynamicsConditions      = eqx.field(default_factory=AerodynamicsConditions)

    controls:           ControlsConditions          = eqx.field(default_factory=ControlsConditions)
    dynamics:           DynamicsConditions          = eqx.field(default_factory=DynamicsConditions)

    unknowns:           jnp.ndarray                 = eqx.field(default_factory=lambda: jnp.zeros((1, 1)))
    residuals:          jnp.ndarray                 = eqx.field(default_factory=lambda: jnp.zeros((1, 1)))

    def check_controls(self, verbose=True) -> bool:
        """
        Checks that the number of active controls is equal to the number of active dynamics residuals.
        """

        valid_controls = (self.controls.count_active_controls() == self.dynamics.count_active_residuals())

        if verbose:
            if valid_controls:
                print("Number of active controls matches number of active dynamics residuals.")
            else:
                print("Number of active controls does not match number of active dynamics residuals.")

            print(f"\nCurrent active controls:")
            for control in self.controls.get_active_controls():
                print(f"- {control.tag}")

            print(f"\nCurrent active dynamics residuals:")
            for residual in self.dynamics.get_active_residuals():
                print(f"- {residual.tag}")

        return valid_controls
    
    def expand_rows(self, n_control_points: int):

        def _expand(leaf):
            if isinstance(leaf, (jnp.ndarray, np.ndarray)):
                if leaf.ndim == 1:
                    return jnp.tile(leaf, (n_control_points, 1))
                elif leaf.ndim == 2 and leaf.shape[0] == 1:
                    return jnp.repeat(leaf, n_control_points, axis=0)
            return leaf
        
        return jax.tree_util.tree_map(_expand, self)
    
    def unpack_unknowns(self, unknowns):
        """
        Finds the active control variables and assigns the unknowns to their locations in state.
        """

        n_points    = int(self.numerics.number_of_control_points)
        control_idx = 0

        current_state = self

        for control_var in self.controls.get_active_controls():
    
            values  = unknowns[control_idx : control_idx + n_points] # Extract control values from unknowns
            where   = lambda s, p=control_var.path: reduce(getattr, p, s)
            
            current_array   = where(current_state)
            new_array       = current_array.at[control_var.path_indices].set(values)
            
            current_state = eqx.tree_at(where, current_state, new_array)

            control_idx += n_points
        
        return current_state

    def pack_residuals(self):
        """
        Pulls the active residual values from the dynamics and packs them into the residuals array.
        """

        n_points = self.numerics.number_of_control_points
        residual_list = []

        for field_name in self.dynamics.__dataclass_fields__:
            residual = getattr(self.dynamics, field_name)
            
            if getattr(residual, 'active', False):
                residual_list.append(residual.value)

        if residual_list:
            stacked_residuals = jnp.concatenate(residual_list)
        else:
            stacked_residuals = jnp.empty((0,))

        return eqx.tree_at(lambda s: s.residuals, self, stacked_residuals)

    #TODO: Update this to use Equinox methods
    def build_controls_from_system(self, system: "System|Component", verbose=True) -> None:
        if verbose:
            print(f"Building controls from {system.tag}...")
        for component in system.subcomponents:
            self.build_controls_from_system(component)

            if system.is_control_component:
                if isinstance(component, RCAIDE.Library.Components.Wing):
                    self.controls.add_control_variable(
                        RCAIDE.Framework.Missions.SurfaceControlVariable(
                            tag=component.tag + "_deflection",
                            surfaces=[system],
                        )
                    )

                else:
                    self.controls.add_control_variable(
                        RCAIDE.Framework.Missions.ControlVariable(
                            tag=component.tag
                        )
                    )
                if verbose:
                    print(f"\tAdded controls.{system.get_field_name()} as a control variable.")
        if verbose:
            print(f"Completed building controls from {system.tag}.\n"
                  f"Controls may be activated for mission segments by setting "
                  f"segment.active_controls = ('control_variable', ...)")
        return


