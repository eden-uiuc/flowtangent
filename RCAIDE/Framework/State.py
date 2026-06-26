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
import jax.numpy as jnp

from RCAIDE.utils import empty_array, init_field

from RCAIDE.Framework.Conditions import (
    AerodynamicsConditions,
    Conditions,
    ControlsConditions,
    DynamicsConditions,
    EnergyNetworkConditions,
    FrameConditions,
    FreestreamConditions,
    MassConditions,
    Numerics,
    StabilityConditions,
)

# ----------------------------------------------------------------------------------------------------------------------
#  State
# ----------------------------------------------------------------------------------------------------------------------

class SolverConditions(Conditions):

    tag: str = init_field('Solver Conditions', static=True)

    unknowns:           jnp.ndarray                 = empty_array(0)
    residuals:          jnp.ndarray                 = empty_array(0)


class State(Conditions):

    # Attribute         Type                        Default Value
    tag:                str                         = init_field('State', static=True)

    initials:           eqx.Module | None           = None
    numerics:           Numerics                    = init_field(Numerics)

    frames:             FrameConditions             = init_field(FrameConditions)
    freestream:         FreestreamConditions        = init_field(FreestreamConditions)

    mass:               MassConditions              = init_field(MassConditions)
    energy:             EnergyNetworkConditions     = init_field(EnergyNetworkConditions)
    aerodynamics:       AerodynamicsConditions      = init_field(AerodynamicsConditions)
    stability:          StabilityConditions         = init_field(StabilityConditions)

    controls:           ControlsConditions          = init_field(ControlsConditions)
    dynamics:           DynamicsConditions          = init_field(DynamicsConditions)

    solver:             SolverConditions            = init_field(SolverConditions)

    def __post_init__(self):
        frozen_initials = eqx.tree_at(lambda s: s.initials, self, None, is_leaf=lambda x: x is None)
        object.__setattr__(self, "initials", frozen_initials)

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

            print("\nCurrent active controls:")
            for control in self.controls.get_active_controls():
                print(f"- {control.tag}")

            print("\nCurrent active dynamics residuals:")
            for residual in self.dynamics.get_active_residuals():
                print(f"- {residual.tag}")

        return valid_controls

    def unpack_unknowns(self, unknowns):
        n_points = int(self.numerics.number_of_control_points)

        # 1. Grab the perfectly static routing table
        routing_table = self.controls.active_routing_table

        # 2. Extract all targets in one shot
        def get_unknown_targets(s):
            targets = []
            for path, _ in routing_table:
                targets.append(reduce(getattr, path, s))
            return tuple(targets)

        current_targets = get_unknown_targets(self)
        new_arrays = []
        control_idx = 0

        # 3. Slice and set
        for i, (_, path_indices) in enumerate(routing_table):
            values = unknowns[control_idx : control_idx + n_points]

            current_array = current_targets[i]
            new_array = current_array.at[path_indices].set(values)

            new_arrays.append(new_array)
            control_idx += n_points

        # 4. Swap all arrays in a single JAX graph node
        return eqx.tree_at(get_unknown_targets, self, tuple(new_arrays))

    def pack_residuals(self):
        """
        Pulls the active residual values from the dynamics and packs them into the residuals array.
        """

        residual_list = []

        residual_list = [res.value for res in self.dynamics.get_active_residuals()]

        if residual_list:
            stacked_residuals = jnp.concatenate(residual_list)
        else:
            stacked_residuals = jnp.empty((0,))

        return eqx.tree_at(lambda s: s.solver.residuals, self, stacked_residuals)


