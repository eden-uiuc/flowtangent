# RCAIDE/Framework/State.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Jul 2024, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

import chex
from dataclasses import field
from typing import Self

# package imports
import numpy as np

# RCAIDE imports
from RCAIDE.Framework.Missions.Conditions import (
    Conditions, Numerics, FrameConditions, FreestreamConditions, MassConditions, EnergyNetworkConditions,
    AerodynamicsConditions, ControlsConditions)

# ----------------------------------------------------------------------------------------------------------------------
#  State
# ----------------------------------------------------------------------------------------------------------------------


@chex.dataclass(kw_only=True)
class State(Conditions):

    # Attribute         Type                        Default Value
    tag:                str                         = 'State'
    initials:           chex.dataclass              = None
    numerics:           Numerics                    = field(default_factory=Numerics)

    frames:             FrameConditions             = field(default_factory=FrameConditions)
    freestream:         FreestreamConditions        = field(default_factory=FreestreamConditions)

    mass:               MassConditions              = field(default_factory=MassConditions)
    energy:             EnergyNetworkConditions     = field(default_factory=EnergyNetworkConditions)

    aerodynamics:       AerodynamicsConditions      = field(default_factory=AerodynamicsConditions)
    controls:           ControlsConditions          = field(default_factory=ControlsConditions)

    unknowns:           np.ndarray                  = field(default_factory=lambda: np.zeros((1, 1)))
    residuals:          np.ndarray                  = field(default_factory=lambda: np.zeros((1, 1)))
    objective:          np.ndarray                  = field(default_factory=lambda: np.zeros((1, 1)))

    def unpack_unknowns(self):
        """
        Finds the active control variables and assigns the unknowns to their locations in state.
        """

        n_points    = self.numerics.number_of_control_points
        control_idx = 0

        for name, control_var in vars(self.controls).items():
            if hasattr(control_var, 'active') and control_var.active:
                values = self.unknowns[control_idx : control_idx + n_points]    # Extract control values from unknowns
                values = np.reshape(values, (-1, 1))                            # Reshape to column vector
                destination = reduce(getattr, control_var.path, state)          # Find destination within state
                destination[control_var.path_indices] = values.flatten()        # Assign to destination in state
                control_idx += n_points

        return

    def pack_residuals(self):
        """
        Pulls the active residual values from the dynamics and packs them into the residuals array.
        """

        n_points = self.numerics.number_of_control_points
        residual_array = np.empty((n_points, 1))

        for name, residual in vars(self.controls.residuals).items():
            if hasattr(residual, 'active') and residual.active:
                residual_array = np.hstack((residual_array, residual.value))

        self.residuals = residual_array

        return


