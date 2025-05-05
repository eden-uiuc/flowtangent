# RCAIDE/Framework/Missions/Initialization/energy.py
# (c) Copyright 2024 Aerospace Research Community LLC
# Created: Aug 2024, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------

# package imports
import numpy as np

# RCAIDE Imports
import RCAIDE.Framework as rcf
from RCAIDE.Framework.Missions.Conditions import Conditions

# ----------------------------------------------------------------------------------------------------------------------
# Initialize Energy
# ----------------------------------------------------------------------------------------------------------------------


def initialize_energy(state: "rcf.State",
                      system: "rcf.System",
                      settings: "rcf.Settings",
                      ):
    def _recursive_initialize_energy(conditions: Conditions, initial_conditions: Conditions):

        for k, v in vars(conditions).items():

            if isinstance(v, np.ndarray):
                v[:, 0] = vars(initial_conditions)[k][-1, 0]
            elif isinstance(v, int) or isinstance(v, float):
                v = vars(initial_conditions)[k]
            if isinstance(v, Conditions):
                _recursive_initialize_energy(v, vars(initial_conditions)[k])

    _recursive_initialize_energy(state.energy, state.initials.energy)

    return state, system, settings