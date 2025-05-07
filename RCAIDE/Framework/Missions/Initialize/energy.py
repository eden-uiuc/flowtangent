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
from RCAIDE.Framework.Missions.Conditions.Energy import EnergyLineConditions as Line
from RCAIDE.Framework.Missions.Conditions.Energy import EnergyStoreConditions as Store
from RCAIDE.Framework.Missions.Conditions.Energy import EnergyConverterConditions as Converter


# ----------------------------------------------------------------------------------------------------------------------
# Initialize Energy
# ----------------------------------------------------------------------------------------------------------------------


def initialize_energy(state: "rcf.State",
                      system: "rcf.Aircraft",
                      settings: "rcf.Settings",
                      ):

    for l_idx, line in enumerate(system.energy.lines):
        state.energy.lines[l_idx] = Line()
        for p_idx, propulsor in enumerate(line.propulsors):
            state.energy.lines[l_idx].propulsors[p_idx] = Converter()
            state.energy.lines[l_idx].propulsors[p_idx].propulsors = Converter()
            for converter in propulsor.converters:
                state.energy.lines[l_idx].propulsors[p_idx].propulsors[converter.get_field_name()] = Converter()
        for store in enumerate(line.stores):
            state.energy.lines[l_idx].stores[store.get_field_name()] = Store()



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
