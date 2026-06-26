# RCAIDE/Framework/Missions/Initialization/energy.py
# (c) Copyright 2024 Aerospace Research Community LLC
# Created: Aug 2024, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from RCAIDE.Framework import Settings, State, System

# package imports
import equinox as eqx

from RCAIDE.Library.Components.Energy.Networks import EnergyNetwork

# RCAIDE Imports
from RCAIDE.Framework.Conditions.Energy import EnergyNodeConditions, FuelTankConditions, TurbojetNetworkConditions

# ----------------------------------------------------------------------------------------------------------------------
# Initialize Energy
# ----------------------------------------------------------------------------------------------------------------------


def initialize_energy(state: State, system: System, settings: Settings):

    flat_state_nodes = {}
    conditions_map = {"FuelTank": FuelTankConditions,
                      "TurbojetEnergyNetwork": TurbojetNetworkConditions}

    def _extract_to_flat_state(n):
        if str(n.__class__.__name__) in conditions_map:
            flat_state_nodes[n.network_ID] = conditions_map[str(n.__class__.__name__)](
                tag=n.network_ID
            )  # Initialize the state
        else:
            flat_state_nodes[n.network_ID] = EnergyNodeConditions(tag=n.network_ID)  # Initialize the state
        if hasattr(n, "subcomponents"):
            for child in n.subcomponents:
                _extract_to_flat_state(child)

    updated_state = state
    updated_system = system

    for network in updated_system.energy_networks:
        network: EnergyNetwork
        network_sc_idx = updated_system.subcomponents.index(network)
        updated_network = network.assign_network_IDs()

        for line in updated_network.lines:
            _extract_to_flat_state(line)

        updated_system = updated_system.replace_subcomponent(updated_network, network_sc_idx)
        updated_state = eqx.tree_at(lambda s: s.energy.nodes, updated_state, flat_state_nodes)

    return updated_state, updated_system, settings
