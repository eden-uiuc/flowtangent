# flowtangent/Framework/Missions/Initialization/energy.py
# (c) Copyright 2024 Aerospace Research Community LLC
# Created: Aug 2024, Flowtangent Team

# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flowtangent.framework import Settings, State, System

# package imports
import equinox as eqx

# Flowtangent Imports
from flowtangent.core._state_data._energy import NodeState, TurbofanState, TurbojetState
from flowtangent.library.components.energy.networks import GraphNetwork

# ----------------------------------------------------------------------------------------------------------------------
# Initialize Energy
# ----------------------------------------------------------------------------------------------------------------------


def initialize_energy(state: State, system: System, settings: Settings):
    node_states = {}
    conditions_map = {
        "TurbojetNetwork": TurbojetState,
        "TurbofanNetwork": TurbofanState,
    }

    def _extract_to_flat_state(n):
        if str(n.__class__.__name__) in conditions_map:
            node_states[n.network_ID] = conditions_map[str(n.__class__.__name__)](
                tag=n.network_ID
            )  # Initialize the state
        else:
            node_states[n.network_ID] = NodeState(tag=n.network_ID)  # Initialize the state
        if hasattr(n, "subcomponents"):
            for child in n.subcomponents:
                _extract_to_flat_state(child)

    updated_state = state
    updated_system = system

    for network in updated_system.energy_networks:
        network: GraphNetwork
        updated_network = network.assign_network_IDs()

        for line in updated_network.lines:
            _extract_to_flat_state(line)

        updated_system = updated_system.replace_subcomponent(updated_network)

        if str(network.__class__.__name__ ) in conditions_map:
            network_state = conditions_map[str(network.__class__.__name__ )]()
            updated_state = eqx.tree_at(lambda s: s.energy, updated_state, network_state)

        updated_state = eqx.tree_at(lambda s: s.energy.nodes, updated_state, node_states)
        updated_state = updated_state.expand_time()

    return updated_state, updated_system, settings
