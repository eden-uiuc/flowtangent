# Trace/Framework/Missions/Initialization/energy.py
# (c) Copyright 2024 Aerospace Research Community LLC
# Created: Aug 2024, Trace Team

# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eden_trace.framework import Settings, State, System

# package imports
import equinox as eqx
import jax.numpy as jnp

from eden_trace.library.components.energy.networks import GraphNetwork

# Trace Imports
from eden_trace.framework.conditions.energy import NodeConditions, TurbojetNetworkConditions, TurbofanNetworkConditions

# ----------------------------------------------------------------------------------------------------------------------
# Initialize Energy
# ----------------------------------------------------------------------------------------------------------------------


def initialize_energy(state: State, system: System, settings: Settings):
    node_states = {}
    conditions_map = {
        "TurbojetNetwork": TurbojetNetworkConditions,
        "TurbofanNetwork": TurbofanNetworkConditions,
    }

    def _extract_to_flat_state(n):
        if str(n.__class__.__name__) in conditions_map:
            node_states[n.network_ID] = conditions_map[str(n.__class__.__name__)](
                tag=n.network_ID
            )  # Initialize the state
        else:
            node_states[n.network_ID] = NodeConditions(tag=n.network_ID)  # Initialize the state
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

    return updated_state, updated_system, settings
