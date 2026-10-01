# flowtangent/Framework/Missions/Initialization/energy.py
# (c) Copyright 2024 Aerospace Research Community LLC
# Created: Aug 2024, FlowTangent Team

# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ... import Settings, State, System

# package imports

# FlowTangent Imports
from ...components.energy.networks import PACTNetwork
from ...core._state_data._energy import NodeState, TurbofanState, TurbojetState
from ...utils import update

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
            node_states[n.network_id] = conditions_map[str(n.__class__.__name__)](
                name=n.network_id
            )  # Initialize the state
        else:
            node_states[n.network_id] = NodeState(name=n.network_id)  # Initialize the state
        if hasattr(n, "subcomponents"):
            for child in n.subcomponents:
                _extract_to_flat_state(child)

    updated_state = state
    updated_system = system

    for network in updated_system.energy_networks:
        network: PACTNetwork
        updated_network = network.compute_topology()

        for line in updated_network.lines:
            _extract_to_flat_state(line)

        updated_system = updated_system.replace_subcomponent(updated_network)

        if str(network.__class__.__name__) in conditions_map:
            network_state = conditions_map[str(network.__class__.__name__)]()
            updated_state = update(updated_state, "energy", network_state)

        updated_state = update(updated_state, "energy.nodes", node_states)
        updated_state = updated_state.expand_time()

    return updated_state, updated_system, settings
