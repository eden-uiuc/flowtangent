# RCAIDE/Framework/Missions/Initialization/energy.py
# (c) Copyright 2024 Aerospace Research Community LLC
# Created: Aug 2024, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------

# package imports
import equinox as eqx
import jax.numpy as jnp

# RCAIDE Imports
from RCAIDE.Library.Components.Energy.Stores import FuelTank
from RCAIDE.Library.Components.Energy.Nodes import EnergyInterface

import RCAIDE.Framework as rcf
from RCAIDE.Framework.Missions.Conditions.Energy import *


# ----------------------------------------------------------------------------------------------------------------------
# Initialize Energy
# ----------------------------------------------------------------------------------------------------------------------


def _resolve_namespaces(node, parent_prefix=""):
    """
    Recursively generates absolute paths for nodes and resolves local connections.
    """
    # Define this node's absolute ID
    absolute_id = f"{parent_prefix}.{node.get_field_name()}" if parent_prefix else node.get_field_name()
    
    # Resolve the input/output connection strings
    # We rebuild the interfaces so they point to the absolute paths
    new_inputs = EnergyInterface(
        mechanical=[f"{parent_prefix}.{port}" for port in node.inputs.mechanical],
        electrical=[f"{parent_prefix}.{port}" for port in node.inputs.electrical],
        flow=[f"{parent_prefix}.{port}" for port in node.inputs.flow],
        force=[f"{parent_prefix}.{port}" for port in node.inputs.force]
    )
    
    new_outputs = EnergyInterface(
        mechanical=[f"{parent_prefix}.{port}" for port in node.outputs.mechanical],
        electrical=[f"{parent_prefix}.{port}" for port in node.outputs.electrical],
        flow=[f"{parent_prefix}.{port}" for port in node.outputs.flow],
        force=[f"{parent_prefix}.{port}" for port in node.outputs.force]
    )
    
    # Update the node itself (using eqx.tree_at since it's immutable)
    node = eqx.tree_at(lambda n: n.tag, node, absolute_id)
    node = eqx.tree_at(lambda n: n.inputs, node, new_inputs)
    node = eqx.tree_at(lambda n: n.outputs, node, new_outputs)
    
    # Recurse through any subcomponents
    if hasattr(node, 'subcomponents') and node.subcomponents:
        resolved_children = tuple(
            _resolve_namespaces(child, parent_prefix=absolute_id) 
            for child in node.subcomponents
        )
        node = eqx.tree_at(lambda n: n.subcomponents, node, resolved_children)
        
    return node

def initialize_energy(state: "rcf.State", system: "rcf.Aircraft", settings: "rcf.Settings"):
    
    flat_state_nodes = {}
    resolved_lines = []

    conditions_map = {
        FuelTank: FuelTankConditions
    }
    
    for l_idx, line in enumerate(system.energy.lines):
        
        # Resolve the namespace for this entire line and all its nested children
        resolved_line = _resolve_namespaces(line, parent_prefix=f"line_{l_idx}")
        resolved_lines.append(resolved_line)
        
        # Helper function to extract all nodes into our flat dict
        def _extract_to_flat_state(n):
            flat_state_nodes[n.tag] = conditions_map[n.__class__](tag=n.tag) # Initialize the state
            if hasattr(n, 'subcomponents'):
                for child in n.subcomponents:
                    _extract_to_flat_state(child)
                    
        _extract_to_flat_state(resolved_line)

    # Update the System with the newly resolved absolute-path nodes
    updated_network = eqx.tree_at(lambda e: e.lines, system.energy, tuple(resolved_lines)).sort_network_topology()
    system = eqx.tree_at(lambda s: s.energy, system, updated_network)
    
    # Update the State with the flat dictionary
    state = eqx.tree_at(lambda s: s.energy.nodes, state, flat_state_nodes)
    
    return state, system, settings
