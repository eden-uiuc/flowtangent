# RCAIDE/Framework/Missions/Initialization/energy.py
# (c) Copyright 2024 Aerospace Research Community LLC
# Created: Aug 2024, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from RCAIDE.Framework import State, System, Settings

from dataclasses import replace

# package imports
import equinox as eqx

# RCAIDE Imports
from RCAIDE.Library.Components.Energy.Networks import EnergyNetwork
from RCAIDE.Framework.Conditions.Energy import *


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
    new_inputs = {
        "mechanical":[f"{parent_prefix}.{port.replace(' ','_').lower()}" for port in node.mechanical_inputs],
        "electrical":[f"{parent_prefix}.{port.replace(' ','_').lower()}" for port in node.electrical_inputs],
        "flow":[f"{parent_prefix}.{port.replace(' ','_').lower()}" for port in node.flow_inputs],
        "force":[f"{parent_prefix}.{port.replace(' ','_').lower()}" for port in node.force_inputs],
        "fuel":[f"{parent_prefix}.{port.replace(' ','_').lower()}" for port in node.fuel_inputs],
    }
    
    # Update the node itself
    node = replace(
        node,
        tag=absolute_id,
        mechanical_inputs=new_inputs['mechanical'],
        electrical_inputs=new_inputs['electrical'],
        flow_inputs=new_inputs['flow'],
        force_inputs=new_inputs['force'],
        fuel_inputs=new_inputs['fuel'],
    )
    
    # Recurse through any subcomponents
    if hasattr(node, 'subcomponents') and node.subcomponents:
        resolved_children = tuple(
            _resolve_namespaces(child, parent_prefix=absolute_id) 
            for child in node.subcomponents
        )
        node = eqx.tree_at(lambda n: n.subcomponents, node, resolved_children)
        
    return node

def initialize_energy(state: State, system: System, settings: Settings):
    
    flat_state_nodes = {}
    resolved_lines = []

    conditions_map = {
        "FuelTank": FuelTankConditions
    }
    
    for l_idx, line in enumerate(system.energy_networks[0].lines):
        
        # Resolve the namespace for this entire line and all its nested children
        resolved_line = _resolve_namespaces(line, parent_prefix=f"line_{l_idx}")
        resolved_lines.append(resolved_line)
        
        # Helper function to extract all nodes into our flat dict
        def _extract_to_flat_state(n):
            if str(n.__class__.__name__) in conditions_map:
                flat_state_nodes[n.tag] = conditions_map[str(n.__class__.__name__)](tag=n.tag) # Initialize the state
            else:
                flat_state_nodes[n.tag] = EnergyNodeConditions(tag=n.tag) # Initialize the state
            if hasattr(n, 'subcomponents'):
                for child in n.subcomponents:
                    _extract_to_flat_state(child)
                    
        _extract_to_flat_state(resolved_line)

    # Update the System with the newly resolved absolute-path nodes
    updated_network = eqx.tree_at(lambda e: e.subcomponents, system.energy_networks[0], tuple(resolved_lines)).sort_network_topology()
    updated_subcomponents = tuple(c for c in system.subcomponents if not isinstance(c, EnergyNetwork)) + (updated_network,)
    updated_system = eqx.tree_at(lambda s: s.subcomponents, system, updated_subcomponents)
    
    # Update the State with the flat dictionary
    updated_state = eqx.tree_at(lambda s: s.energy.nodes, state, flat_state_nodes)
    
    return updated_state, updated_system, settings
