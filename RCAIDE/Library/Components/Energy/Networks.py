# RCAIDE/Library/Components/Energy/Network.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: Apr 2025, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from __future__ import annotations
from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from RCAIDE.Framework import State, System, Settings
    from RCAIDE.Framework.Conditions.Controls import ControlVariable, DynamicResidual

from dataclasses import replace
from graphlib import TopologicalSorter, CycleError

# package imports
import jax
import equinox as eqx

# RCAIDE imports
from RCAIDE.utils import init_field

from RCAIDE.Library.Components.Energy.Nodes import EnergyNode, EnergySplitter, EnergyStore, EnergyDomain, EnergyInput
from RCAIDE.Library.Components.Energy.Propulsors import Propulsor

# ----------------------------------------------------------------------------------------------------------------------
#  Energy Lines
# ----------------------------------------------------------------------------------------------------------------------


class EnergyLine(EnergyNode):

    _bookkeeping: dict = init_field(lambda: {
        "propulsors": Propulsor,
        "splitters": EnergySplitter,
        "stores": EnergyStore,
    }, static=True)

# ----------------------------------------------------------------------------------------------------------------------
#  Energy Networks
# ----------------------------------------------------------------------------------------------------------------------

def _resolve_namespaces(node, parent_prefix=""):
    """
    Recursively generates absolute paths for nodes and resolves local connections.
    """
    # Define this node's absolute ID
    absolute_id = f"{parent_prefix}.{node.get_field_name()}" if parent_prefix else node.get_field_name()

    def parse_input(input:str):
        flat_input_parts = input.replace(' ','_').lower().split('.')
        if flat_input_parts[0]=="self" or node.get_field_name() in flat_input_parts:
            return absolute_id+"."+flat_input_parts[-1]
        else:
            return parent_prefix+"."+flat_input_parts[-1]
    
        
    # Resolve the input/output connection strings
    # We rebuild the interfaces so they point to the absolute paths
    # new_inputs = {
    #     "mechanical":tuple(parse_input(port) for port in node.mechanical_inputs),
    #     "electrical":tuple(parse_input(port) for port in node.electrical_inputs),
    #     "flow":tuple(parse_input(port) for port in node.flow_inputs),
    #     "force":tuple(parse_input(port) for port in node.force_inputs),
    #     "fuel":tuple(parse_input(port) for port in node.fuel_inputs),
    # }

    new_inputs = tuple(EnergyInput(i.domain, parse_input(i.network_ID)) for i in node.inputs)
    
    # Update the node itself
    node = replace(
        node,
        network_ID=absolute_id,
        inputs=new_inputs
        # mechanical_inputs=new_inputs['mechanical'],
        # electrical_inputs=new_inputs['electrical'],
        # flow_inputs=new_inputs['flow'],
        # force_inputs=new_inputs['force'],
        # fuel_inputs=new_inputs['fuel'],
    )
    
    # Recurse through any subcomponents
    if hasattr(node, 'subcomponents') and node.subcomponents:
        resolved_children = tuple(
            _resolve_namespaces(child, parent_prefix=absolute_id) 
            for child in node.subcomponents
        )
        node = eqx.tree_at(lambda n: n.subcomponents, node, resolved_children)
        
    return node

class EnergyNetwork(EnergyNode):

    tag: str = init_field('Energy Network', static=True)

    domains: tuple[EnergyDomain, ...] = init_field(tuple, static=True)

    _bookkeeping: dict = init_field(lambda: {
        "lines": EnergyLine
    }, static=True)
    
    nodes: dict[str, "EnergyNode"] = init_field(dict)
    _execution_order: tuple[str, ...] = init_field(tuple, static=True)

    controls: tuple[ControlVariable, ...] = init_field(tuple, static=True)
    residuals: tuple[DynamicResidual, ...] = init_field(tuple, static=True)

    def _rebalance_flow_splitters(self) -> "EnergyNetwork":
        """Rebalances fractions directly within the subcomponents tree."""
        # Grab a temporary flat dict just to look at the hierarchy
        temp_dict = {}
        def _temp_recurse(subs):
            for c in subs:
                if isinstance(c, EnergyNode): temp_dict[c.network_ID] = c
                if hasattr(c, "subcomponents") and c.subcomponents: _temp_recurse(c.subcomponents)
        _temp_recurse(self.subcomponents)

        source_to_splitters = {}
        for ID, node in temp_dict.items():
            if hasattr(node, 'extraction_fraction') and node.inputs:
                upstream_source = node.inputs[0]
                source_to_splitters.setdefault(upstream_source, []).append(node)
                
        corrected_fractions = {}
        for source, splitters in source_to_splitters.items():
            total = sum(s.extraction_fraction for s in splitters)
            if abs(total - 1.0) > 1e-6 and total > 0:
                for s in splitters:
                    corrected_fractions[s.network_ID] = s.extraction_fraction / total

        if not corrected_fractions:
            return self

        def _apply(node):
            if isinstance(node, EnergyNode) and node.network_ID in corrected_fractions:
                return eqx.tree_at(lambda n: n.extraction_fraction, node, corrected_fractions[node.network_ID])
            return node

        return jax.tree_util.tree_map(_apply, self, is_leaf=lambda x: isinstance(x, EnergyNode))
    
    def assign_network_IDs(self):

        updated_network = replace(self, network_ID=self.get_field_name())
        resolved_lines = []

        for line in updated_network.lines:
            
            # Resolve the namespace for this entire line and all its nested children
            resolved_line = _resolve_namespaces(line, parent_prefix=f"{updated_network.get_field_name()}")
            resolved_lines.append(resolved_line)

        updated_network = eqx.tree_at(
            lambda e: e.subcomponents,
            updated_network,
            tuple(resolved_lines)
        ).sort_network_topology()
        
        return updated_network

    def _get_all_nodes(self) -> "EnergyNetwork":
        nodes_dict = {}
        def _recurse(subcomponents):
            for comp in subcomponents:
                if isinstance(comp, EnergyNode):
                    nodes_dict[comp.network_ID] = comp
                if hasattr(comp, "subcomponents") and comp.subcomponents:
                    _recurse(comp.subcomponents)
        
        _recurse(self.subcomponents)
        return eqx.tree_at(lambda n: n.nodes, self, nodes_dict)
    
    def sort_network_topology(self) -> "EnergyNetwork":
        """The single entry point to finalize the network for execution.
        Use after running initialize_energy so parts are properly ID'd."""
        
        balanced_network = self._rebalance_flow_splitters()
        updated_network = balanced_network._get_all_nodes()
        dependency_graph = {
            ID: set(node.inputs) 
            for ID, node in updated_network.nodes.items()
        }

        try:
            sorter = TopologicalSorter(dependency_graph)
            updated_order = tuple(sorter.static_order())
            
            return replace(
                updated_network, 
                _execution_order=updated_order
            )
        except CycleError as e:
            raise ValueError(f"Cyclic dependency detected: {e}")
    
    

