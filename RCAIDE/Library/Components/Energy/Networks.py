# RCAIDE/Library/Components/Energy/Network.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: Apr 2025, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from __future__ import annotations
from typing import TYPE_CHECKING
from graphlib import TopologicalSorter, CycleError

# package imports
import jax
import equinox as eqx

# RCAIDE imports
from RCAIDE.utils import init_field

from .Nodes import EnergyNode, EnergySplitter
from RCAIDE.Library import Component
from RCAIDE.Library.Components.Energy.Propulsors import Propulsor
from RCAIDE.Library.Components.Energy.Nodes import FlowNode, EnergySplitter
from RCAIDE.Library.Components.Energy.Stores import EnergyStore

if TYPE_CHECKING:
    from RCAIDE.Framework import State, System, Settings
# ----------------------------------------------------------------------------------------------------------------------
#  Network
# ----------------------------------------------------------------------------------------------------------------------


class EnergyLine(EnergyNode):

    @property
    def propulsors(self) -> tuple:
        return tuple(c for c in self.subcomponents if isinstance(c, Propulsor))

    @property
    def splitters(self) -> tuple:
        return tuple(c for c in self.subcomponents if isinstance(c, EnergySplitter))
    
    @property
    def stores(self) -> tuple:
        return tuple(c for c in self.subcomponents if isinstance(c, EnergyStore))


class EnergyNetwork(EnergyNode):

    tag: str = init_field('Energy Network', static=True)

    @property
    def lines(self) -> tuple:
        return tuple(c for c in self.subcomponents if isinstance(c, EnergyLine))
    
    nodes: dict[str, "EnergyNode"] = init_field(dict)
    _execution_order: tuple[str, ...] = init_field(tuple, static=True)
    
    def _rebalance_flow_splitters(self) -> "EnergyNetwork":
        """Rebalances fractions directly within the subcomponents tree."""
        # Grab a temporary flat dict just to look at the hierarchy
        temp_dict = {}
        def _temp_recurse(subs):
            for c in subs:
                if isinstance(c, EnergyNode): temp_dict[c.tag] = c
                if hasattr(c, "subcomponents") and c.subcomponents: _temp_recurse(c.subcomponents)
        _temp_recurse(self.subcomponents)

        source_to_splitters = {}
        for tag, node in temp_dict.items():
            if hasattr(node, 'extraction_fraction') and node.inputs:
                upstream_source = node.inputs[0]
                source_to_splitters.setdefault(upstream_source, []).append(node)
                
        corrected_fractions = {}
        for source, splitters in source_to_splitters.items():
            total = sum(s.extraction_fraction for s in splitters)
            if abs(total - 1.0) > 1e-6 and total > 0:
                for s in splitters:
                    corrected_fractions[s.tag] = s.extraction_fraction / total

        if not corrected_fractions:
            return self

        def _apply(node):
            if isinstance(node, EnergyNode) and node.tag in corrected_fractions:
                return eqx.tree_at(lambda n: n.extraction_fraction, node, corrected_fractions[node.tag])
            return node

        return jax.tree_util.tree_map(_apply, self, is_leaf=lambda x: isinstance(x, EnergyNode))
    
    def _get_all_nodes(self) -> "EnergyNetwork":
        nodes_dict = {}
        def _recurse(subcomponents):
            for comp in subcomponents:
                if isinstance(comp, EnergyNode):
                    nodes_dict[comp.tag] = comp
                if hasattr(comp, "subcomponents") and comp.subcomponents:
                    _recurse(comp.subcomponents)
        
        _recurse(self.subcomponents)
        return eqx.tree_at(lambda n: n.nodes, self, nodes_dict)
    
    def sort_network_topology(self) -> "EnergyNetwork":
        """The single entry point to finalize the network for execution."""
        
        balanced_network = self._rebalance_flow_splitters()
        updated_network = balanced_network._get_all_nodes()
        dependency_graph = {
            tag: set(node.all_causal_inputs) 
            for tag, node in updated_network.nodes.items()
        }

        try:
            sorter = TopologicalSorter(dependency_graph)
            
            return eqx.tree_at(
                lambda n: n._execution_order, 
                updated_network, 
                tuple(sorter.static_order())
            )
        except CycleError as e:
            raise ValueError(f"Cyclic dependency detected: {e}")

    def transmit(self, state: "State", system: "System", settings: "Settings"):
        """The XLA-compiled mission execution loop."""
        
        for tag in self._execution_order:
            
            state, system, settings = self.nodes[tag].transmit(state, system, settings) 

        return state, system, settings

