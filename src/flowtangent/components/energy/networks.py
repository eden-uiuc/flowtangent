# flowtangent/Library/Components/Energy/Network.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: Apr 2025, Flowtangent Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flowtangent.data.atmospheres import Atmosphere


from dataclasses import replace
from graphlib import CycleError, TopologicalSorter

import jax

# Flowtangent imports
from ...data import units
from ...data.atmospheres import USStandard1976
from ...utils import Module, NameType, field, static_field, update
from .lines import PACTLine
from .nodes import BleedFlow, GraphDomain, PACTNode

# ----------------------------------------------------------------------------------------------------------------------
#  Design Conditions
# ----------------------------------------------------------------------------------------------------------------------


class NetworkParameters(Module):
    altitude: float = 0.0
    mach_number: float = 0.01
    thrust: float = 1.0 * units.N
    atmosphere: Atmosphere = field(USStandard1976)


# ----------------------------------------------------------------------------------------------------------------------
#  Energy Networks
# ----------------------------------------------------------------------------------------------------------------------


def _resolve_namespaces(node, parent_prefix=""):
    """
    Recursively generates absolute paths for nodes and resolves local connections.
    """
    # Define this node's absolute ID
    absolute_id = f"{parent_prefix}.{node.field_name}" if parent_prefix else node.field_name

    def parse_input(input: str):
        if input == "freestream":
            return "freestream"
        flat_input_parts = input.replace(" ", "_").lower().split(".")
        if len(flat_input_parts) > 1:
            if flat_input_parts[0] == "self" or node.field_name in flat_input_parts:
                return absolute_id + "." + flat_input_parts[-1]
            elif flat_input_parts[0] == "parent":
                grandparent_prefix = ".".join(parent_prefix.split(".")[:-1])
                return grandparent_prefix + "." + flat_input_parts[-1]
            else:
                return parent_prefix + "." + ".".join(flat_input_parts)
        else:
            if flat_input_parts[0] == "parent":
                return parent_prefix
            else:
                return parent_prefix + "." + flat_input_parts[-1]

    # fmt: off
    new_inputs = tuple(
        i if i._assigned
        else replace(i, network_id=parse_input(i.network_id), _assigned=True)
        for i in node.inputs
    )
    # fmt: on

    # Update the node itself
    node = replace(
        node,
        network_id=absolute_id,
        inputs=new_inputs,
    )

    # Special handling for bleed flows
    if isinstance(node, BleedFlow):
        node = replace(
            node,
            parent_ID=parent_prefix,
            grandparent_ID=parent_prefix + ".mixer",
        )

    # Recurse through any subcomponents
    if hasattr(node, "subcomponents") and node.subcomponents:
        resolved_children = tuple(_resolve_namespaces(child, parent_prefix=absolute_id) for child in node.subcomponents)
        node = update(node, "subcomponents", resolved_children)

    return node


class PACTNetwork[DesignType: NetworkParameters](PACTNode):
    name: NameType = static_field("Network")
    network_id: str = static_field("network")

    nodes: dict[str, "PACTNode"] = field(dict)
    domains: tuple[GraphDomain, ...] = static_field(tuple)
    design_parameters: DesignType = field(NetworkParameters)  # type: ignore

    _bookkeeping: dict = static_field(lambda: {"lines": PACTLine})
    _execution_order: tuple[str, ...] = static_field(tuple)

    def _rebalance_flow_splitters(self) -> PACTNetwork:
        """Rebalances fractions directly within the subcomponents tree."""
        # Grab a temporary flat dict just to look at the hierarchy
        temp_dict = {}

        def _temp_recurse(subs):
            for c in subs:
                if isinstance(c, PACTNode):
                    temp_dict[c.network_id] = c
                if hasattr(c, "subcomponents") and c.subcomponents:
                    _temp_recurse(c.subcomponents)

        _temp_recurse(self.subcomponents)

        source_to_splitters = {}
        for node in temp_dict.values():
            if hasattr(node, "extraction_fraction") and node.inputs:
                upstream_source = node.inputs[0]
                source_to_splitters.setdefault(upstream_source, []).append(node)

        corrected_fractions = {}
        for source, splitters in source_to_splitters.items():
            total = sum(s.extraction_fraction for s in splitters)
            if abs(total - 1.0) > 1e-6 and total > 0:
                for s in splitters:
                    corrected_fractions[s.network_id] = s.extraction_fraction / total

        if not corrected_fractions:
            return self

        def _apply(node):
            if isinstance(node, PACTNode) and node.network_id in corrected_fractions:
                return update(node, "extraction_fraction", corrected_fractions[node.network_id])
            return node

        return jax.tree_util.tree_map(_apply, self, is_leaf=lambda x: isinstance(x, PACTNode))

    def compute_topology(self):

        updated_network = replace(self, network_id=self.field_name)
        resolved_lines = []

        for line in updated_network.lines:
            # Resolve the namespace for this entire line and all its nested children
            resolved_line = _resolve_namespaces(line, parent_prefix=f"{updated_network.field_name}")
            resolved_lines.append(resolved_line)

        updated_network = update(updated_network, "subcomponents", tuple(resolved_lines))._update_node_topology()

        return updated_network

    def _get_all_nodes(self) -> PACTNetwork:
        nodes_dict = {}

        def _recurse(subcomponents):
            for comp in subcomponents:
                if isinstance(comp, PACTNode):
                    nodes_dict[comp.network_id] = comp
                if hasattr(comp, "subcomponents") and comp.subcomponents:
                    _recurse(comp.subcomponents)

        _recurse(self.subcomponents)
        return update(self, "nodes", nodes_dict)

    def _update_node_topology(self) -> PACTNetwork:
        """The single entry point to finalize the network for execution.
        Use after running initialize_energy so parts are properly ID'd."""

        balanced_network = self._rebalance_flow_splitters()
        updated_network = balanced_network._get_all_nodes()
        dependency_graph = {ID: set([i.network_id for i in node.inputs]) for ID, node in updated_network.nodes.items()}

        try:
            sorter = TopologicalSorter(dependency_graph)
            updated_order = tuple(sorter.static_order())

            return replace(updated_network, _execution_order=updated_order)
        except CycleError as e:
            raise ValueError(f"Cyclic dependency detected: {e}")

    def sync_and_clear_nodes(self) -> PACTNetwork:
        """
        Projects the updated nodes from the flat 'nodes' dictionary
        back onto their original positions in the nested subcomponents tree.
        Clears nodes dict and execution order.
        """

        def _walk_and_sync(component):
            # If we hit an EnergyNode, replace it with the latest version from the dict
            if isinstance(component, PACTNode):
                # Grab the updated node (fallback to current if not in dict)
                component = self.nodes.get(component.network_id, component)

            # Recurse down through any nested wrappers (like EnergyLines or Engine Pods)
            if hasattr(component, "subcomponents") and component.subcomponents:
                synced_children = tuple(_walk_and_sync(child) for child in component.subcomponents)
                # Functionally update the component's subcomponents
                component = update(component, "subcomponents", synced_children)

            return component

        # Start the recursive sync from the top-level subcomponents
        synced_subcomponents = tuple(_walk_and_sync(child) for child in self.subcomponents)

        # Return a new network
        return replace(
            self,
            subcomponents=synced_subcomponents,
            nodes={},
            _execution_order=(),
        )
