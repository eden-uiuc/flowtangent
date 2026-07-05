# Trace/Library/Components/Energy/Network.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: Apr 2025, Trace Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.eden_trace.framework import State, System, Settings
    from src.eden_trace.framework.conditions.controls import Control, Residual
    from src.eden_trace.library.atmospheres import Atmosphere

from dataclasses import replace
from graphlib import CycleError, TopologicalSorter

import jax
import jax.numpy as jnp
import equinox as eqx


# Trace imports
from src.eden_trace.utils import init_field, register

from src.eden_trace.library import units
from src.eden_trace.library.components.energy.nodes import EnergyDomain, EnergyInput, EnergyNode
from src.eden_trace.library.atmospheres import USStandard1976

from .lines import EnergyLine, TurbojetEnergyLine

# ----------------------------------------------------------------------------------------------------------------------
#  Design Conditions
# ----------------------------------------------------------------------------------------------------------------------

@register
class NetworkDesign(eqx.Module):

    altitude: float = 0.0
    mach_number: float = 0.01
    thrust: float = 1.0 * units.N
    atmosphere_model: Atmosphere = init_field(USStandard1976)

# ----------------------------------------------------------------------------------------------------------------------
#  Energy Networks
# ----------------------------------------------------------------------------------------------------------------------


def _resolve_namespaces(node, parent_prefix=""):
    """
    Recursively generates absolute paths for nodes and resolves local connections.
    """
    # Define this node's absolute ID
    absolute_id = f"{parent_prefix}.{node.get_field_name()}" if parent_prefix else node.get_field_name()

    def parse_input(input: str):
        flat_input_parts = input.replace(" ", "_").lower().split(".")
        if flat_input_parts[0] == "self" or node.get_field_name() in flat_input_parts:
            return absolute_id + "." + flat_input_parts[-1]
        else:
            return parent_prefix + "." + flat_input_parts[-1]

    new_inputs = tuple(EnergyInput(i.domain, parse_input(i.network_ID)) for i in node.inputs)

    # Update the node itself
    node = replace(
        node,
        network_ID=absolute_id,
        inputs=new_inputs,
    )

    # Recurse through any subcomponents
    if hasattr(node, "subcomponents") and node.subcomponents:
        resolved_children = tuple(_resolve_namespaces(child, parent_prefix=absolute_id) for child in node.subcomponents)
        node = eqx.tree_at(lambda n: n.subcomponents, node, resolved_children)

    return node


@register
class EnergyNetwork(EnergyNode):
    
    tag: str = init_field("Energy Network", static=True)
    nodes: dict[str, "EnergyNode"] = init_field(dict)
    domains: tuple[EnergyDomain, ...] = init_field(tuple, static=True)
    design_parameters: NetworkDesign = init_field(NetworkDesign)

    _bookkeeping: dict = init_field(lambda: {"lines": EnergyLine}, static=True)
    _execution_order: tuple[str, ...] = init_field(tuple, static=True)

    controls: tuple[Control, ...] = init_field(tuple, static=True)
    residuals: tuple[Residual, ...] = init_field(tuple, static=True)

    def _rebalance_flow_splitters(self) -> "EnergyNetwork":
        """Rebalances fractions directly within the subcomponents tree."""
        # Grab a temporary flat dict just to look at the hierarchy
        temp_dict = {}

        def _temp_recurse(subs):
            for c in subs:
                if isinstance(c, EnergyNode):
                    temp_dict[c.network_ID] = c
                if hasattr(c, "subcomponents") and c.subcomponents:
                    _temp_recurse(c.subcomponents)

        _temp_recurse(self.subcomponents)

        source_to_splitters = {}
        for ID, node in temp_dict.items():
            if hasattr(node, "extraction_fraction") and node.inputs:
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
            lambda e: e.subcomponents, updated_network, tuple(resolved_lines)
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
        dependency_graph = {ID: set([i.network_ID for i in node.inputs]) for ID, node in updated_network.nodes.items()}

        try:
            sorter = TopologicalSorter(dependency_graph)
            updated_order = tuple(sorter.static_order())

            return replace(updated_network, _execution_order=updated_order)
        except CycleError as e:
            raise ValueError(f"Cyclic dependency detected: {e}")
    
    def sync_and_clear_nodes(self) -> EnergyNetwork:
        """
        Projects the updated nodes from the flat 'nodes' dictionary
        back onto their original positions in the nested subcomponents tree.
        Clears nodes dict and execution order.
        """
        def _walk_and_sync(component):
            # If we hit an EnergyNode, replace it with the latest version from the dict
            if isinstance(component, EnergyNode):
                # Grab the updated node (fallback to current if not in dict)
                component = self.nodes.get(component.network_ID, component)

            # Recurse down through any nested wrappers (like EnergyLines or Engine Pods)
            if hasattr(component, "subcomponents") and component.subcomponents:
                synced_children = tuple(_walk_and_sync(child) for child in component.subcomponents)
                # Functionally update the component's subcomponents
                component = eqx.tree_at(lambda c: c.subcomponents, component, synced_children)

            return component

        # Start the recursive sync from the top-level subcomponents
        synced_subcomponents = tuple(_walk_and_sync(child) for child in self.subcomponents)

        # Return a new network
        return replace(
            self, 
            subcomponents=synced_subcomponents, 
            nodes={}, 
            _execution_order=()
        )


# ----------------------------------------------------------------------------------------------------------------------
#  Turbojet Energy Network
# ----------------------------------------------------------------------------------------------------------------------

def _TurbojetNetworkSetup():
    return (TurbojetEnergyLine(tag="Line"),)

@register
class TurbojetDesign(NetworkDesign):

    initial_MFR: float = 100.0 * units.lbm / units.s
    initial_turb_PR: float = 5.0

@register
class TurbojetEnergyNetwork(EnergyNetwork):
    tag: str = init_field("Network", static=True)
    network_ID: str = init_field("network", static=True)

    subcomponents: tuple = init_field(_TurbojetNetworkSetup)
    design_parameters: TurbojetDesign = init_field(TurbojetDesign)

    inputs: tuple = init_field(
        (
            EnergyInput("force", "network.line"),
            EnergyInput("residual", "network.line"),
        )
    )

    def transmit(self, state: State, system: System, settings: Settings):

        updated_state = state

        # Total Thrust----------------------------------------------------------

        total_thrust = jnp.atleast_2d(self.sum_domain_inputs(state, "force", "thrust"))

        total_force_vector = jnp.hstack(
            (total_thrust, jnp.zeros((total_thrust.shape[0], 2)))
        )
        if settings.analysis.energy.design_mode:
            target_thrust = self.design_parameters.thrust
        else:
            target_thrust = state.energy.target_thrust
        
        updated_state = eqx.tree_at(
            lambda s: (
                s.energy.total_force_vector,
                s.energy.outputs.residual.thrust,
            ),
            updated_state,(
                total_force_vector,
                (total_thrust - target_thrust)/target_thrust,
            )
        )


        # Mass & Work Imbalance -------------------------------------------------------

        total_d_work = self.sum_domain_inputs(updated_state, "residual", "work")
        # total_d_mass = self.sum_domain_inputs(updated_state, "residual", "mass_flow_rate")

        updated_state = eqx.tree_at(
            lambda s: (
                s.energy.outputs.residual.work,
                # s.energy.outputs.residual.mass_flow_rate
            ),
            updated_state,(
                total_d_work,
                # total_d_mass
            )
        )

        return updated_state, system, settings
