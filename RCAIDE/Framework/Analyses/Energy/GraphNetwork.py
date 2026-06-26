# RCAIDE/Framework/Analyses/Energy/GraphNetwork.py
# (c) Copyright 2026 Aerospace Research Community LLC
#
# Created: Jun 2026, J. Smart
# Modified: Jun 2026, J. Smart

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

import networkx as nx

from RCAIDE.utils import init_field, inputs, outputs

from RCAIDE.Library.Components.Energy.Networks import EnergyNetwork

from RCAIDE.Framework import Process, ProcessStep

# ----------------------------------------------------------------------------------------------------------------------
#  Graph Energy Network Analysis
# ----------------------------------------------------------------------------------------------------------------------

class GraphEnergyAnalysis(Process):

    analysis_network: EnergyNetwork = init_field(EnergyNetwork)

    def graph(self, **kwargs) -> nx.DiGraph:

        G   = nx.DiGraph()
        net = self.analysis_network

        for e_idx, network_ID in enumerate(net._execution_order):
            node = net.nodes[network_ID]
            G.add_node(e_idx, name=node.tag, network_ID=node.network_ID)
            for input in node.inputs:
                input_idx = net._execution_order.index(input.network_ID)
                domain = input.domain
                G.add_edge(input_idx, e_idx, domain=domain)

        return G


def build_analysis_from_network(network: EnergyNetwork):

    analysis_network = network.assign_network_IDs()

    def make_node_function(node_ID: str):
        node_func = analysis_network.nodes[node_ID].__class__.transmit
        node_inputs = getattr(node_func, '_inputs', set())
        node_outputs = getattr(node_func, '_outputs', set())

        @inputs(*node_inputs)
        @outputs(*node_outputs)
        def _pure_transmit(state, system, settings):
            return analysis_network.nodes[node_ID].transmit(state, system, settings)

        return _pure_transmit

    network_analysis =  GraphEnergyAnalysis(
        tag=f"{network.tag} Analysis",
        steps=tuple(
            ProcessStep(
                tag=f"{ID}",
                function=make_node_function(ID)
            ) for ID in analysis_network._execution_order
        )
    )

    return network_analysis
