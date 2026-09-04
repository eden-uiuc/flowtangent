# Trace/Framework/Analyses/Energy/GraphNetwork.py
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

from eden_trace.utils import field, inputs, outputs, parse_io

from eden_trace.library.components.energy.networks import GraphNetwork

from eden_trace.framework import Process, ProcessStep

# ----------------------------------------------------------------------------------------------------------------------
#  Graph Energy Network Analysis
# ----------------------------------------------------------------------------------------------------------------------


class PACTNetwork(Process):
    analysis_network: GraphNetwork = field(GraphNetwork)

    def __init__(self, analysis_network: GraphNetwork, **kwargs):
        super().__init__(**kwargs)
        self.analysis_network = analysis_network

    def graph(self, **kwargs) -> nx.DiGraph:

        G = nx.DiGraph()
        net = self.analysis_network

        for e_idx, network_ID in enumerate(net._execution_order):
            node = net.nodes[network_ID]
            G.add_node(e_idx, name=node.tag, network_ID=node.network_ID)
            for input in node.inputs:
                input_idx = net._execution_order.index(input.network_ID)
                domain = input.domain
                G.add_edge(input_idx, e_idx, domain=domain)

        return G

def build_analysis_from_network(network: GraphNetwork):

    analysis_network = network.assign_network_IDs()

    def make_node_function(network_ID: str):
        node = analysis_network.nodes[network_ID]
        node_func = node.__class__.transmit

        raw_inputs = getattr(node_func, "_inputs", set())
        node_inputs = set()
        for io_str in raw_inputs:
            node_inputs.update(parse_io(io_str, node))

        raw_outputs = getattr(node_func, "_outputs", set())
        node_outputs = set()
        for io_str in raw_outputs:
            node_outputs.update(parse_io(io_str, node))

        @inputs(*node_inputs)
        @outputs(*node_outputs)
        def transmit(state, system, settings):
            return analysis_network.nodes[network_ID].transmit(state, system, settings)

        return transmit
    
    def make_network_function():
        net_func = analysis_network.__class__.transmit

        raw_inputs = getattr(net_func, "_inputs", set())
        node_inputs = set()
        for io_str in raw_inputs:
            node_inputs.update(parse_io(io_str, analysis_network))

        raw_outputs = getattr(net_func, "_outputs", set())
        node_outputs = set()
        for io_str in raw_outputs:
            node_outputs.update(parse_io(io_str, analysis_network))

        @inputs(*node_inputs)
        @outputs(*node_outputs)
        def net_transmit(state, system, settings):
            return analysis_network.transmit(state, system, settings)

        return net_transmit

    node_steps = tuple(
            ProcessStep(
                tag=f"{ID}",
                function=make_node_function(ID)
            ) for ID in analysis_network._execution_order
        )
    
    net_step = ProcessStep(
        tag=f"{analysis_network.network_ID}",
        function=make_network_function()
    )

    full_steps = node_steps + (net_step,)

    network_analysis = PACTNetwork(
        tag=f"{network.tag} Analysis",
        analysis_network=analysis_network,
        steps=full_steps,
    )

    return network_analysis
