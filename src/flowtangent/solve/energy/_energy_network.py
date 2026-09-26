# flowtangent/Framework/Analyses/Energy/GraphNetwork.py
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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

import networkx as nx

from ... import Process, ProcessStep
from ...components.energy.networks import PACTNetwork
from ...utils import inputs, outputs, parse_io

# API

__all__ = [
    "PACTAnalysis",
]


# ----------------------------------------------------------------------------------------------------------------------
#  Graph Energy Network Analysis
# ----------------------------------------------------------------------------------------------------------------------

def make_node_function(analysis_network: PACTNetwork, network_id: str):
    node = analysis_network.nodes[network_id]
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
        return analysis_network.nodes[network_id].transmit(state, system, settings)

    return transmit

def make_network_function(analysis_network: PACTNetwork):
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


class PACTAnalysis(Process):

    def __init__(self, analysis_network: PACTNetwork, **kwargs):
        super().__init__(**kwargs)

        node_steps = tuple(
            ProcessStep(
                name=f"{ID}",
                function=make_node_function(analysis_network, ID),
            )
            for ID in analysis_network._execution_order
        )

        net_step = ProcessStep(
                name=f"{analysis_network.network_id}",
                function=make_network_function(analysis_network),
            )

        full_steps = node_steps + (net_step,)

        return super().__init__(steps=full_steps, **kwargs)

    def graph(self, **kwargs) -> nx.DiGraph:

        G = nx.DiGraph()
        net = self.analysis_network

        for e_idx, network_id in enumerate(net._execution_order):
            node = net.nodes[network_id]
            G.add_node(e_idx, name=node.name, network_id=node.network_id)
            for input in node.inputs:
                input_idx = net._execution_order.index(input.network_id)
                domain = input.domain
                G.add_edge(input_idx, e_idx, domain=domain)

        return G
