# RCAIDE/Framework/Analyses/Energy/GraphNetwork.py
# (c) Copyright 2026 Aerospace Research Community LLC
#
# Created: Jun 2026, J. Smart
# Modified: Jun 2026, J. Smart

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from typing import TYPE_CHECKING
import jax
import jax.numpy as jnp
import equinox as eqx

# --- Framework Imports (Strictly for Type Hinting to avoid Circular Imports) ---
if TYPE_CHECKING:
    from RCAIDE.Framework.State import State
    from RCAIDE.Framework.Systems import System
    from RCAIDE.Framework.Settings import Settings

from RCAIDE.utils import inputs, outputs
from RCAIDE.Framework import Process, ProcessStep
from RCAIDE.Framework.Missions.Initialize import initialize_energy
from RCAIDE.Library.Components.Energy.Networks import EnergyNetwork

# ----------------------------------------------------------------------------------------------------------------------
#  Graph Energy Network Analysis
# ----------------------------------------------------------------------------------------------------------------------

def build_analysis_from_network(network: EnergyNetwork):

    analysis_network = network.assign_network_IDs()

    # def make_node_function(node_ID: str):
    #     def _pure_transmit(state, system, settings):
    #         return analysis_network.nodes[node_ID].transmit(state, system, settings)
    #     return _pure_transmit
    
    def make_node_function(node_ID: str):
        node_func = analysis_network.nodes[node_ID].__class__.transmit
        node_inputs = getattr(node_func, '_inputs', set())
        node_outputs = getattr(node_func, '_outputs', set())
        
        @inputs(*node_inputs)
        @outputs(*node_outputs)
        def _pure_transmit(state, system, settings):
            return analysis_network.nodes[node_ID].transmit(state, system, settings)
        
        return _pure_transmit

    network_analysis =  Process(
        tag=f"{network.tag} Analysis",
        steps=tuple(
            ProcessStep(
                tag=f"{ID}",
                function=make_node_function(ID)
            ) for ID in analysis_network._execution_order
        )
    )

    return network_analysis
