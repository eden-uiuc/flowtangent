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

from RCAIDE.Framework import Process, ProcessStep
from RCAIDE.Framework.Missions.Initialize import initialize_energy
from RCAIDE.Library.Components.Energy.Networks import EnergyNetwork

# ----------------------------------------------------------------------------------------------------------------------
#  Graph Energy Network Analysis
# ----------------------------------------------------------------------------------------------------------------------

def build_analysis_from_network(network: EnergyNetwork):

    def make_node_function(node_ID: str):
        def _pure_transmit(state, system, settings):
            return network.nodes[node_ID].transmit(state, system, settings)
        return _pure_transmit
    
    analysis_network = network.assign_network_IDs()

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
