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
from RCAIDE.Library.Components.Energy.Networks import EnergyNetwork

# ----------------------------------------------------------------------------------------------------------------------
#  Graph Energy Network Analysis
# ----------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------
# 2. STATEFUL FRAMEWORK ROUTER
# ---------------------------------------------------------
def build_analysis_from_network(network: EnergyNetwork):
    
    return Process(
                tag=f"{network.tag} Analysis",
                steps=tuple(
                    ProcessStep(
                        tag=f"tag Transmission",
                        function=network.nodes[tag].transmit
                    ) for tag in network._execution_order
                )
            )
