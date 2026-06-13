# RCAIDE/Library/Components/Energy/Nodes.py
# (c) Copyright 2026 Aerospace Research Community LLC
#
# Created: Jun 2026, J. Smart
# Modified: Jun 2026, J. Smart

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from __future__ import annotations
from typing import TYPE_CHECKING
import jax
import jax.numpy as jnp
import equinox as eqx

# --- Framework Imports (Strictly for Type Hinting to avoid Circular Imports) ---
if TYPE_CHECKING:
    from RCAIDE.Framework.State import State
    from RCAIDE.Framework.System import System
    from RCAIDE.Framework.Settings import Settings

from RCAIDE.utils import init_field
from RCAIDE.Library import Component

# ----------------------------------------------------------------------------------------------------------------------
#  Energy Nodes
# ----------------------------------------------------------------------------------------------------------------------

class EnergyEfficiencies(eqx.Module):

    total: float = 1.0
    
    mechanical: float = 1.0
    electrical: float = 1.0
    fuel:       float = 1.0
    flow:       float = 1.0
    force:      float = 1.0

class EnergyNode(Component):
    
    efficiencies:   EnergyEfficiencies  = init_field(EnergyEfficiencies)

    mechanical_inputs: list[str] = init_field(list, static=True)
    electrical_inputs: list[str] = init_field(list, static=True)
    fuel_inputs:       list[str] = init_field(list, static=True)
    flow_inputs:       list[str] = init_field(list, static=True)
    force_inputs:      list[str] = init_field(list, static=True)
    
    @property
    def inputs(self):
        return (self.mechanical_inputs + self.electrical_inputs + self.fuel_inputs
                + self.self.flow_inputs + self.self.force_inputs) #type: ignore
    
    def _get_all_inputs(self, state, input_type: str, input_field: str):
        output_conditions = [getattr(state.energy.nodes[i], input_type) for i in self.inputs]
        return jnp.concatenate([getattr(out, input_field) for out in output_conditions], axis=-1)

    def sum_inputs(self, state, input_type: str, input_field: str):
        all_inputs = self._get_all_inputs(state, input_type, input_field)
        return jnp.sum(all_inputs, axis=-1)

    def average_inputs(self, state, input_type: str, input_field: str):
        all_inputs = self._get_all_inputs(state, input_type, input_field)
        return jnp.mean(all_inputs, axis=-1)

    def transmit(self, state: State, system: System, settings: Settings):
        raise NotImplementedError("No transmission method implemented. " \
        "Subclasses of EnergyNode must implement their individual transmission methods.")

class EnergySplitter(EnergyNode):

    output_fractions: tuple = init_field(tuple)

# ----------------------------------------------------------------------------------------------------------------------
#  Flow Nodes
# ----------------------------------------------------------------------------------------------------------------------

class FlowNode(EnergyNode):

    pressure_ratio:             float = 1.0
    pressure_recovery:          float = 1.0
    area_ratio:                 float = 1.0

    design_intake_temperature:  float = 298.15    # Kelvin

    rotation_speed:             float = 0.0
    noise_speed:                float = 0.0

class FlowSplitter(EnergyNode):
    
    # The fraction of the upstream mass flow this node will extract
    extraction_fraction: float = init_field(1.0)

    def transmit(self, state, system, settings):
        
        # Pull the TOTAL flow state from the upstream node
        upstream_tag = self.flow_inputs[0]
        total_inlet_flow = state.energy.nodes[upstream_tag].outputs.flow
        
        # Copy the thermodynamic state (pressure, temperature, etc.)
        # Scale only the mass flow by the extraction fraction
        extracted_flow = eqx.tree_at(
            lambda f: f.mass_flow, 
            total_inlet_flow, 
            total_inlet_flow.mass_flow * self.extraction_fraction
        )
        
        # Dump the scaled flow into this node's outputs
        state = eqx.tree_at(
            lambda s: s.energy.nodes[self.tag].outputs.flow, 
            state, 
            extracted_flow
        )
        
        return state

# ----------------------------------------------------------------------------------------------------------------------
#  Mechanical Nodes
# ----------------------------------------------------------------------------------------------------------------------

