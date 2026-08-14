# Trace/Framework/Missions/Conditions/Energy.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Aug 2024, Trace Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

# package imports
import jax.numpy as jnp

# Trace imports
from eden_trace.utils import empty_array, init_field

from eden_trace.library.gases import IdealGas, Air
from eden_trace.framework.conditions import Condition

# ----------------------------------------------------------------------------------------------------------------------
#  Energy Interfaces
# ----------------------------------------------------------------------------------------------------------------------

class MechanicalOutputs(Condition):
    tag = "Mechanical Outputs"

    work: jnp.ndarray = empty_array()
    power: jnp.ndarray = empty_array()


class ElectricalOutputs(Condition):
    tag = "Electrical Outputs"

    power: jnp.ndarray = empty_array()
    voltage: jnp.ndarray = empty_array()
    current: jnp.ndarray = empty_array()


class FuelOutputs(Condition):
    tag = "Fuel Outputs"

    TSFC: jnp.ndarray = empty_array()
    flow_rate: jnp.ndarray = empty_array()


class FlowOutputs(Condition):
    tag = "Flow Outputs"
    fluid: IdealGas = init_field(Air)

    speed: jnp.ndarray = empty_array()
    speed_of_sound: jnp.ndarray = empty_array()
    mach_number: jnp.ndarray = empty_array()
    reynolds_number: jnp.ndarray = empty_array()
    
    pressure: jnp.ndarray = empty_array()
    temperature: jnp.ndarray = empty_array()
    enthalpy: jnp.ndarray = empty_array()

    stagnation_pressure: jnp.ndarray = empty_array()
    stagnation_temperature: jnp.ndarray = empty_array()
    stagnation_enthalpy: jnp.ndarray = empty_array()

    area: jnp.ndarray = empty_array()
    density: jnp.ndarray = empty_array()
    mass_flow_rate: jnp.ndarray = empty_array()
    fuel_air_ratio: jnp.ndarray = empty_array()

    dynamic_viscosity: jnp.ndarray = empty_array()
    dynamic_pressure: jnp.ndarray = empty_array()

    gamma: jnp.ndarray = empty_array()
    Cp: jnp.ndarray = empty_array()
    R: jnp.ndarray = empty_array()

class ResidualOutputs(Condition):
    tag = "Residual Outputs"

    mass: jnp.ndarray = empty_array()
    mass_flow_rate: jnp.ndarray = empty_array()
    
    work: jnp.ndarray = empty_array()
    power: jnp.ndarray = empty_array()

    thrust: jnp.ndarray = empty_array()
    area: jnp.ndarray = empty_array()

    # Single Spool Turbojet Residuals
    compressor_Wc: jnp.ndarray = empty_array()
    turbine_Wp: jnp.ndarray = empty_array()

    # Dual Spool Turbofan Residuals
    fan_Wc: jnp.ndarray = empty_array()
    lpc_Wc: jnp.ndarray = empty_array()
    hpc_Wc: jnp.ndarray = empty_array()

    lpt_Wp: jnp.ndarray = empty_array()
    hpt_Wp: jnp.ndarray = empty_array()

class ForceOutputs(Condition):
    tag = "Force Outputs"

    thrust: jnp.ndarray = empty_array()
    nondimensional_thrust: jnp.ndarray = empty_array()
    specific_impulse: jnp.ndarray = empty_array()


class OutputConditions(Condition):
    tag = "Node Outputs"

    mechanical: MechanicalOutputs = init_field(MechanicalOutputs)
    electrical: ElectricalOutputs = init_field(ElectricalOutputs)
    fuel: FuelOutputs = init_field(FuelOutputs)
    flow: FlowOutputs = init_field(FlowOutputs)
    force: ForceOutputs = init_field(ForceOutputs)
    residual: ResidualOutputs = init_field(ResidualOutputs)


class NodeConditions(Condition):
    # Attribute         Type                Default Value
    tag: str = init_field("Energy Node Conditions", static=True)

    outputs: OutputConditions = init_field(OutputConditions)

    mass: jnp.ndarray = empty_array()

# ----------------------------------------------------------------------------------------------------------------------
#  Energy Stores
# ----------------------------------------------------------------------------------------------------------------------

class BatteryCellConditions(NodeConditions):
    # Attribute                 Type        Default Value
    tag: str = init_field("Battery Cell", static=True)

    cycle_in_day: int = init_field(0, static=True)
    resistance_growth_factor: float = init_field(0.0, static=True)
    capacity_fade_factor: float = init_field(0.0, static=True)

    temperature: jnp.ndarray = empty_array()
    charge_throughput: jnp.ndarray = empty_array()
    state_of_charge: jnp.ndarray = empty_array()


class BatteryPackConditions(NodeConditions):
    # Attribute             Type                    Default Value
    tag: str = init_field("Battery Pack", static=True)

    maximum_total_energy: float = init_field(0.0, static=True)

    cell: BatteryCellConditions = init_field(BatteryCellConditions)

    temperature: jnp.ndarray = empty_array()


# ----------------------------------------------------------------------------------------------------------------------
#  Energy Networks
# ----------------------------------------------------------------------------------------------------------------------


class EnergyNetworkConditions(NodeConditions):
    tag: str = init_field("Energy Network", static=True)

    nodes: dict = init_field(dict)

    total_energy: jnp.ndarray = empty_array()
    total_efficiency: jnp.ndarray = empty_array()

    throttle: jnp.ndarray = empty_array()
    total_power: jnp.ndarray = empty_array()

    total_force_vector: jnp.ndarray = empty_array((0, 3))
    total_moment_vector: jnp.ndarray = empty_array((0, 3))


class TurbojetNetworkConditions(EnergyNetworkConditions):
    
    tag: str = init_field("Turbojet Network", static=True)

    # Control hooks
    fuel_air_ratio: jnp.ndarray = empty_array()
    mass_flow_rate: jnp.ndarray = empty_array()
    rotation_speed: jnp.ndarray = empty_array()
    compressor_Rline: jnp.ndarray = empty_array()
    turbine_PR: jnp.ndarray = empty_array()

    target_thrust: jnp.ndarray = empty_array()
    target_temperature: jnp.ndarray = empty_array()

class TurbofanNetworkConditions(EnergyNetworkConditions):

    tag: str = init_field("Turbofan Network", static=True)

    # Control hooks
    fuel_air_ratio: jnp.ndarray = empty_array()
    mass_flow_rate: jnp.ndarray = empty_array()
    
    LP_speed: jnp.ndarray = empty_array()
    HP_speed: jnp.ndarray = empty_array()

    fan_Rline: jnp.ndarray = empty_array()
    lpc_Rline: jnp.ndarray = empty_array()
    hpc_Rline: jnp.ndarray = empty_array()
    
    lpt_PR: jnp.ndarray = empty_array()
    hpt_PR: jnp.ndarray = empty_array()

    bypass_ratio: jnp.ndarray = empty_array()
    target_thrust: jnp.ndarray = empty_array()
    target_temperature: jnp.ndarray = empty_array()

