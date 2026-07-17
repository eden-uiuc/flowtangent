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
from eden_trace.utils import empty_array, init_field, register

from eden_trace.framework.conditions import Condition

# ----------------------------------------------------------------------------------------------------------------------
#  Energy Interfaces
# ----------------------------------------------------------------------------------------------------------------------

class MechanicalOutputs(Condition):
    tag = "Mechanical Outputs"

    work: jnp.ndarray = empty_array(0)
    
    power: jnp.ndarray = empty_array(0)


class ElectricalOutputs(Condition):
    tag = "Electrical Outputs"

    power: jnp.ndarray = empty_array(0)
    voltage: jnp.ndarray = empty_array(0)
    current: jnp.ndarray = empty_array(0)


class FuelOutputs(Condition):
    tag = "Fuel Outputs"

    fuel_air_ratio: jnp.ndarray = empty_array(0)
    TSFC: jnp.ndarray = empty_array(0)
    flow_rate: jnp.ndarray = empty_array(0)


class FlowOutputs(Condition):
    tag = "Flow Outputs"

    speed: jnp.ndarray = empty_array(0)
    speed_of_sound: jnp.ndarray = empty_array(0)
    mach_number: jnp.ndarray = empty_array(0)
    reynolds_number: jnp.ndarray = empty_array(0)
    
    pressure: jnp.ndarray = empty_array(0)
    temperature: jnp.ndarray = empty_array(0)
    enthalpy: jnp.ndarray = empty_array(0)

    stagnation_pressure: jnp.ndarray = empty_array(0)
    stagnation_temperature: jnp.ndarray = empty_array(0)
    stagnation_enthalpy: jnp.ndarray = empty_array(0)

    area: jnp.ndarray = empty_array(0)
    density: jnp.ndarray = empty_array(0)
    mass_flow_rate: jnp.ndarray = empty_array(0)

    dynamic_viscosity: jnp.ndarray = empty_array(0)
    dynamic_pressure: jnp.ndarray = empty_array(0)

    gamma: jnp.ndarray = empty_array(0)
    Cp: jnp.ndarray = empty_array(0)
    R: jnp.ndarray = empty_array(0)

class ResidualOutputs(Condition):
    tag = "Residual Outputs"

    mass: jnp.ndarray = empty_array(0)
    mass_flow_rate: jnp.ndarray = empty_array(0)
    
    work: jnp.ndarray = empty_array(0)
    power: jnp.ndarray = empty_array(0)

    thrust: jnp.ndarray = empty_array(0)
    area: jnp.ndarray = empty_array(0)

    # Fixed Nozzle Turbofan Residuals
    Wc: jnp.ndarray = empty_array(0)
    Wp: jnp.ndarray = empty_array(0)

class ForceOutputs(Condition):
    tag = "Force Outputs"

    thrust: jnp.ndarray = empty_array(0)
    nondimensional_thrust: jnp.ndarray = empty_array(0)
    specific_impulse: jnp.ndarray = empty_array(0)


class OutputConditions(Condition):
    tag = "Node Outputs"

    mechanical: MechanicalOutputs = init_field(MechanicalOutputs)
    electrical: ElectricalOutputs = init_field(ElectricalOutputs)
    fuel: FuelOutputs = init_field(FuelOutputs)
    flow: FlowOutputs = init_field(FlowOutputs)
    force: ForceOutputs = init_field(ForceOutputs)
    residual: ResidualOutputs = init_field(ResidualOutputs)


class EnergyNodeConditions(Condition):
    # Attribute         Type                Default Value
    tag: str = init_field("Energy Node Conditions", static=True)

    outputs: OutputConditions = init_field(OutputConditions)

# ----------------------------------------------------------------------------------------------------------------------
#  Energy Stores
# ----------------------------------------------------------------------------------------------------------------------

class BatteryCellConditions(EnergyNodeConditions):
    # Attribute                 Type        Default Value
    tag: str = init_field("Battery Cell", static=True)

    cycle_in_day: int = init_field(0, static=True)
    resistance_growth_factor: float = init_field(0.0, static=True)
    capacity_fade_factor: float = init_field(0.0, static=True)

    mass: jnp.ndarray = empty_array(0)
    temperature: jnp.ndarray = empty_array(0)
    charge_throughput: jnp.ndarray = empty_array(0)
    state_of_charge: jnp.ndarray = empty_array(0)


class BatteryPackConditions(EnergyNodeConditions):
    # Attribute             Type                    Default Value
    tag: str = init_field("Battery Pack", static=True)

    maximum_total_energy: float = init_field(0.0, static=True)

    cell: BatteryCellConditions = init_field(BatteryCellConditions)

    mass: jnp.ndarray = empty_array(0)
    temperature: jnp.ndarray = empty_array(0)


# ----------------------------------------------------------------------------------------------------------------------
#  Energy Networks
# ----------------------------------------------------------------------------------------------------------------------


class EnergyNetworkConditions(EnergyNodeConditions):
    tag: str = init_field("Energy Network", static=True)

    nodes: dict = init_field(dict)

    total_energy: jnp.ndarray = empty_array(0)
    total_efficiency: jnp.ndarray = empty_array(0)

    throttle: jnp.ndarray = empty_array(0)
    total_power: jnp.ndarray = empty_array(0)

    total_force_vector: jnp.ndarray = empty_array((0, 3))
    total_moment_vector: jnp.ndarray = empty_array((0, 3))


class TurbojetNetworkConditions(EnergyNetworkConditions):
    
    tag: str = init_field("Turbojet Network", static=True)

    # Control hooks
    fuel_air_ratio: jnp.ndarray = empty_array(0)
    mass_flow_rate: jnp.ndarray = empty_array(0)
    rotation_speed: jnp.ndarray = empty_array(0)
    Rline: jnp.ndarray = empty_array(0)
    turbine_PR: jnp.ndarray = empty_array(0)

    target_thrust: jnp.ndarray = empty_array(0)

class TurbofanNetworkConditions(EnergyNetworkConditions):

    tag: str = init_field("Turbofan Network", static=True)

    # Control hooks
    fuel_air_ratio: jnp.ndarray = empty_array(0)
    mass_flow_rate: jnp.ndarray = empty_array(0)
    
    LP_speed: jnp.ndarray = empty_array(0)
    HP_speed: jnp.ndarray = empty_array(0)

    LP_Rline: jnp.ndarray = empty_array(0)
    HP_Rline: jnp.ndarray = empty_array(0)
    
    LPT_PR: jnp.ndarray = empty_array(0)
    HPT_PR: jnp.ndarray = empty_array(0)

    bypass_ratio: jnp.ndarray = empty_array(0)
    target_thrust: jnp.ndarray = empty_array(0)

