# ruff: noqa: N815
# flowtangent/Framework/Missions/Conditions/Energy.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Aug 2024, FlowTangent Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

# package imports

from flowtangent.core._state_data import StateData
from flowtangent.data.gases import Air, Gas

# FlowTangent imports
from ...utils import field
from ...utils.typing import TimeScalar, TimeVector3, _

# ----------------------------------------------------------------------------------------------------------------------
#  Energy Interfaces
# ----------------------------------------------------------------------------------------------------------------------


class MechanicalOutputs(StateData):
    name: str = field("Mechanical Outputs", static=True)

    work: TimeScalar = _
    power: TimeScalar = _


class ElectricalOutputs(StateData):
    name: str = field("Electrical Outputs", static=True)

    power: TimeScalar = _
    voltage: TimeScalar = _
    current: TimeScalar = _


class FuelOutputs(StateData):
    name: str = field("Fuel Outputs", static=True)

    TSFC: TimeScalar = _
    flow_rate: TimeScalar = _


class FlowOutputs(StateData):
    name: str = field("Flow Outputs", static=True)
    fluid: Gas = field(Air)

    speed: TimeScalar = _
    speed_of_sound: TimeScalar = _
    mach_number: TimeScalar = _
    reynolds_number: TimeScalar = _

    pressure: TimeScalar = _
    temperature: TimeScalar = _
    enthalpy: TimeScalar = _

    stagnation_pressure: TimeScalar = _
    stagnation_temperature: TimeScalar = _
    stagnation_enthalpy: TimeScalar = _

    area: TimeScalar = _
    density: TimeScalar = _
    mass_flow_rate: TimeScalar = _
    fuel_air_ratio: TimeScalar = _

    dynamic_viscosity: TimeScalar = _
    dynamic_pressure: TimeScalar = _

    gamma: TimeScalar = _
    Cp: TimeScalar = _
    R: TimeScalar = _


class ResidualOutputs(StateData):
    name: str = field("Residual Outputs", static=True)

    mass: TimeScalar = _
    mass_flow_rate: TimeScalar = _

    work: TimeScalar = _
    power: TimeScalar = _

    thrust: TimeScalar = _
    area: TimeScalar = _

    # Single Spool Turbojet Residuals
    compressor_Wc: TimeScalar = _
    turbine_Wp: TimeScalar = _

    # Dual Spool Turbofan Residuals
    fan_Wc: TimeScalar = _
    lpc_Wc: TimeScalar = _
    hpc_Wc: TimeScalar = _

    lpt_Wp: TimeScalar = _
    hpt_Wp: TimeScalar = _


class ForceOutputs(StateData):
    name: str = field("Force Outputs", static=True)

    thrust: TimeScalar = _
    nondimensional_thrust: TimeScalar = _
    specific_impulse: TimeScalar = _


class NodeState(StateData):
    name: str = field("Node Outputs", static=True)

    mechanical: MechanicalOutputs = field(MechanicalOutputs)
    electrical: ElectricalOutputs = field(ElectricalOutputs)
    fuel: FuelOutputs = field(FuelOutputs)
    flow: FlowOutputs = field(FlowOutputs)
    force: ForceOutputs = field(ForceOutputs)
    residual: ResidualOutputs = field(ResidualOutputs)

    mass: TimeScalar = _


# ----------------------------------------------------------------------------------------------------------------------
#  Energy Stores
# ----------------------------------------------------------------------------------------------------------------------


class BatteryCellConditions(NodeState):
    # Attribute                 Type        Default Value
    name: str = field("Battery Cell", static=True)

    cycle_in_day: int = field(0, static=True)
    resistance_growth_factor: float = field(0.0, static=True)
    capacity_fade_factor: float = field(0.0, static=True)

    temperature: TimeScalar = _
    charge_throughput: TimeScalar = _
    state_of_charge: TimeScalar = _


class BatteryPackConditions(NodeState):
    # Attribute             Type                    Default Value
    name: str = field("Battery Pack", static=True)

    maximum_total_energy: float = field(0.0, static=True)

    cell: BatteryCellConditions = field(BatteryCellConditions)

    temperature: TimeScalar = _


# ----------------------------------------------------------------------------------------------------------------------
#  Energy Networks
# ----------------------------------------------------------------------------------------------------------------------


class NetworkState(NodeState):
    name: str = field("Energy Network", static=True)

    nodes: dict = field(dict)

    total_energy: TimeScalar = _
    total_efficiency: TimeScalar = _

    throttle: TimeScalar = _
    total_power: TimeScalar = _

    total_force_vector: TimeVector3 = _
    total_moment_vector: TimeVector3 = _


class TurbojetState(NetworkState):
    name: str = field("Turbojet Network", static=True)

    # Variable hooks
    fuel_air_ratio: TimeScalar = _
    mass_flow_rate: TimeScalar = _
    rotation_speed: TimeScalar = _
    compressor_Rline: TimeScalar = _
    turbine_PR: TimeScalar = _

    target_thrust: TimeScalar = _
    target_temperature: TimeScalar = _


class TurbofanState(NetworkState):
    name: str = field("Turbofan Network", static=True)

    # Variable hooks
    fuel_air_ratio: TimeScalar = _
    mass_flow_rate: TimeScalar = _

    LP_speed: TimeScalar = _
    HP_speed: TimeScalar = _

    fan_Rline: TimeScalar = _
    lpc_Rline: TimeScalar = _
    hpc_Rline: TimeScalar = _

    lpt_PR: TimeScalar = _
    hpt_PR: TimeScalar = _

    bypass_ratio: TimeScalar = _
    target_thrust: TimeScalar = _
    target_temperature: TimeScalar = _
