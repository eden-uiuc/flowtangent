# RCAIDE/Framework/Missions/Conditions/Energy.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Aug 2024, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

# package imports
import equinox as eqx
import jax.numpy as jnp

# RCAIDE imports
from RCAIDE.utils import empty_array, init_field
from RCAIDE.Framework.Missions.Conditions import Conditions

# ----------------------------------------------------------------------------------------------------------------------
#  Energy Interfaces
# ----------------------------------------------------------------------------------------------------------------------


class MechanicalOutputs(Conditions):
    tag = "Mechanical Outputs"

    work: jnp.ndarray  = empty_array(0)
    power: jnp.ndarray  = empty_array(0)

class ElectricalOutputs(Conditions):
    tag = "Electrical Outputs"

    power: jnp.ndarray  = empty_array(0)
    voltage: jnp.ndarray  = empty_array(0)
    current: jnp.ndarray  = empty_array(0)

class FuelOutputs(Conditions):
    tag = "Fuel Outputs"

    fuel_air_ratio:         jnp.ndarray = empty_array(0)
    TSFC:                   jnp.ndarray = empty_array(0)
    flow_rate:              jnp.ndarray = empty_array(0)

class FlowOutputs(Conditions):
    tag = "Flow Outputs"

    speed:                  jnp.ndarray  = empty_array(0)
    speed_of_sound:         jnp.ndarray  = empty_array(0)
    area_ratio:             jnp.ndarray  = empty_array(0)
    
    pressure:               jnp.ndarray  = empty_array(0)
    temperature:            jnp.ndarray  = empty_array(0)
    density:                jnp.ndarray  = empty_array(0)
    enthalpy:               jnp.ndarray  = empty_array(0)

    mass_flow_rate:         jnp.ndarray = empty_array(0)

    dynamic_viscosity:      jnp.ndarray  = empty_array(0)
    dynamic_pressure:       jnp.ndarray  = empty_array(0)

    stagnation_pressure:    jnp.ndarray  = empty_array(0)
    stagnation_temperature: jnp.ndarray  = empty_array(0)
    stagnation_enthalpy:    jnp.ndarray  = empty_array(0)

    mach_number:            jnp.ndarray  = empty_array(0)
    reynolds_number:        jnp.ndarray  = empty_array(0)

    gamma:                  jnp.ndarray  = empty_array(0)
    Cp:                     jnp.ndarray  = empty_array(0)
    R:                      jnp.ndarray  = empty_array(0)

class ForceOutputs(Conditions):
    tag = "Force Interface Conditions"

    thrust: jnp.ndarray = empty_array(0)
    nondimensional_thrust: jnp.ndarray = empty_array(0)
    specific_impulse: jnp.ndarray = empty_array(0)

class OutputConditions(Conditions):

    tag = "Energy Interface Conditons"
    
    mechanical : MechanicalOutputs  = init_field(MechanicalOutputs)
    electrical : ElectricalOutputs  = init_field(ElectricalOutputs)
    fuel       : FuelOutputs        = init_field(FuelOutputs)
    flow       : FlowOutputs        = init_field(FlowOutputs)
    force      : ForceOutputs       = init_field(ForceOutputs)

class EnergyNodeConditions(Conditions):

    # Attribute         Type         Default Value
    tag:                str          = init_field('Energy Node Conditions', static=True)

    outputs:            OutputConditions  = init_field(lambda: OutputConditions)
    throttle:           jnp.ndarray = empty_array(0)

    def __post_init__(self):
        object.__setattr__(self.outputs, "tag", f"{self.tag} Outputs")


# ----------------------------------------------------------------------------------------------------------------------
#  Battery Stores
# ----------------------------------------------------------------------------------------------------------------------
class EnergyStoreConditions(EnergyNodeConditions):

    # Attribute             Type        Default Value
    tag:                    str         = init_field('Energy Store', static=True)

    total_energy:           jnp.ndarray = empty_array(0)
    total_change_rate:      jnp.ndarray = empty_array(0)


class BatteryCellConditions(EnergyStoreConditions):

    # Attribute                 Type        Default Value
    tag:                        str         = init_field('Battery Cell', static=True)

    cycle_in_day:               int         = init_field(0, static=True)
    resistance_growth_factor:   float       = init_field(0.0, static=True)
    capacity_fade_factor:       float       = init_field(0.0, static=True)

    mass:                       jnp.ndarray  = empty_array(0)
    temperature:                jnp.ndarray  = empty_array(0)
    charge_throughput:          jnp.ndarray  = empty_array(0)
    state_of_charge:            jnp.ndarray  = empty_array(0)


class BatteryPackConditions(EnergyStoreConditions):

    # Attribute             Type                    Default Value
    tag:                    str                     = init_field('Battery Pack', static=True)

    maximum_total_energy:   float                   = init_field(0.0, static=True)

    cell:                   BatteryCellConditions   = init_field(BatteryCellConditions)

    mass:                   jnp.ndarray              = empty_array(0)
    temperature:            jnp.ndarray              = empty_array(0)


# ----------------------------------------------------------------------------------------------------------------------
#  Fuel Stores
# ----------------------------------------------------------------------------------------------------------------------

class FuelTankConditions(EnergyStoreConditions):

    # Attribute         Type        Default Value
    tag:                str         = init_field('Fuel', static=True)

    mass:               jnp.ndarray  = empty_array(0)


class EnergyLineConditions(Conditions):

    converters: Conditions = init_field(lambda: Conditions(tag='Energy Line Converters'))
    propulsors: Conditions = init_field(lambda: Conditions(tag='Energy Line Converters'))
    stores:     Conditions = init_field(lambda: Conditions(tag='Energy Line Stores'))


class EnergyNetworkConditions(Conditions):

    # Attribute             Type        Default Value
    tag:                    str         = init_field('Energy Network', static=True)

    nodes:                  dict        = init_field(dict)

    total_energy:           jnp.ndarray = empty_array(0)
    total_efficiency:       jnp.ndarray = empty_array(0)
    
    throttle:               jnp.ndarray = empty_array(0)
    total_power:            jnp.ndarray = empty_array(0)
    
    total_force_vector:     jnp.ndarray = empty_array((0, 3))
    total_moment_vector:    jnp.ndarray = empty_array((0, 3))

