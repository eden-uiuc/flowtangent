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
#  Energy Networks and Stores
# ----------------------------------------------------------------------------------------------------------------------


class EnergyStoreConditions(Conditions):

    # Attribute         Type        Default Value
    tag:                str         = init_field('Energy Store', static=True)

    total_energy:       jnp.ndarray = empty_array(0)


class EnergyConverterConditions(Conditions):

    # Attribute         Type        Default Value
    tag:                str         = init_field('Energy Converter', static=True)

    efficiency:         jnp.ndarray  = empty_array(0)
    power:              jnp.ndarray  = empty_array(0)

    thrust_vector:      jnp.ndarray  = init_field(lambda: jnp.zeros((1, 3)))

    x_axis_rotation:    jnp.ndarray  = empty_array(0)
    y_axis_rotation:    jnp.ndarray  = empty_array(0)
    z_axis_rotation:    jnp.ndarray  = empty_array(0)

    inputs:             Conditions  = init_field(lambda: Conditions(tag='Energy Converter Inputs'))
    outputs:            Conditions  = init_field(lambda: Conditions(tag='Energy Converter Outputs'))


# ----------------------------------------------------------------------------------------------------------------------
#  Battery Stores
# ----------------------------------------------------------------------------------------------------------------------


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

class FuelConditions(EnergyStoreConditions):

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

    total_energy:           jnp.ndarray = empty_array(0)
    total_efficiency:       jnp.ndarray = empty_array(0)
    
    throttle:               jnp.ndarray = empty_array(0)
    total_power:            jnp.ndarray = empty_array(0)
    
    total_force_vector:     jnp.ndarray = empty_array((0, 3))
    total_moment_vector:    jnp.ndarray = empty_array((0, 3))

    lines:                  Conditions  = init_field(lambda: Conditions(tag='Energy Network Lines'))
