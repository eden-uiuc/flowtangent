# RCAIDE/Framework/State.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Jul 2024, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from __future__ import annotations
from typing import TYPE_CHECKING, Optional

# package imports
import jax
import jax.numpy as jnp
import equinox as eqx

# RCAIDE imports
from RCAIDE.utils import empty_array, init_field

from RCAIDE.Library import Component, MassProperties
from RCAIDE.Library.Attributes import AircraftClass, MediumRange
from RCAIDE.Library.Components.Energy.Networks import EnergyNetwork
from RCAIDE.Library.Components.Wings import Wing
from RCAIDE.Library.Components.Fuselages import Fuselage
from RCAIDE.Library.Components.Nacelles import Nacelle
from RCAIDE.Library.Components.Landing_Gear import LandingGear

# ----------------------------------------------------------------------------------------------------------------------
# Components
# ----------------------------------------------------------------------------------------------------------------------


class VehicleEnvelope(eqx.Module):
    # Attribute             Type        Default Value
    ultimate_load_factor:   float        = 0.0
    limit_load_factor:      float        = 0.0

# ----------------------------------------------------------------------------------------------------------------------
#  System
# ----------------------------------------------------------------------------------------------------------------------


class System(Component):

    tag: str = init_field('System', static=True)

    configurations: Component = init_field(lambda: Component(tag='Configurations'))

# ----------------------------------------------------------------------------------------------------------------------
#  Aircraft
# ----------------------------------------------------------------------------------------------------------------------

class AircraftReferenceGeometry(eqx.Module):
    
    mean_aerodynamic_chord: jnp.ndarray = empty_array(0)
    projected_span: jnp.ndarray         = empty_array(0)
    aerodynamic_center: jnp.ndarray     = empty_array((0, 3))
    center_of_gravity: jnp.ndarray      = empty_array((0, 3))


class AircraftMassProperties(MassProperties):

    max_takeoff         :float = 0.
    takeoff             :float = 0.
    operating_empty     :float = 0.
    max_zero_fuel       :float = 0.
    cargo               :float = 0.

class Aircraft(System):

    tag:                str = init_field('Aircraft', static=True)
    
    ac_class:           AircraftClass = init_field(MediumRange, static=True)
    envelope:           VehicleEnvelope = init_field(VehicleEnvelope, static=True)
    mass_properties:    AircraftMassProperties = init_field(AircraftMassProperties) #type: ignore

    passengers:         int     = init_field(0, static=True)
    
    design_mach_number: float   = init_field(0., static=True)
    design_range:       float   = init_field(0., static=True)
    design_cruise_alt:  float   = init_field(0., static=True)

    energy:         EnergyNetwork = init_field(lambda: EnergyNetwork(tag="Energy"))
    
    wings:          Component = init_field(lambda: Component(tag='Wings'))
    fuselages:      Component = init_field(lambda: Component(tag='Fuselages'))
    nacelles:       Component = init_field(lambda: Component(tag='Nacelles'))
    landing_gear:   Component = init_field(lambda: Component(tag='Landing Gear'))

    reference_geometry:  AircraftReferenceGeometry = init_field(AircraftReferenceGeometry)
    analysis_data:      dict = init_field(dict)

    def add_subcomponent(
            self,
            subcomponent: Component,
    ):

        if isinstance(subcomponent, Wing):
            new_wings = self.wings.add_subcomponent(subcomponent)
            return eqx.tree_at(lambda a: a.wings, self, new_wings)
        elif isinstance(subcomponent, Fuselage):
            new_fuses = self.fuselages.add_subcomponent(subcomponent)
            return eqx.tree_at(lambda a: a.fuselages, self, new_fuses)
        elif isinstance(subcomponent, Nacelle):
            new_nacs = self.nacelles.add_subcomponent(subcomponent)
            return eqx.tree_at(lambda a: a.nacelles, self, new_nacs)
        elif isinstance(subcomponent, LandingGear):
            new_LGs = self.landing_gear.add_subcomponent(subcomponent)
            return eqx.tree_at(lambda a: a.landing_gear, self, new_LGs)
        else:
            return super(Aircraft, self).add_subcomponent(subcomponent)

    def get_all_components(self):
        return self.subcomponents + (
            self.energy,
            self.wings,
            self.fuselages,
            self.nacelles,
            self.landing_gear
        )


