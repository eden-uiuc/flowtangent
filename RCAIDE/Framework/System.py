# RCAIDE/Framework/State.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Jul 2024, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

import unittest
import chex
from dataclasses import field, make_dataclass
from typing import TypeVar

# package imports
import numpy as np

# RCAIDE imports
import RCAIDE.Library as rcl

ComponentType = TypeVar("ComponentType", bound="Component")

# ----------------------------------------------------------------------------------------------------------------------
# Components
# ----------------------------------------------------------------------------------------------------------------------


@chex.dataclass(kw_only=True)
class VehicleEnvelope:
    # Attribute                 Type        Default Value
    ultimate_load:             float        = 0.0
    limit_load_factor:         float        = 0.0

# ----------------------------------------------------------------------------------------------------------------------
#  System
# ----------------------------------------------------------------------------------------------------------------------


@chex.dataclass(kw_only=True)
class System(rcl.Component):

    name: str = 'System'

    configurations: rcl.Component = field(default_factory=lambda: rcl.Component(name='Configurations'))


@chex.dataclass(kw_only=True)
class Aircraft(System):

    name:           str = 'Aircraft'

    energy:         rcl.Component = field(default_factory=lambda: rcl.Components.Energy.Networks.EnergyNetwork())

    wings:          rcl.Component = field(default_factory=lambda: rcl.Component(name='Wings'))

    fuselages:      rcl.Component = field(default_factory=lambda: rcl.Component(name='Fuselages'))

    nacelles:       rcl.Component = field(default_factory=lambda: rcl.Component(name='Nacelles'))

    landing_gear:   rcl.Component = field(default_factory=lambda: rcl.Component(name='Landing Gear'))

    def add_subcomponent(
            self,
            subcomponent: rcl.Component,
            sum_mass=False,
            sum_center_of_gravity=False,
            sum_moments_of_inertia=False
):

        if isinstance(subcomponent, rcl.Components.Wings.Wing):
            self.wings.add_subcomponent(subcomponent, sum_mass, sum_center_of_gravity, sum_moments_of_inertia)
        elif isinstance(subcomponent, rcl.Components.Fuselages.Fuselage):
            self.fuselages.add_subcomponent(subcomponent, sum_mass, sum_center_of_gravity, sum_moments_of_inertia)
        elif isinstance(subcomponent, rcl.Components.Nacelles.Nacelle):
            self.nacelles.add_subcomponent(subcomponent, sum_mass, sum_center_of_gravity, sum_moments_of_inertia)
        elif isinstance(subcomponent, rcl.Components.Landing_Gear.LandingGear):
            self.landing_gear.add_subcomponent(subcomponent, sum_mass, sum_center_of_gravity, sum_moments_of_inertia)
        else:
            super(Aircraft, self).add_subcomponent(subcomponent, sum_mass, sum_center_of_gravity, sum_moments_of_inertia)

    def __post_init__(self):
        self.add_subcomponent(self.energy)
        self.add_subcomponent(self.wings)
        self.add_subcomponent(self.fuselages)
        self.add_subcomponent(self.nacelles)
        self.add_subcomponent(self.landing_gear)


