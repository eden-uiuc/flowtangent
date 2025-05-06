# RCAIDE/Framework/State.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Jul 2024, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

import unittest
from dataclasses import dataclass, field, make_dataclass
from typing import TypeVar

# package imports
import numpy as np

# RCAIDE imports
import RCAIDE.Library as rcl

ComponentType = TypeVar("ComponentType", bound="Component")

# ----------------------------------------------------------------------------------------------------------------------
# Components
# ----------------------------------------------------------------------------------------------------------------------


@dataclass(kw_only=True)
class VehicleEnvelope:
    # Attribute                 Type        Default Value
    ultimate_load:             float        = 0.0
    limit_load_factor:         float        = 0.0

# ----------------------------------------------------------------------------------------------------------------------
#  System
# ----------------------------------------------------------------------------------------------------------------------


@dataclass(kw_only=True)
class System(rcl.Component):

    name: str = 'System'

    configurations: dataclass = field(default_factory=
                                      lambda: make_dataclass('SystemConfigurations', []))


@dataclass(kw_only=True)
class Aircraft(System):

    name:           str = 'Aircraft'

    energy:         dataclass = field(default_factory=rcl.Components.Energy.EnergyNetwork)

    wings:          dataclass = field(default_factory=
                                      lambda: make_dataclass(cls_name='Wings',
                                                             fields=[])
                                      )

    fuselages:      dataclass = field(default_factory=
                                      lambda: make_dataclass(cls_name='Fuselages',
                                                             fields=[])
                                      )

    nacelles:       dataclass = field(default_factory=
                                      lambda: make_dataclass(cls_name='Nacelles',
                                                             fields=[])
                                      )

    landing_gear:   dataclass = field(default_factory=
                                      lambda: make_dataclass(cls_name='LandingGear',
                                                             fields=[])
                                      )

    def add_subcomponent(self,
                         subcomponent: rcl.Component,
                         sum_mass=False,
                         sum_center_of_gravity=False,
                         sum_moments_of_inertia=False
                         ):

        if isinstance(subcomponent, rcl.Components.Wing):
            setattr(self.wings, subcomponent.get_field_name(), subcomponent)
        elif isinstance(subcomponent, rcl.Components.Fuselage):
            setattr(self.fuselages, subcomponent.get_field_name(), subcomponent)
        elif isinstance(subcomponent, rcl.Components.Nacelle):
            setattr(self.nacelles, subcomponent.get_field_name(), subcomponent)

        super().add_subcomponent(subcomponent, sum_mass, sum_center_of_gravity, sum_moments_of_inertia)


