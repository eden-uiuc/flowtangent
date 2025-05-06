# RCAIDE/Library/Components/Energy/Network.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: Apr 2025, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from dataclasses import dataclass, field, make_dataclass

# package imports
import numpy as np

# RCAIDE imports
import RCAIDE.Library as rcl
from RCAIDE.Library.Component import ComponentType


# ----------------------------------------------------------------------------------------------------------------------
#  Network
# ----------------------------------------------------------------------------------------------------------------------


@dataclass(kw_only=True)
class EnergyLine(rcl.Component):

    propulsors:     dataclass = field(default_factory=lambda: make_dataclass('LinePropulsors', []))
    converters:     dataclass = field(default_factory=lambda: make_dataclass('LineConverters', []))
    stores:         dataclass = field(default_factory=lambda: make_dataclass('LineStores', []))

    def add_subcomponent(self,
                         subcomponent: ComponentType,
                         sum_mass=False,
                         sum_center_of_gravity=False,
                         sum_moments_of_inertia=False
                         ):

        field_name = subcomponent.name.replace(' ', '_').lower()

        if isinstance(subcomponent, rcl.Components.Energy.EnergyConverter):
            setattr(self.converters, field_name, subcomponent)
        if (isinstance(subcomponent, rcl.Components.Energy.FuelTank) or
            isinstance(subcomponent, rcl.Components.Energy.Battery)):
            setattr(self.stores, field_name, subcomponent)


@dataclass(kw_only=True)
class EnergyNetwork(rcl.Component):

    name: str = 'Energy Network'

    lines: dataclass = field(default_factory=lambda: make_dataclass('NetworkLines', []))

