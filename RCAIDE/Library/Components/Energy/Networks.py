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
import RCAIDE.Framework as rcf
import RCAIDE.Library as rcl

# ----------------------------------------------------------------------------------------------------------------------
#  Network
# ----------------------------------------------------------------------------------------------------------------------


@dataclass(kw_only=True)
class EnergyLine(rcl.Component):

    converters:     dataclass = field(default_factory=lambda: rcl.Component(name='Converters'))
    stores:         dataclass = field(default_factory=lambda: rcl.Component(name='Stores'))

    def add_subcomponent(self,
                         subcomponent: rcl.Component,
                         sum_mass=False,
                         sum_center_of_gravity=False,
                         sum_moments_of_inertia=False
                         ):

        field_name = subcomponent.name.replace(' ', '_').lower()

        if isinstance(subcomponent, rcl.Components.Energy.Converters.EnergyConverter):
            setattr(self.converters, field_name, subcomponent)
        if (isinstance(subcomponent, rcl.Components.Energy.Stores.FuelTank) or
            isinstance(subcomponent, rcl.Components.Energy.Stores.Battery)):
            setattr(self.stores, field_name, subcomponent)

    def __post_init__(self):
        self.add_subcomponent(rcl.Component(name='Converters'))
        self.add_subcomponent(rcl.Component(name='Stores'))

    @staticmethod
    def calculate_performance(state: "rcf.State",
                              system: "rcf.System",
                              settings: "rcf.Settings"
                              ):

        raise NotImplementedError('Subclasses must implement this method')


@dataclass(kw_only=True)
class EnergyNetwork(rcl.Component):

    name: str = 'Energy Network'

    efficiency: float = 1.0

    lines: list[rcl.Component] = field(default_factory=list)

    @staticmethod
    def calculate_performance(state: "rcf.State",
                              system: "rcf.System",
                              settings: "rcf.Settings"
                              ):

        for line in system.energy.lines:
            state, system, settings = line.calculate_performance(state, system, settings)

        return state, system, settings

