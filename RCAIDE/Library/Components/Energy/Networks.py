# RCAIDE/Library/Components/Energy/Network.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: Apr 2025, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

# package imports
import chex
from dataclasses import field

# RCAIDE imports
import RCAIDE.Framework as rcf
import RCAIDE.Library as rcl

# ----------------------------------------------------------------------------------------------------------------------
#  Network
# ----------------------------------------------------------------------------------------------------------------------


@chex.dataclass(kw_only=True)
class EnergyLine(rcl.Component):

    propulsors:     rcl.Component = field(default_factory=lambda: rcl.Component(tag='Propulsors'))
    converters:     rcl.Component = field(default_factory=lambda: rcl.Component(tag='Converters'))
    stores:         rcl.Component = field(default_factory=lambda: rcl.Component(tag='Stores'))

    def add_subcomponent(
            self,
            subcomponent: rcl.Component,
            sum_mass=False,
            sum_center_of_gravity=False,
            sum_moments_of_inertia=False
    ):

        if isinstance(subcomponent, rcl.Components.Energy.Propulsors.Propulsor):
            self.propulsors.add_subcomponent(subcomponent, sum_mass, sum_center_of_gravity, sum_moments_of_inertia)
        elif isinstance(subcomponent, rcl.Components.Energy.Converters.EnergyConverter):
            self.converters.add_subcomponent(subcomponent, sum_mass, sum_center_of_gravity, sum_moments_of_inertia)
        elif isinstance(subcomponent, rcl.Components.Energy.Stores.EnergyStore):
            self.stores.add_subcomponent(subcomponent, sum_mass, sum_center_of_gravity, sum_moments_of_inertia)
        else:
            super(EnergyLine, self).add_subcomponent(subcomponent, sum_mass, sum_center_of_gravity, sum_moments_of_inertia)

    def __post_init__(self):
        self.add_subcomponent(self.propulsors)
        self.add_subcomponent(self.converters)
        self.add_subcomponent(self.stores)

    @staticmethod
    def calculate_performance(state: "rcf.State",
                              system: "rcf.System",
                              settings: "rcf.Settings"
                              ):

        raise NotImplementedError('Subclasses must implement this method')


@chex.dataclass(kw_only=True)
class EnergyNetwork(rcl.Component):

    tag: str = 'Energy Network'

    efficiency: float = 1.0

    lines: rcl.Component = field(default_factory=lambda: rcl.Component(tag='Lines'))

    def __post_init__(self):
        self.add_subcomponent(self.lines)

    @staticmethod
    def calculate_performance(state: "rcf.State",
                              system: "rcf.System",
                              settings: "rcf.Settings"
                              ):

        for line in system.energy.lines:
            state, system, settings = line.calculate_performance(state, system, settings)

        return state, system, settings

