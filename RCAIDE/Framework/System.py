# RCAIDE/Framework/State.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Jul 2024, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from __future__ import annotations

# package imports
import equinox as eqx

# RCAIDE imports
import RCAIDE.Library as rcl

# ----------------------------------------------------------------------------------------------------------------------
# Components
# ----------------------------------------------------------------------------------------------------------------------

class VehicleEnvelope(eqx.Module):
    # Attribute                 Type        Default Value
    ultimate_load:             float        = 0.0
    limit_load_factor:         float        = 0.0

# ----------------------------------------------------------------------------------------------------------------------
#  System
# ----------------------------------------------------------------------------------------------------------------------


class System(rcl.Component):

    tag: str = eqx.field(static=True, default='System')

    configurations: rcl.Component = eqx.field(default_factory=lambda: rcl.Component(tag='Configurations'))


class Aircraft(System):

    tag:           str = 'Aircraft'

    energy:         rcl.Component = eqx.field(default_factory=lambda: rcl.Components.Energy.Networks.EnergyNetwork(tag="Energy"))
    wings:          rcl.Component = eqx.field(default_factory=lambda: rcl.Component(tag='Wings'))
    fuselages:      rcl.Component = eqx.field(default_factory=lambda: rcl.Component(tag='Fuselages'))
    nacelles:       rcl.Component = eqx.field(default_factory=lambda: rcl.Component(tag='Nacelles'))
    landing_gear:   rcl.Component = eqx.field(default_factory=lambda: rcl.Component(tag='Landing Gear'))

    def add_subcomponent(
            self,
            subcomponent: rcl.Component,
):

        if isinstance(subcomponent, rcl.Components.Wings.Wing):
            new_wings = self.wings.add_subcomponent(subcomponent)
            return eqx.tree_at(lambda a: a.wings, self, new_wings)
        elif isinstance(subcomponent, rcl.Components.Fuselages.Fuselage):
            new_fuses = self.fuselages.add_subcomponent(subcomponent)
            return eqx.tree_at(lambda a: a.fuselages, self, new_fuses)
        elif isinstance(subcomponent, rcl.Components.Nacelles.Nacelle):
            new_nacs = self.nacelles.add_subcomponent(subcomponent)
            return eqx.tree_at(lambda a: a.nacelles, self, new_nacs)
        elif isinstance(subcomponent, rcl.Components.Landing_Gear.LandingGear):
            new_LGs = self.landing_gear.add_subcomponent(subcomponent)
            return eqx.tree_at(lambda a: a.landing_gear, self, new_LGs)
        else:
            return super().add_subcomponent(subcomponent)

    def get_all_components(self):
        return self.subcomponents + (
            self.energy,
            self.wings,
            self.fuselages,
            self.nacelles,
            self.landing_gear
        )


