# flowtangent/Framework/State.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Jul 2024, Flowtangent Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from __future__ import annotations

import equinox as eqx

# package imports
import jax.numpy as jnp
from flowtangent.library import Component, MassProperties
from flowtangent.library.attributes import AircraftClass, MediumRange
from flowtangent.library.components.energy.networks import GraphNetwork
from flowtangent.library.components.fuselages import Fuselage
from flowtangent.library.components.landing_gear import LandingGear
from flowtangent.library.components.nacelles import Nacelle
from flowtangent.library.components.wings import Wing

# Flowtangent imports
from flowtangent.utils import empty_array, field, register

# ----------------------------------------------------------------------------------------------------------------------
# Components
# ----------------------------------------------------------------------------------------------------------------------


class VehicleEnvelope(eqx.Module):
    # Attribute             Type        Default Value
    ultimate_load_factor: float = 0.0
    limit_load_factor: float = 0.0


# ----------------------------------------------------------------------------------------------------------------------
#  System
# ----------------------------------------------------------------------------------------------------------------------

@register
class System(Component):
    tag: str = field("System", static=True)

    configurations: Component = field(lambda: Component(tag="Configurations"))


# ----------------------------------------------------------------------------------------------------------------------
#  Aircraft
# ----------------------------------------------------------------------------------------------------------------------

@register
class AircraftReferenceGeometry(eqx.Module):
    mean_aerodynamic_chord: jnp.ndarray = empty_array()
    projected_span: jnp.ndarray = empty_array()
    aerodynamic_center: jnp.ndarray = empty_array((0, 3))
    center_of_gravity: jnp.ndarray = empty_array((0, 3))

@register
class AircraftMassProperties(MassProperties):
    max_takeoff: float = 0.0
    takeoff: float = 0.0
    operating_empty: float = 0.0
    max_zero_fuel: float = 0.0
    cargo: float = 0.0

@register
class AircraftDesign(eqx.Module):

    ac_class: AircraftClass = field(MediumRange, static=True)
    envelope: VehicleEnvelope = field(VehicleEnvelope, static=True)

    passengers: int = field(0, static=True)

    mach_number: float = field(0.0, static=True)
    range: float = field(0.0, static=True)
    cruise_alt: float = field(0.0, static=True)

@register
class Aircraft[EnergyType: GraphNetwork](System):
    tag: str = field("Aircraft", static=True)

    mass_properties: AircraftMassProperties = field(AircraftMassProperties)  # type: ignore
    design_parameters: AircraftDesign = field(AircraftDesign)

    _bookkeeping: dict = field(
        lambda: {
            "energy_networks": GraphNetwork,
            "wings": Wing,
            "fuselages": Fuselage,
            "nacelles": Nacelle,
            "landing_gear": LandingGear,
        },
        static=True,
    )

    @property
    def energy(self) -> EnergyType:
        return self.energy_networks[0]

    reference_geometry: AircraftReferenceGeometry = field(AircraftReferenceGeometry)
    analysis_data: dict = field(dict)

    def update_network_topology(self) -> Aircraft:
        sorted_network = self.energy.update_node_topology()
        return self.replace_subcomponent(sorted_network)

