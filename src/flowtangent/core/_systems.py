# flowtangent/Framework/State.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Jul 2024, FlowTangent Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    pass

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..components.energy import PACTNetwork

import jax

from ..components import Fuselage, LandingGear, Nacelle, PACTNetwork, Wing
from ..core._component import Component, MassProperties
from ..data.ac_classes import AircraftClass, MediumRange

# FlowTangent imports
from ..utils import Module, empty_array, field

# ----------------------------------------------------------------------------------------------------------------------
# Components
# ----------------------------------------------------------------------------------------------------------------------


class VehicleEnvelope(Module):
    # Attribute             Type        Default Value
    ultimate_load_factor: float = 0.0
    limit_load_factor: float = 0.0


# ----------------------------------------------------------------------------------------------------------------------
#  System
# ----------------------------------------------------------------------------------------------------------------------


class System(Component):
    configurations: Component = field(lambda: Component(name="Configurations"))


# ----------------------------------------------------------------------------------------------------------------------
#  Aircraft
# ----------------------------------------------------------------------------------------------------------------------


class AircraftReferenceGeometry(Module):
    mean_aerodynamic_chord: jax.Array = empty_array()
    projected_span: jax.Array = empty_array()
    aerodynamic_center: jax.Array = empty_array((0, 3))
    center_of_gravity: jax.Array = empty_array((0, 3))


class AircraftMassProperties(MassProperties):
    max_takeoff: float = 0.0
    takeoff: float = 0.0
    operating_empty: float = 0.0
    max_zero_fuel: float = 0.0
    cargo: float = 0.0


class AircraftDesign(Module):
    ac_class: AircraftClass = field(MediumRange, static=True)
    envelope: VehicleEnvelope = field(VehicleEnvelope, static=True)

    passengers: int = field(0, static=True)

    mach_number: float = field(0.0, static=True)
    range: float = field(0.0, static=True)
    cruise_alt: float = field(0.0, static=True)


class Aircraft[EnergyType: PACTNetwork](System):
    name: str = field("Aircraft", static=True)

    mass_properties: AircraftMassProperties = field(AircraftMassProperties)  # type: ignore
    design_parameters: AircraftDesign = field(AircraftDesign)

    _bookkeeping: dict = field(
        lambda: {
            "energy_networks": PACTNetwork,
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
    analysis_data: Optional[Module] = None

    def update_network_topology(self) -> Aircraft:
        sorted_network = self.energy._update_node_topology()
        return self.replace_subcomponent(sorted_network)
