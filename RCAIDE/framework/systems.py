# RCAIDE/Framework/State.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Jul 2024, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from __future__ import annotations

import equinox as eqx

# package imports
import jax.numpy as jnp

# RCAIDE imports
from RCAIDE.utils import empty_array, init_field

from RCAIDE.library import Component, MassProperties
from RCAIDE.library.attributes import AircraftClass, MediumRange
from RCAIDE.library.components.energy.networks import EnergyNetwork
from RCAIDE.library.components.fuselages import Fuselage
from RCAIDE.library.components.landing_gear import LandingGear
from RCAIDE.library.components.nacelles import Nacelle
from RCAIDE.library.components.wings import Wing

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


class System(Component):
    tag: str = init_field("System", static=True)

    configurations: Component = init_field(lambda: Component(tag="Configurations"))


# ----------------------------------------------------------------------------------------------------------------------
#  Aircraft
# ----------------------------------------------------------------------------------------------------------------------


class AircraftReferenceGeometry(eqx.Module):
    mean_aerodynamic_chord: jnp.ndarray = empty_array(0)
    projected_span: jnp.ndarray = empty_array(0)
    aerodynamic_center: jnp.ndarray = empty_array((0, 3))
    center_of_gravity: jnp.ndarray = empty_array((0, 3))


class AircraftMassProperties(MassProperties):
    max_takeoff: float = 0.0
    takeoff: float = 0.0
    operating_empty: float = 0.0
    max_zero_fuel: float = 0.0
    cargo: float = 0.0


class Aircraft(System):
    tag: str = init_field("Aircraft", static=True)

    ac_class: AircraftClass = init_field(MediumRange, static=True)
    envelope: VehicleEnvelope = init_field(VehicleEnvelope, static=True)
    mass_properties: AircraftMassProperties = init_field(AircraftMassProperties)  # type: ignore

    passengers: int = init_field(0, static=True)

    design_mach_number: float = init_field(0.0, static=True)
    design_range: float = init_field(0.0, static=True)
    design_cruise_alt: float = init_field(0.0, static=True)

    _bookkeeping: dict = init_field(
        lambda: {
            "energy_networks": EnergyNetwork,
            "wings": Wing,
            "fuselages": Fuselage,
            "nacelles": Nacelle,
            "landing_gear": LandingGear,
        },
        static=True,
    )

    reference_geometry: AircraftReferenceGeometry = init_field(AircraftReferenceGeometry)
    analysis_data: dict = init_field(dict)
