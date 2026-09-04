# flowtangent/Library/Attributes/AC_Classes.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created:  May 2024, J. Smart
# Modified: Feb 2026, J. Smart

# -------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------

from typing import Literal

# package imports
import equinox as eqx

from flowtangent.data import units

# Flowtangent imports
from flowtangent.utils import field

# -------------------------------------------------------------------------------
# Aircraft Classes
# -------------------------------------------------------------------------------

ControlType = Literal["full_powered", "partially_powered", "full_aerodynamic"]


class FixedMasses(eqx.Module):
    flight_crew_mass: float = field(0.0, static=True)
    flight_attendants_mass: float = field(0.0, static=True)
    instruments_mass: float = field(0.0, static=True)
    avionics_mass: float = field(0.0, static=True)
    apu_mass: float = field(0.0, static=True)
    flight_control_mass: float = field(0.0, static=True)
    hyd_pnu_mass: float = field(0.0, static=True)


class PerSeatMasses(eqx.Module):
    operating_items_mass: float = field(0.0, static=True)
    electrical_equipment_mass: float = field(0.0, static=True)
    environmental_mass: float = field(0.0, static=True)
    furnishings_mass: float = field(0.0, static=True)


class AircraftClass(eqx.Module):
    tag: str = field("Aircraft Class", static=True)

    control_type: ControlType = field("full_powered", static=True)

    fixed_masses: FixedMasses = field(FixedMasses, static=True)
    per_seat_masses: PerSeatMasses = field(PerSeatMasses, static=True)


# -------------------------------------------------------------------------------
# Business Jet
# -------------------------------------------------------------------------------


def _BizJetFixed():
    return FixedMasses(
        flight_crew_mass=480.0 * units.lbm,
        flight_attendants_mass=210.0 * units.lbm,
        instruments_mass=100.0 * units.lbm,
        avionics_mass=300.0 * units.lbm,
        apu_mass=154.0 * units.lbm,
        flight_control_mass=0.0 * units.lbm,
        hyd_pnu_mass=0.0 * units.lbm,
    )


def _BizJetPer():
    return PerSeatMasses(
        operating_items_mass=28.0 * units.lbm,
        electrical_equipment_mass=13.0 * units.lbm,
        environmental_mass=15.0 * units.lbm,
        furnishings_mass=89.663 * units.lbm,
    )


class BusinessJet(AircraftClass):
    tag: str = field("Business Jet", static=True)

    fixed_masses: FixedMasses = field(_BizJetFixed, static=True)
    per_seat_masses: PerSeatMasses = field(_BizJetPer, static=True)


# -------------------------------------------------------------------------------
# Medium Range
# -------------------------------------------------------------------------------


def _MRFixed():
    return FixedMasses(
        flight_crew_mass=720.0 * units.lbm,
        flight_attendants_mass=1050.0 * units.lbm,
        instruments_mass=800.0 * units.lbm,
        avionics_mass=900.0 * units.lbm,
        apu_mass=154.0 * units.lbm,
        flight_control_mass=0.0 * units.lbm,
        hyd_pnu_mass=0.0 * units.lbm,
    )


def _MRPer():
    return PerSeatMasses(
        operating_items_mass=28.0 * units.lbm,
        electrical_equipment_mass=13.0 * units.lbm,
        environmental_mass=15.0 * units.lbm,
        furnishings_mass=89.663 * units.lbm,
    )


class MediumRange(AircraftClass):
    tag: str = field("Medium Range Jet", static=True)

    fixed_masses: FixedMasses = field(_MRFixed, static=True)
    per_seat_masses: PerSeatMasses = field(_MRPer, static=True)
