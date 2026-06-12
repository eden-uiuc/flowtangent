# RCAIDE/Library/Attributes/AC_Classes.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created:  May 2024, J. Smart
# Modified: Feb 2026, J. Smart

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from typing import Literal

# package imports
import equinox as eqx

# RCAIDE imports
from RCAIDE.utils import init_field
from RCAIDE.Library import Units

#-------------------------------------------------------------------------------
# Aircraft Classes
#-------------------------------------------------------------------------------

ControlType = Literal["full_powered", "partially_powered", "full_aerodynamic"]

class FixedMasses(eqx.Module):

    flight_crew_mass            : float = init_field(0., static=True)
    flight_attendants_mass      : float = init_field(0., static=True)
    instruments_mass            : float = init_field(0., static=True)
    avionics_mass               : float = init_field(0., static=True)
    apu_mass                    : float = init_field(0., static=True)
    flight_control_mass         : float = init_field(0., static=True)
    hyd_pnu_mass                : float = init_field(0., static=True)

class PerSeatMasses(eqx.Module):

    operating_items_mass        : float = init_field(0., static=True)
    electrical_equipment_mass   : float = init_field(0., static=True)
    environmental_mass          : float = init_field(0., static=True)
    furnishings_mass            : float = init_field(0., static=True)

class AircraftClass(eqx.Module):

    tag             : str           = init_field("Aircraft Class", static=True)

    control_type    : ControlType   = init_field("full_powered", static=True)

    fixed_masses    : FixedMasses   = init_field(FixedMasses, static=True)
    per_seat_masses : PerSeatMasses = init_field(PerSeatMasses, static=True)

#-------------------------------------------------------------------------------
# Business Jet
#-------------------------------------------------------------------------------

def _BizJetFixed():
    return FixedMasses(
        flight_crew_mass        = 480. * Units.lbm,
        flight_attendants_mass  = 210. * Units.lbm,
        instruments_mass        = 100. * Units.lbm,
        avionics_mass           = 300. * Units.lbm,
        apu_mass                = 154. * Units.lbm,
        flight_control_mass     = 0. * Units.lbm,
        hyd_pnu_mass            = 0. * Units.lbm
    )

def _BizJetPer():
    return PerSeatMasses(
        operating_items_mass        = 28. * Units.lbm,
        electrical_equipment_mass   = 13. * Units.lbm,
        environmental_mass          = 15. * Units.lbm,
        furnishings_mass            = 89.663 * Units.lbm,
    )

class BusinessJet(AircraftClass):

    tag             : str           = init_field("Business Jet", static=True)

    fixed_masses    : FixedMasses   = init_field(_BizJetFixed, static=True)
    per_seat_masses : PerSeatMasses = init_field(_BizJetPer, static=True)

#-------------------------------------------------------------------------------
# Medium Range
#-------------------------------------------------------------------------------

def _MRFixed():
    return FixedMasses(
        flight_crew_mass        = 720. * Units.lbm,
        flight_attendants_mass  = 1050. * Units.lbm,
        instruments_mass        = 800. * Units.lbm,
        avionics_mass           = 900. * Units.lbm,
        apu_mass                = 154. * Units.lbm,
        flight_control_mass     = 0. * Units.lbm,
        hyd_pnu_mass            = 0. * Units.lbm
    )

def _MRPer():
    return PerSeatMasses(
        operating_items_mass        = 28. * Units.lbm,
        electrical_equipment_mass   = 13. * Units.lbm,
        environmental_mass          = 15. * Units.lbm,
        furnishings_mass            = 89.663 * Units.lbm,
    )

class MediumRange(AircraftClass):

    tag             : str           = init_field("Medium Range Jet", static=True)

    fixed_masses    : FixedMasses   = init_field(_MRFixed, static=True)
    per_seat_masses : PerSeatMasses = init_field(_MRPer, static=True)
