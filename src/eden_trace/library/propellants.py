# Trace/Library/Propellants.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: Apr 2025, Trace Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

# package imports
import equinox as eqx

from eden_trace.utils import init_field

# Trace imports
from eden_trace.library import units
from eden_trace.library.gases import O2, Gas, BurnedJetA

# ----------------------------------------------------------------------------------------------------------------------
#  Propellants
# ----------------------------------------------------------------------------------------------------------------------

class MaxPropellantMassFractions(eqx.Module):
    Air: float = init_field(0.0, static=True)
    O2: float = init_field(0.0, static=True)


class PropellantTemperatures(eqx.Module):
    flash: float = init_field(0.0, static=True)
    autoignition: float = init_field(0.0, static=True)
    freeze: float = init_field(0.0, static=True)
    boiling: float = init_field(0.0, static=True)


class Propellant(eqx.Module):
    tag: str = init_field("Propellant", static=True)

    oxidizer: Gas = init_field(Gas)

    density: float = init_field(0.0, static=True)
    specific_energy: float = init_field(0.0, static=True)
    energy_density: float = init_field(0.0, static=True)
    enthalpy_of_formation: float = init_field(0.0, static=True)

    max_mass_fraction: MaxPropellantMassFractions = init_field(MaxPropellantMassFractions)
    temperatures: PropellantTemperatures = init_field(PropellantTemperatures)

    def oxidized_form(self, *args, **kwargs):
        raise NotImplementedError("Generic propellant class has no oxidized form.")


def _JetAFractions():
    return MaxPropellantMassFractions(Air=0.0633, O2=0.3022)


def _JetATemperatures():
    return PropellantTemperatures(
        flash=311.15 * units.K,
        autoignition=483.15 * units.K,
        freeze=233.15 * units.K,
        boiling=0.0 * units.K
    )

class JetA(Propellant):
    oxidizer: Gas = init_field(O2)

    density: float = init_field(820.0, static=True)
    
    # Specific energy is higher than reference value (43.15 MJ/kg) due to stoichiometric burn assumption
    specific_energy: float = init_field(44.0e6 * units.parse('J/kg'), static=True)
    energy_density: float = init_field(35.3e6 * units.parse('J/m**3'), static=True)

    max_mass_fraction: MaxPropellantMassFractions = init_field(_JetAFractions, static=True)

    temperatures: PropellantTemperatures = init_field(_JetATemperatures, static=True)

    def oxidized_form(self, FAR):
        return BurnedJetA(FAR)
