# flowtangent/Library/Propellants.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: Apr 2025, FlowTangent Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

# FlowTangent imports
from ..utils import Module, field, static_field
from . import units
from .gases import O2, BurnedJetA, Gas

# ----------------------------------------------------------------------------------------------------------------------
#  Propellants
# ----------------------------------------------------------------------------------------------------------------------


class MaxPropellantMassFractions(Module):
    Air: float = static_field(0.0)
    O2: float = static_field(0.0)


class PropellantTemperatures(Module):
    flash: float = static_field(0.0)
    autoignition: float = static_field(0.0)
    freeze: float = static_field(0.0)
    boiling: float = static_field(0.0)


class Propellant(Module):
    oxidizer: Gas = field(Gas)

    density: float = static_field(0.0)
    specific_energy: float = static_field(0.0)
    energy_density: float = static_field(0.0)
    enthalpy_of_formation: float = static_field(0.0)

    max_mass_fraction: MaxPropellantMassFractions = field(MaxPropellantMassFractions)
    temperatures: PropellantTemperatures = field(PropellantTemperatures)

    def oxidized_form(self, *args, **kwargs):
        raise NotImplementedError("Generic propellant class has no oxidized form.")


def _JetAFractions():
    return MaxPropellantMassFractions(Air=0.0633, O2=0.3022)


def _JetATemperatures():
    return PropellantTemperatures(
        flash=311.15 * units.K,
        autoignition=483.15 * units.K,
        freeze=233.15 * units.K,
        boiling=0.0 * units.K,
    )


class JetA(Propellant):
    oxidizer: Gas = field(O2)

    density: float = static_field(820.0)

    # Specific energy is higher than reference value (43.15 MJ/kg) due to stoichiometric burn assumption
    specific_energy: float = static_field(42.7984e6 * units.parse("J/kg"))
    energy_density: float = static_field(35.3e6 * units.parse("J/m**3"))

    max_mass_fraction: MaxPropellantMassFractions = static_field(_JetAFractions)

    temperatures: PropellantTemperatures = static_field(_JetATemperatures)

    def oxidized_form(self, FAR):
        return BurnedJetA(FAR)
