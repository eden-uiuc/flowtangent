# flowtangent/Library/Propellants.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: Apr 2025, Flowtangent Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

# package imports
import equinox as eqx

# Flowtangent imports
from flowtangent.data import units
from flowtangent.data.gases import O2, BurnedJetA, Gas
from flowtangent.utils import field

# ----------------------------------------------------------------------------------------------------------------------
#  Propellants
# ----------------------------------------------------------------------------------------------------------------------

class MaxPropellantMassFractions(eqx.Module):
    Air: float = field(0.0, static=True)
    O2: float = field(0.0, static=True)


class PropellantTemperatures(eqx.Module):
    flash: float = field(0.0, static=True)
    autoignition: float = field(0.0, static=True)
    freeze: float = field(0.0, static=True)
    boiling: float = field(0.0, static=True)


class Propellant(eqx.Module):
    tag: str = field("Propellant", static=True)

    oxidizer: Gas = field(Gas)

    density: float = field(0.0, static=True)
    specific_energy: float = field(0.0, static=True)
    energy_density: float = field(0.0, static=True)
    enthalpy_of_formation: float = field(0.0, static=True)

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
        boiling=0.0 * units.K
    )

class JetA(Propellant):
    oxidizer: Gas = field(O2)

    density: float = field(820.0, static=True)

    # Specific energy is higher than reference value (43.15 MJ/kg) due to stoichiometric burn assumption
    specific_energy: float = field(42.7984e6 * units.parse('J/kg'), static=True)
    energy_density: float = field(35.3e6 * units.parse('J/m**3'), static=True)

    max_mass_fraction: MaxPropellantMassFractions = field(_JetAFractions, static=True)

    temperatures: PropellantTemperatures = field(_JetATemperatures, static=True)

    def oxidized_form(self, FAR):
        return BurnedJetA(FAR)
