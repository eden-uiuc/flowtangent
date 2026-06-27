# RCAIDE/Library/Propellants.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: Apr 2025, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

# package imports
import equinox as eqx

from RCAIDE.utils import init_field

# RCAIDE imports
from RCAIDE.Library.gases import O2, IdealGas

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

    oxidizer: IdealGas = init_field(IdealGas)

    density: float = init_field(0.0, static=True)
    specific_energy: float = init_field(0.0, static=True)
    energy_density: float = init_field(0.0, static=True)

    max_mass_fraction: MaxPropellantMassFractions = init_field(MaxPropellantMassFractions)
    temperatures: PropellantTemperatures = init_field(PropellantTemperatures)


def _JetAFractions():
    return MaxPropellantMassFractions(Air=0.0633, O2=0.3022)


def _JetATemperatures():
    return PropellantTemperatures(flash=311.15, autoignition=483.15, freeze=233.15, boiling=0.0)


class JetA(Propellant):
    oxidizer: IdealGas = init_field(O2)

    density: float = init_field(820.0, static=True)
    specific_energy: float = init_field(43.02e6, static=True)
    energy_density: float = init_field(35276.4e6, static=True)

    max_mass_fraction: MaxPropellantMassFractions = init_field(_JetAFractions, static=True)

    temperatures: PropellantTemperatures = init_field(_JetATemperatures, static=True)
