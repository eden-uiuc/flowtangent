# RCAIDE/Library/Propellants.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: Apr 2025, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from dataclasses import dataclass, field

# package imports
import numpy as np

# RCAIDE imports
import RCAIDE.Library as rcl

# ----------------------------------------------------------------------------------------------------------------------
#  Propellants
# ----------------------------------------------------------------------------------------------------------------------


@dataclass(kw_only=True)
class MaxPropellantMassFractions:

    Air:    float = 0.0
    O2:     float = 0.0


@dataclass(kw_only=True)
class PropellantTemperatures:

    flash:          float = 0.0
    autoignition:   float = 0.0
    freeze:         float = 0.0
    boiling:        float = 0.0


@dataclass(kw_only=True)
class Propellant:

    name: str = 'Propellant'

    oxidizer: rcl.Gas = field(default_factory=rcl.Gas)

    density:            float = 0.0
    specific_energy:    float = 0.0
    energy_density:     float = 0.0

    max_mass_fraction:  MaxPropellantMassFractions  = field(default_factory=MaxPropellantMassFractions)
    temperatures:       PropellantTemperatures      = field(default_factory=PropellantTemperatures)


@dataclass(kw_only=True)
class JetA(Propellant):

    oxidizer: rcl.Gas = field(default_factory=rcl.O2)

    density         = 820.
    specific_energy = 43.02e6
    energy_density  = 35276.4e6

    max_mass_fraction = MaxPropellantMassFractions(Air=0.0633,
                                                   O2=0.3022)

    temperatures = PropellantTemperatures(flash=311.15,
                                          autoignition=483.15,
                                          freeze=233.15,
                                          boiling=0.0)

