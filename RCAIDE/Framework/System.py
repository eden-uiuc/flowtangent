# RCAIDE/Framework/State.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Jul 2024, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

import unittest
from dataclasses import dataclass, field, make_dataclass
from typing import TypeVar

# package imports
import numpy as np

# RCAIDE imports
import RCAIDE.Library as rcl

ComponentType = TypeVar("ComponentType", bound="Component")

# ----------------------------------------------------------------------------------------------------------------------
# Components
# ----------------------------------------------------------------------------------------------------------------------


@dataclass(kw_only=True)
class VehicleEnvelope:
    # Attribute                 Type        Default Value
    ultimate_load:             float        = 0.0
    limit_load_factor:         float        = 0.0

# ----------------------------------------------------------------------------------------------------------------------
#  System
# ----------------------------------------------------------------------------------------------------------------------


@dataclass(kw_only=True)
class System(rcl.Component):

    name: str = 'System'

    energy:         dataclass = field(default_factory=rcl.Components.Energy.EnergyNetwork)

    configurations: dataclass = field(default_factory=
                                      lambda: make_dataclass('SystemConfigurations', []))


