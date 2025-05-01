# RCAIDE/Library/Components/Energy/Network.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: Apr 2025, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from dataclasses import dataclass, field, make_dataclass

# package imports
import numpy as np

# RCAIDE imports
import RCAIDE.Library as rcl

# ----------------------------------------------------------------------------------------------------------------------
#  Network
# ----------------------------------------------------------------------------------------------------------------------


@dataclass(kw_only=True)
class EnergyNetwork(rcl.Component):

    propulsors:         list[rcl.Component] = field(default_factory=list)

    distributors:   dataclass = field(default_factory=lambda: make_dataclass('NetworkDistributors', []))
    converters:     dataclass = field(default_factory=lambda: make_dataclass('NetworkConverters', []))
    stores:         dataclass = field(default_factory=lambda: make_dataclass('NetworkStores', []))
