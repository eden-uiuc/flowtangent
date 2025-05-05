# RCAIDE/Library/Planets.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: May 2025, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from dataclasses import dataclass, field

# package imports
import numpy as np

# RCAIDE imports
import RCAIDE.Framework as rcf

# ----------------------------------------------------------------------------------------------------------------------
#  Planets
# ----------------------------------------------------------------------------------------------------------------------


@dataclass(kw_only=True)
class Planet:

    mass: float = 0.0
    mean_radius: float = 0.0


@dataclass(kw_only=True)
class Earth(Planet):

    mass: float = 5.972e24  # in kg
    mean_radius: float = 6371e3  # in meters
    sea_level_gravity: float = 9.80665  # in m/s^2

    def compute_gravity(self, altitude: float = 0.0) -> float:

        return self.sea_level_gravity * (self.mean_radius / (self.mean_radius + altitude)) ** 2
