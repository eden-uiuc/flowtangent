# flowtangent/data/planets.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: May 2025, FlowTangent Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from ..utils import Module, static_field
from . import units

# ----------------------------------------------------------------------------------------------------------------------
#  Planets
# ----------------------------------------------------------------------------------------------------------------------


class Planet(Module):
    mass: float = static_field(0.0)
    mean_radius: float = static_field(0.0)
    sea_level_gravity: float = static_field(0.0)

    def compute_gravity(self, altitude: float = 0.0) -> float:

        return self.sea_level_gravity * (self.mean_radius / (self.mean_radius + altitude)) ** 2


class Earth(Planet):
    mass: float = static_field(5.972e24 * units.kg)
    mean_radius: float = static_field(6371e3 * units.m)
    sea_level_gravity: float = static_field(9.80665 * units.parse("m / s**2"))
    HitchHikersGuide: str = static_field("MostlyHarmless")
