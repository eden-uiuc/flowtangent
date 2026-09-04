# flowtangent/Library/Planets.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: May 2025, Flowtangent Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

# package imports
import equinox as eqx

from flowtangent.utils import field

from flowtangent.library import units

# ----------------------------------------------------------------------------------------------------------------------
#  Planets
# ----------------------------------------------------------------------------------------------------------------------


class Planet(eqx.Module):
    mass: float = field(0.0, static=True)
    mean_radius: float = field(0.0, static=True)
    sea_level_gravity: float = field(0.0, static=True)

    def compute_gravity(self, altitude: float = 0.0) -> float:

        return self.sea_level_gravity * (self.mean_radius / (self.mean_radius + altitude)) ** 2


class Earth(Planet):
    mass: float = field(5.972e24 * units.kg, static=True)
    mean_radius: float = field(6371e3 * units.m, static=True)
    sea_level_gravity: float = field(9.80665 * units.parse("m / s**2"), static=True)
    HitchHikersGuide: str = field("MostlyHarmless", static=True)
