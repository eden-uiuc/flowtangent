# Trace/Library/Planets.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: May 2025, Trace Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

# package imports
import equinox as eqx

from eden_trace.utils import init_field

from eden_trace.library import units

# ----------------------------------------------------------------------------------------------------------------------
#  Planets
# ----------------------------------------------------------------------------------------------------------------------


class Planet(eqx.Module):
    mass: float = init_field(0.0, static=True)
    mean_radius: float = init_field(0.0, static=True)
    sea_level_gravity: float = init_field(0.0, static=True)

    def compute_gravity(self, altitude: float = 0.0) -> float:

        return self.sea_level_gravity * (self.mean_radius / (self.mean_radius + altitude)) ** 2


class Earth(Planet):
    mass: float = init_field(5.972e24 * units.kg, static=True)
    mean_radius: float = init_field(6371e3 * units.m, static=True)
    sea_level_gravity: float = init_field(9.80665 * units.parse("m / s**2"), static=True)
    HitchHikersGuide: str = init_field("MostlyHarmless", static=True)
