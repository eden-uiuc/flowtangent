# RCAIDE/Library/Planets.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: May 2025, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

# package imports
import equinox as eqx

# ----------------------------------------------------------------------------------------------------------------------
#  Planets
# ----------------------------------------------------------------------------------------------------------------------


class Planet(eqx.Module):

    mass:           float = eqx.field(static=True, default=0.0)
    mean_radius:    float = eqx.field(static=True, default=0.0)


class Earth(Planet):

    mass:               float = eqx.field(static=True, default=5.972e24)  # in kg
    mean_radius:        float = eqx.field(static=True, default=6371e3)  # in meters
    sea_level_gravity:  float = eqx.field(static=True, default=9.80665)  # in m/s^2

    def compute_gravity(self, altitude: float = 0.0) -> float:

        return self.sea_level_gravity * (self.mean_radius / (self.mean_radius + altitude)) ** 2
