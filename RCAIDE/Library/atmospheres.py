# RCAIDE/Library/Atmospheres.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: May 2025, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from typing import Literal

# package imports
import equinox as eqx
import jax.numpy as jnp

# RCAIDE imports
from RCAIDE.utils import init_field

from RCAIDE.Library.gases import Air, IdealGas
from RCAIDE.Library.planets import Earth, Planet

# ----------------------------------------------------------------------------------------------------------------------
#  Atmospheres
# ----------------------------------------------------------------------------------------------------------------------


class AtmosphericBreakpoints(eqx.Module):
    altitude: jnp.ndarray
    temperature: jnp.ndarray
    pressure: jnp.ndarray
    density: jnp.ndarray


class Atmosphere(eqx.Module):
    tag: str = init_field("Atmosphere", static=True)

    fluid: IdealGas = init_field(Air)

    planet: Planet = init_field(Earth)

    breaks: AtmosphericBreakpoints = init_field(AtmosphericBreakpoints)

    def __repr__(self):
        return self.tag

    def _compute_property(self, altitude, property: Literal["temperature", "pressure", "density"]):
        return jnp.interp(altitude, self.breaks.altitude, getattr(self.breaks, property))

    def compute_temperature(self, altitude: jnp.ndarray | float):
        return self._compute_property(altitude, "temperature")

    def compute_pressure(self, altitude: jnp.ndarray | float):
        return self._compute_property(altitude, "pressure")

    def compute_density(self, altitude: jnp.ndarray | float):
        return self._compute_property(altitude, "density")

    def compute_speed_of_sound(self, altitude: jnp.ndarray | float):
        T = self.compute_temperature(altitude)
        return self.fluid.compute_speed_of_sound(T)

    def compute_dynamic_viscosity(self, altitude: jnp.ndarray | float):
        T = self.compute_temperature(altitude)
        return self.fluid.compute_absolute_viscosity(T)

    def compute_kinematic_viscosity(self, altitude: jnp.ndarray | float):
        mu = self.compute_dynamic_viscosity(altitude)
        rho = self.compute_density(altitude)
        return mu / rho

    def compute_thermal_conductivity(self, altitude: jnp.ndarray | float):
        T = self.compute_temperature(altitude)
        return self.fluid.compute_thermal_conductivity(T)

    def compute_prandtl_number(self, altitude: jnp.ndarray | float):
        T = self.compute_temperature(altitude)
        return self.fluid.compute_prandtl_number(T)

    def compute_gamma(self, altitude: jnp.ndarray | float):
        T = self.compute_temperature(altitude)
        return self.fluid.compute_gamma(T)

    def compute_Cp(self, altitude: jnp.ndarray | float):
        T = self.compute_temperature(altitude)
        return self.fluid.compute_Cp(T)


def _USStandardBreaks():
    return AtmosphericBreakpoints(
        altitude=jnp.array([-2.0e3, 0.0e3, 11.0e3, 20.0e3, 32.0e3, 47.0e3, 51.0e3, 71.0e3, 84.852e3]),  # m
        temperature=jnp.array([301.15, 288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.95]),  # K
        pressure=jnp.array([127774.0, 101325.0, 22632.1, 5474.89, 868.019, 110.906, 66.9389, 3.95642, 0.3734]),  # Pa
        density=jnp.array(
            [1.47808e0, 1.2250e0, 3.63918e-1, 8.80349e-2, 1.32250e-2, 1.42753e-3, 8.61606e-4, 6.42099e-5, 6.95792e-6]
        ),  # kg/m^3
    )


class USStandard1976(Atmosphere):
    tag: str = init_field("US Standard Atmosphere, 1976", static=True)
    breaks: AtmosphericBreakpoints = init_field(_USStandardBreaks)


def _ConstantTempBreaks(self):
    return AtmosphericBreakpoints(
        altitude=jnp.array([-2.0e3, 0.0e3, 11.0e3, 20.0e3, 32.0e3, 47.0e3, 51.0e3, 71.0e3, 84.852e3]),  # m
        temperature=jnp.array([301.15, 301.15, 301.15, 301.15, 301.15, 301.15, 301.15, 301.15, 301.15]),  # K
        pressure=jnp.array([127774.0, 101325.0, 22632.1, 5474.89, 868.019, 110.906, 66.9389, 3.95642, 0.3734]),  # Pa
        density=jnp.array(
            [1.545586, 1.2256523, 0.273764, 0.0662256, 0.0105000, 1.3415e-03, 8.0971e-04, 4.78579e-05, 4.51674e-06]
        ),  # kg/m^3
    )


class ConstantTemperature(Atmosphere):
    tag: str = init_field("Constant Temprerature Atmosphere", static=True)
    breaks: AtmosphericBreakpoints = init_field(_USStandardBreaks)
