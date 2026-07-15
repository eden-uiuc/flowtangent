# Trace/Library/Atmospheres.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: May 2025, Trace Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from typing import Literal

# package imports
import numpy as np      # For precomputing atmospheric tables
import equinox as eqx
import jax.numpy as jnp

# Trace imports
from eden_trace.utils import init_field

from eden_trace.library.gases import Air, IdealGas
from eden_trace.library.planets import Earth, Planet

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


def generate_us_standard_atmosphere(max_alt=84852.0, step=10.0):
    # 1976 Standard Atmosphere Base layers
    # Alt (m), Temp (K), Press (Pa), Lapse Rate (K/m)
    layers = [
        (-2000.0, 301.15, 127774.0, -0.0065),
        (0.0, 288.15, 101325.0, -0.0065),
        (11000.0, 216.65, 22632.1, 0.0),
        (20000.0, 216.65, 5474.89, 0.001),
        (32000.0, 228.65, 868.019, 0.0028),
        (47000.0, 270.65, 110.906, 0.0),
        (51000.0, 270.65, 66.9389, -0.0028),
        (71000.0, 214.65, 3.95642, -0.002),
        (84852.0, 186.95, 0.3734, 0.0)
    ]
    
    R = 287.0528
    g0 = 9.80665
    
    alts = np.arange(-2000.0, max_alt + step, step)
    temps = np.zeros_like(alts)
    press = np.zeros_like(alts)
    
    hb, Tb, Pb, L = layers[0]
    for i, h in enumerate(alts):
        # Find which layer we are in
        for j in range(len(layers)-1):
            if layers[j][0] <= h < layers[j+1][0] or (j == len(layers)-2 and h >= layers[j+1][0]):
                hb, Tb, Pb, L = layers[j]
                break
                
        # Calculate T
        T = Tb + L * (h - hb)
        temps[i] = T
        
        # Calculate P
        if L == 0.0: # Isothermal
            press[i] = Pb * np.exp(-g0 * (h - hb) / (R * Tb))
        else:        # Gradient
            press[i] = Pb * (T / Tb)**(-g0 / (R * L))
            
    densities = press / (R * temps)
    
    return AtmosphericBreakpoints(
        altitude=jnp.array(alts),
        temperature=jnp.array(temps),
        pressure=jnp.array(press),
        density=jnp.array(densities)
    )

class USStandard1976(Atmosphere):
    tag: str = init_field("US Standard Atmosphere, 1976", static=True)
    breaks: AtmosphericBreakpoints = init_field(generate_us_standard_atmosphere)

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
    breaks: AtmosphericBreakpoints = init_field(_ConstantTempBreaks)
