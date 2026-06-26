# RCAIDE/Library/Gases.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: Apr 2025, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

# package imports
import equinox as eqx
import jax.numpy as jnp

# RCAIDE imports
from RCAIDE.utils import init_field, empty_array

# ----------------------------------------------------------------------------------------------------------------------
#  Ideal Gases
# ----------------------------------------------------------------------------------------------------------------------

class IdealGas(eqx.Module):

    tag:                    str     = init_field('Gas', static=True)
    molecular_mass:         float   = init_field(0.0, static=True)
    R:                      float   = init_field(8.314462, static=True)  # ideal gas constant (J/(mol·K))

    @property
    def R_specific(self):
        return self.R / self.molecular_mass

    # NASA 7-Coefficient Polynomials
    nasa_low_coeffs:    tuple = init_field(tuple, static=True)
    nasa_high_coeffs:   tuple = init_field(tuple, static=True)
    nasa_T_mid:         float = init_field(1000., static=True)

    thermal_coefficients:   tuple = init_field(tuple)

    def __post_init__(self):
        self.molar_mass = self.R / self.R_specific

    def compute_density(self, T: float|jnp.ndarray=298.15, P: float|jnp.ndarray=101325.):
        return P / (self.R_specific * T)

    def compute_Cp(self, T: float|jnp.ndarray=298.):
        T_arr = jnp.atleast_1d(T)

        low_poly = jnp.array(self.nasa_low_coeffs[:5])[::-1]
        high_poly = jnp.array(self.nasa_high_coeffs[:5])[::-1]

        cp_low = self.R_specific * jnp.polyval(low_poly, T_arr)
        cp_high = self.R_specific * jnp.polyval(high_poly, T_arr)

        cp_eval = jnp.where(T_arr > self.nasa_T_mid, cp_high, cp_low)

        return cp_eval.squeeze()

    def compute_enthalpy(self, T: float | jnp.ndarray = 298.15):
        T_arr = jnp.atleast_1d(T)

        def _h(coeffs, t):
            # Create a polynomial array for: a5/5*T^4 + a4/4*T^3 + a3/3*T^2 + a2/2*T + a1
            # We multiply by T outside to get the T^5 to T terms, then add a6
            poly_coeffs = jnp.array([
                coeffs[4] / 5.0,
                coeffs[3] / 4.0,
                coeffs[2] / 3.0,
                coeffs[1] / 2.0,
                coeffs[0]
            ])
            return self.R_specific * (t * jnp.polyval(poly_coeffs, t) + coeffs[5])

        h_low = _h(self.nasa_low_coeffs, T_arr)
        h_high = _h(self.nasa_high_coeffs, T_arr)

        h_eval = jnp.where(T_arr > self.nasa_T_mid, h_high, h_low)
        return h_eval.squeeze()

    def compute_entropy(self, T: float | jnp.ndarray = 298.15, P: float | jnp.ndarray = 101325.0):
        T_arr = jnp.atleast_1d(T)

        def _s0(coeffs, t):
            # Create a polynomial array for: a5/4*T^4 + a4/3*T^3 + a3/2*T^2 + a2*T
            poly_coeffs = jnp.array([
                coeffs[4] / 4.0,
                coeffs[3] / 3.0,
                coeffs[2] / 2.0,
                coeffs[1],
                0.0 # Shift by 0 so polyval evaluates correctly without an offset
            ])
            poly_part = jnp.polyval(poly_coeffs, t)

            # Combine with the ln(T) and a7 terms
            return self.R_specific * (coeffs[0] * jnp.log(t) + poly_part + coeffs[6])

        s0_low = _s0(self.nasa_low_coeffs, T_arr)
        s0_high = _s0(self.nasa_high_coeffs, T_arr)

        s0_eval = jnp.where(T_arr > self.nasa_T_mid, s0_high, s0_low)

        # Apply the pressure correction (assuming P is in Pascals)
        P_ref = 101325.0
        s_eval = s0_eval - self.R_specific * jnp.log(P / P_ref)

        return s_eval.squeeze()

    def compute_gamma(self, T: float|jnp.ndarray=298.15):
        cp = self.compute_Cp(T)
        return cp / (cp - self.R_specific)

    def compute_thermal_conductivity(self, T: float|jnp.ndarray=298.):
        return jnp.polyval(jnp.array(self.thermal_coefficients), T)

    def compute_speed_of_sound(self, T: float|jnp.ndarray=298.):
        g = self.compute_gamma(T)
        return jnp.sqrt(g * self.R_specific * T)

    def compute_absolute_viscosity(self, T: float|jnp.ndarray=298.):
        raise NotImplementedError(f'Compute absolute viscosity not implemented for this {self.tag}')

    def compute_prandtl_number(self, T: float|jnp.ndarray=298.):
        return self.compute_absolute_viscosity(T) * self.compute_Cp(T) / self.compute_thermal_conductivity(T)

def Steam() -> IdealGas:
    return IdealGas(
        tag = 'Steam',
        molecular_mass= 18.0,
        nasa_low_coeffs = (4.19864056, -2.0364341e-03,  6.52040211e-06, -5.48797062e-09, 1.77197817e-12, -30293.7267, -0.849032208),
        nasa_high_coeffs = (3.03399249, 2.17691804e-03, -1.64072518e-07, -9.7041987e-11,  1.68200992e-14, -30004.2971, 4.9667701),
    )


def CO2() -> IdealGas:
    return IdealGas(
        tag = 'Carbon Dioxide',
        molecular_mass = 44.01,
        nasa_low_coeffs = (2.34433112, 7.98052075e-03, -1.9478151e-05, 2.01572094e-08, -7.37611761e-12, -917.935173, 0.683010238),
        nasa_high_coeffs = (3.3372792, -4.94024731e-05, 4.99456778e-07, -1.79566394e-10, 2.00255376e-14, -950.158922, -3.20502331),
    )

def O2() -> IdealGas:
    return IdealGas(
        tag = 'Oxygen',
        molecular_mass = 32.00,
        nasa_low_coeffs = (3.78245636, -2.99673416e-03, 9.84730201e-06, -9.68129509e-09, 3.24372837e-12, -1063.94356, 3.65767573),
        nasa_high_coeffs = (3.28253784, 1.48308754e-03, -7.57966669e-07, 2.09470555e-10, -2.16717794e-14, -1088.45772, 5.45323129),
    )

def N2() -> IdealGas:
    return IdealGas(
        tag = 'Nitrogen',
        molecular_mass = 28.01,
        nasa_low_coeffs = (3.298677, 1.4082404e-03, -3.963222e-06, 5.641515e-09, -2.444854e-12, -1020.8999, 3.950372),
        nasa_high_coeffs = (2.926640, 1.4879768e-03, -5.684760e-07, 1.0097038e-10, -6.753351e-15, -922.7977, 5.980528),
    )

def Argon() -> IdealGas:
    return IdealGas(
        tag = 'Argon',
        molecular_mass = 39.948,
        nasa_low_coeffs = (2.5, 0.0, 0.0, 0.0, 0.0, -745.375, 4.36600118),
        nasa_high_coeffs = (2.5, 0.0, 0.0, 0.0, 0.0, -745.375, 4.36600118),
    )

# ----------------------------------------------------------------------------------------------------------------------
#  Mixed Gases
# ----------------------------------------------------------------------------------------------------------------------

element_mapping: dict = init_field({
        "o2": O2,
        "oxygen": O2,
        "n2": N2,
        "nitrogen": N2,
        "steam": Steam,
        "h2o": Steam,
        "co2": CO2,
        "carbon dioxide": CO2,
        "ar": Argon,
        "argon": Argon,
    }, static=True)

class GasComposition(eqx.Module):

    elements:           tuple[str|IdealGas, ...] = init_field(tuple)
    mass_fractions:     jnp.ndarray    = empty_array()
    mole_fractions:     jnp.ndarray    = empty_array()

    def __post_init__(self):

        new_elements = tuple()

        for elem in self.elements:
            if isinstance(elem, str) and elem.lower() in element_mapping:
                new_elements += (element_mapping[elem](),)
            elif isinstance(elem, IdealGas):
                new_elements += (elem,)
            else:
                raise ValueError(f"Unrecognized element {elem} supplied in gas composition. Discarding...")
        object.__setattr__(self, "elements", new_elements)

        if not self.mole_fractions:

            mm_mix = 1 / jnp.sum(jnp.asarray([self.mass_fractions[i]/e.molecular_mass for i, e in enumerate(self.elements)])).item()

            object.__setattr__(self, "mole_fractions", tuple(self.mass_fractions[i] * mm_mix/e.molecular_mass for i, e in enumerate(self.elements)))


class MixedGas(IdealGas):

    tag: str = "Mixed Gas"

    composition:            GasComposition = init_field(GasComposition)

    def compute_Cp(self, T: float | jnp.ndarray = 298.15):
        Cp_arr = jnp.stack([
            elem.compute_Cp(T)
            for elem in self.composition.elements
        ], axis=-1)
        frac_arr = jnp.stack(self.composition.mass_fractions, axis=-1)

        # Mass-weighted sum
        mixed_Cp = jnp.sum(Cp_arr * frac_arr, axis=-1)

        return mixed_Cp

    def compute_enthalpy(self, T: float | jnp.ndarray = 298.15):
        h_arr = jnp.stack([
            elem.compute_enthalpy(T)
            for elem in self.composition.elements
        ], axis=-1)
        frac_arr = jnp.stack(self.composition.mass_fractions, axis=-1)

        # Mass-weighted sum
        mixed_h = jnp.sum(h_arr * frac_arr, axis=-1)

        return mixed_h

    def compute_partial_pressures(self, P: float | jnp.ndarray = 101325.):
        return P * jnp.asarray(self.composition.mole_fractions)[None, :]

    def compute_entropy(self, T: float | jnp.ndarray = 298.15, P: float | jnp.ndarray = 101325.):
        X_arr = jnp.stack(self.composition.mole_fractions, axis=-1)
        S_arr = jnp.stack([
            elem.compute_entropy(T, P * X_arr[..., i])
            for i, elem in enumerate(self.composition.elements)
        ], axis=-1)
        frac_arr = jnp.stack(self.composition.mass_fractions, axis=-1)

        # Mass-weighted sum
        mixed_S = jnp.sum(S_arr * frac_arr, axis=-1)

        return mixed_S

def Air():
    return MixedGas(
        tag="Air",
        composition=GasComposition(
            elements = ("O2", "Ar", "CO2", "N2"),
            mass_fractions=jnp.asarray([0.2314, 0.0128, 0.0006, 0.7552])
        )
    )

def burned_JetA_composition(FAR: float | jnp.ndarray) -> GasComposition:
    """
    Dynamically generates the combustion product mass fractions for Jet-A.
    Assumes complete lean combustion.
    """

    # Base air masses (kg) per 1 kg of inlet air
    m_O2_air  = 0.2314
    m_N2_air  = 0.7552
    m_Ar_air  = 0.0128
    m_CO2_air = 0.0006

    # Jet-A reaction mass balances (kg per 1 kg of fuel burned)
    O2_consumed  = 3.396
    CO2_produced = 3.155
    H2O_produced = 1.242

    # Calculate new absolute masses
    # Using jnp.maximum prevents negative mass during aggressive solver iterations
    m_O2  = jnp.maximum(m_O2_air - (O2_consumed * FAR), 0.0)
    m_CO2 = m_CO2_air + (CO2_produced * FAR)
    m_H2O = H2O_produced * FAR
    m_N2  = m_N2_air # Inert
    m_Ar  = m_Ar_air # Inert

    m_total = 1.0 + FAR

    # Return the dynamically mixed composition
    return GasComposition(
        elements=("O2", "Ar", "CO2", "N2", "Steam"),
        mass_fractions=jnp.asarray([
            m_O2 / m_total,
            m_Ar / m_total,
            m_CO2 / m_total,
            m_N2 / m_total,
            m_H2O / m_total
        ])
    )

def BurnedJetA(FAR: float | jnp.ndarray) -> MixedGas:
    return MixedGas(
        tag="Burned Jet-A",
        composition=burned_JetA_composition(FAR)
    )
