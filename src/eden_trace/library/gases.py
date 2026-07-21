# Trace/Library/Gases.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: Apr 2025, Trace Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
import json
from functools import lru_cache
from collections import defaultdict

# package imports
import jax
import equinox as eqx
import jax.numpy as jnp

from eden_trace.utils import empty_array, get_Trace_root, init_field

# Trace imports
from eden_trace.library import units

# ----------------------------------------------------------------------------------------------------------------------
#  Ideal Gases
# ----------------------------------------------------------------------------------------------------------------------


class IdealGas(eqx.Module):
    tag: str = init_field("Gas", static=True)
    molecular_mass: float = 1.0 * units.gram
    R: float = init_field(8.314462, static=True)  # ideal gas constant (J/(mol·K))

    @property
    def R_specific(self):
        return self.R / self.molecular_mass

    # NASA 7-Coefficient Polynomials
    nasa_low_coeffs: tuple = init_field(tuple, static=True)
    nasa_high_coeffs: tuple = init_field(tuple, static=True)
    nasa_T_mid: float = init_field(1000.0, static=True)

    thermal_coefficients: tuple = init_field(tuple)

    def __repr__(self) -> str:
        return f"{self.tag}".upper()

    def __post_init__(self):
        self.molecular_mass = self.R / self.R_specific

    def compute_density(self, T: float | jnp.ndarray = 298.15, P: float | jnp.ndarray = 101325.0):
        return P / (self.R_specific * T)

    def compute_Cp(self, T: float | jnp.ndarray = 298.0):
        # T_arr = jnp.atleast_2d(T)

        low_poly = jnp.array(self.nasa_low_coeffs[:5])[::-1]
        high_poly = jnp.array(self.nasa_high_coeffs[:5])[::-1]

        cp_low = self.R_specific * jnp.polyval(low_poly, T)
        cp_high = self.R_specific * jnp.polyval(high_poly, T)

        cp_eval = jnp.where(T > self.nasa_T_mid, cp_high, cp_low)

        return cp_eval

    def compute_absolute_enthalpy(self, T: float | jnp.ndarray = 298.15):
        T_arr = jnp.atleast_1d(T)

        def _h(coeffs, t):
            # Create a polynomial array for: a5/5*T^4 + a4/4*T^3 + a3/3*T^2 + a2/2*T + a1
            # We multiply by T outside to get the T^5 to T terms, then add a6
            poly_coeffs = jnp.array([coeffs[4] / 5.0, coeffs[3] / 4.0, coeffs[2] / 3.0, coeffs[1] / 2.0, coeffs[0]])
            return self.R_specific * (t * jnp.polyval(poly_coeffs, t) + coeffs[5])

        h_low = _h(self.nasa_low_coeffs, T_arr)
        h_high = _h(self.nasa_high_coeffs, T_arr)

        h_eval = jnp.where(T_arr > self.nasa_T_mid, h_high, h_low)
        return h_eval.squeeze()
    
    def compute_enthalpy(self, T: float | jnp.ndarray = 298.15):
        h_abs = self.compute_absolute_enthalpy(T)
        h_ref = self.compute_absolute_enthalpy(298.15)
        return h_abs - h_ref

    def invert_enthalpy(self, h_target: float | jnp.ndarray, T_guess: float | jnp.ndarray=1000.0):

        def newton_step(i, T):
            h = self.compute_enthalpy(T)
            cp = self.compute_Cp(T)

            return T - (h - h_target) / cp

        T_final = jax.lax.fori_loop(0, 5, newton_step, T_guess)
        return T_final

    def compute_entropy(self, T: float | jnp.ndarray = 298.15, P: float | jnp.ndarray = 101325.0):
        T_arr = jnp.atleast_1d(T)

        def _s0(coeffs, t):
            # Create a polynomial array for: a5/4*T^4 + a4/3*T^3 + a3/2*T^2 + a2*T
            poly_coeffs = jnp.array(
                [
                    coeffs[4] / 4.0,
                    coeffs[3] / 3.0,
                    coeffs[2] / 2.0,
                    coeffs[1],
                    0.0,  # Shift by 0 so polyval evaluates correctly without an offset
                ]
            )
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

    def compute_gamma(self, T: float | jnp.ndarray = 298.15):
        cp = self.compute_Cp(T)
        return cp / (cp - self.R_specific)

    def compute_thermal_conductivity(self, T: float | jnp.ndarray = 298.0):
        return jnp.polyval(jnp.array(self.thermal_coefficients), T)

    def compute_speed_of_sound(self, T: float | jnp.ndarray = 298.0):
        g = self.compute_gamma(T)
        return jnp.sqrt(g * self.R_specific * T)

    def compute_absolute_viscosity(self, T: float | jnp.ndarray = 298.0):
        return 1.8e-5

    def compute_prandtl_number(self, T: float | jnp.ndarray = 298.0):
        return self.compute_absolute_viscosity(T) * self.compute_Cp(T) / self.compute_thermal_conductivity(T)

# ----------------------------------------------------------------------------------------------------------------------
#  Mixed Gases
# ----------------------------------------------------------------------------------------------------------------------

class GasComposition(eqx.Module):
    elements: tuple[str | IdealGas, ...] = init_field(tuple)
    mass_fractions: jnp.ndarray = empty_array()

    def __repr__(self) -> str:
        return "; ".join([f"{e.tag}: {jnp.atleast_2d(self.mass_fractions[:, i]).squeeze()}" for i, e in enumerate(self.elements)])
    
    def flatten(self):

        def _extract_elements(comp: GasComposition, current_fraction: jnp.ndarray) -> dict[str, jnp.ndarray]:
            base_dict = defaultdict(lambda: jnp.zeros_like(current_fraction))
            for e_idx, elem in enumerate(comp.elements):
                
                e_frac = current_fraction * jnp.atleast_2d(comp.mass_fractions)[..., e_idx:e_idx+1]
                
                if hasattr(elem, "composition"):
                    sub_dict = _extract_elements(elem.composition, e_frac)
                    for gas, frac in sub_dict.items():
                        gas_name = str(gas).upper() 
                        base_dict[gas_name] = base_dict[gas_name] + frac
                else:
                    gas_name = str(elem).upper()
                    base_dict[gas_name] = base_dict[gas_name] + e_frac
            
            return base_dict

        base_dict = _extract_elements(self, jnp.ones((1, 1)))
        sorted_elements = tuple(sorted(base_dict.keys()))
        stacked_fractions = jnp.concatenate([base_dict[k] for k in sorted_elements], axis=-1)

        flat_composition =  GasComposition(
            elements=sorted_elements,
            mass_fractions=stacked_fractions
        )

        return flat_composition

    
    def __post_init__(self):

        new_elements = tuple()

        for elem in self.elements:

            if isinstance(elem, str):
                try:
                    new_elements += (_get_gas(elem.upper()),)
                except AttributeError:
                    raise ValueError(f"Unrecognized element '{elem}' not found in database.")
            elif isinstance(elem, IdealGas):
                new_elements += (elem,)
            else:
                raise ValueError(f"Invalid element type {type(elem)} supplied. Discarding...")
        object.__setattr__(self, "elements", new_elements)

    @property
    def mole_fractions(self):
        mm_mix = (
            1
            / jnp.sum(
                jnp.asarray([self.mass_fractions[i] / e.molecular_mass for i, e in enumerate(self.elements)])
            ).item()
        )

        return tuple(self.mass_fractions[i] * mm_mix / e.molecular_mass for i, e in enumerate(self.elements)),


class MixedGas(IdealGas):
    tag: str = init_field("Mixed Gas", static=True)

    composition: GasComposition = init_field(GasComposition)

    def __post_init__(self):
        object.__setattr__(self, "composition", self.composition.flatten())
    
    def __repr__(self) -> str:
        return f"{self.tag}: [{self.composition}]"

    @property
    def R_specific(self):
        R_arr = jnp.stack([jnp.atleast_1d(elem.R_specific).squeeze() for elem in self.composition.elements], axis=-1)

        R_mixed = jnp.sum(R_arr * self.composition.mass_fractions, axis=-1)
        return R_mixed

    def compute_Cp(self, T: float | jnp.ndarray = 298.15):
        Cp_arr = jnp.stack([elem.compute_Cp(T) for elem in self.composition.elements], axis=-1)

        # Mass-weighted sum
        Cp_mixed = jnp.sum(Cp_arr * self.composition.mass_fractions, axis=-1)
        return Cp_mixed

    def compute_absolute_enthalpy(self, T: float | jnp.ndarray = 298.15):
        h_arr = jnp.stack([elem.compute_absolute_enthalpy(T) for elem in self.composition.elements], axis=-1)

        # Mass-weighted sum
        h_mixed = jnp.sum(h_arr * self.composition.mass_fractions, axis=-1)
        return h_mixed
    
    def compute_enthalpy(self, T: float | jnp.ndarray = 298.15):
        h_arr = jnp.stack([jnp.atleast_1d(elem.compute_enthalpy(T)).squeeze() for elem in self.composition.elements], axis=-1)

        # Mass-weighted sum
        h_mixed = jnp.sum(h_arr * self.composition.mass_fractions, axis=-1)
        return h_mixed

    def compute_partial_pressures(self, P: float | jnp.ndarray = 101325.0):
        return P * jnp.asarray(self.composition.mole_fractions)[None, :]

    def compute_entropy(self, T: float | jnp.ndarray = 298.15, P: float | jnp.ndarray = 101325.0):
        X_arr = jnp.stack(self.composition.mole_fractions, axis=-1)
        S_arr = jnp.stack(
            [elem.compute_entropy(T, P * X_arr[..., i]) for i, elem in enumerate(self.composition.elements)], axis=-1
        )

        # Mass-weighted sum
        S_mixed = jnp.sum(S_arr * self.composition.mass_fractions, axis=-1)
        return S_mixed


@lru_cache(maxsize=1)
def _build_air():
    """Private builder for standard air."""
    return MixedGas(
        tag="Air",
        composition=GasComposition(
            elements=("O2", "AR", "CO2", "N2"), mass_fractions=jnp.asarray([0.2314, 0.0128, 0.0006, 0.7552])
        ),
    )


_CUSTOM_MIXTURES = {
    "Air": _build_air,
}


def Air():
    return MixedGas(
        tag="Air",
        composition=GasComposition(
            elements=("O2", "Ar", "CO2", "N2"),
            mass_fractions=jnp.asarray([0.2314, 0.0128, 0.0006, 0.7552]),
        ),
    )


def burned_JetA_composition(FAR: float | jnp.ndarray) -> GasComposition:
    """
    Dynamically generates the combustion product mass fractions for Jet-A.
    Assumes complete lean combustion.
    """

    FAR_arr = jnp.atleast_2d(FAR)

    # Base air masses (kg) per 1 kg of inlet air
    m_O2_air = 0.2314
    m_N2_air = 0.7552
    m_Ar_air = 0.0128
    m_CO2_air = 0.0006

    # Jet-A reaction mass balances (kg per 1 kg of fuel burned)
    O2_consumed = 3.396
    CO2_produced = 3.155
    H2O_produced = 1.242

    # Calculate new absolute masses
    # Using jnp.maximum prevents negative mass during aggressive solver iterations
    m_O2 = jnp.maximum(m_O2_air - (O2_consumed * FAR_arr), 0.0)
    m_CO2 = m_CO2_air + (CO2_produced * FAR_arr)
    m_H2O = H2O_produced * FAR_arr
    m_N2 = m_N2_air  # Inert
    m_Ar = m_Ar_air  # Inert

    m_total = 1.0 + FAR

    # Return the dynamically mixed composition
    return GasComposition(
        elements=("O2", "AR", "CO2", "N2", "H2O"),
        mass_fractions=jnp.concatenate(
            [
                jnp.atleast_2d(m_O2 / m_total),
                jnp.atleast_2d(m_Ar / m_total),
                jnp.atleast_2d(m_CO2 / m_total),
                jnp.atleast_2d(m_N2 / m_total),
                jnp.atleast_2d(m_H2O / m_total),
            ],
            axis=-1,
        ),
    )

def BurnedJetA(FAR: float | jnp.ndarray) -> MixedGas:
    return MixedGas(tag="Burned Jet-A", composition=burned_JetA_composition(FAR))


# ----------------------------------------------------------------------------------------------------------------------
#  CHEMKIN Harvester
# ----------------------------------------------------------------------------------------------------------------------

# Standard atomic weights in g/mol
ATOMIC_MASSES = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "AR": 39.948, "S": 32.065}


def parse_chemkin_thermo(filepath: str, output_path: str):
    database = {}

    with open(filepath, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].replace("\n", "")

        # Skip header lines, blank lines, or comments (usually start with !)
        if not line or line.startswith("!") or line.startswith("THERMO"):
            i += 1
            continue

        # The line ending in '1' (at column 79) is the species header
        if len(line) >= 80 and line[79] == "1":
            # 1. Parse the Header Line (Strict Column Slicing)
            species_name = line[0:18].strip()

            # Extract Atomic Composition (Four 5-character blocks)
            composition = {}
            for col in range(24, 44, 5):
                element = line[col : col + 2].strip().upper()  # .upper() for safety
                count_str = line[col + 2 : col + 5].strip()
                if element and count_str:
                    composition[element] = int(count_str)

            # Calculate Molecular Weight
            try:
                molecular_mass = sum(ATOMIC_MASSES[elem] * count for elem, count in composition.items())
            except KeyError as e:
                print(f"Warning: Missing atomic weight for element {e} in species {species_name}. Defaulting to 0.0")
                molecular_mass = 0.0

            phase = line[44].strip()

            # Temperature bounds
            t_low = float(line[45:55].strip())
            t_high = float(line[55:65].strip())
            t_mid = float(line[65:75].strip())

            # 2. Parse the Coefficients (Next 3 lines)
            line2 = lines[i + 1]
            line3 = lines[i + 2]
            line4 = lines[i + 3]

            def extract(l, start):
                return float(l[start : start + 15].strip())

            # Line 2: First 5 High-Temp Coeffs
            h1, h2, h3, h4, h5 = [extract(line2, j) for j in range(0, 75, 15)]

            # Line 3: Last 2 High-Temp Coeffs, First 3 Low-Temp Coeffs
            h6, h7, l1, l2, l3 = [extract(line3, j) for j in range(0, 75, 15)]

            # Line 4: Last 4 Low-Temp Coeffs
            l4, l5, l6, l7 = [extract(line4, j) for j in range(0, 60, 15)]

            # 3. Store in Dictionary
            database[species_name] = {
                "phase": phase,
                "composition": composition,
                "molecular_mass": molecular_mass,
                "T_low": t_low,
                "T_high": t_high,
                "T_mid": t_mid,
                "nasa_high_coeffs": [h1, h2, h3, h4, h5, h6, h7],
                "nasa_low_coeffs": [l1, l2, l3, l4, l5, l6, l7],
            }

            i += 4  # Skip past the 4 lines we just parsed
        else:
            i += 1

    with open(output_path, "w") as f:
        json.dump(database, f, indent=4)

    print(f"Successfully harvested {len(database)} species into {output_path}")


# ----------------------------------------------------------------------------------------------------------------------
#  Thermo Database
# ----------------------------------------------------------------------------------------------------------------------

_DB_PATH = get_Trace_root() / "library/data/thermo_database.json"


@lru_cache(maxsize=1)
def _load_database():
    """Loads the entire JSON database into memory on the first call."""
    if not _DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {_DB_PATH}")
    with open(_DB_PATH, "r") as f:
        return json.load(f)


@lru_cache(maxsize=None)
def _get_gas(name: str):
    """Fetches the gas from the cached database and builds the Equinox module."""
    db = _load_database()

    if name not in db:
        raise AttributeError(f"Species '{name}' not found in the thermo database.")

    data = db[name]

    # Construct and return your Equinox IdealGas module
    return IdealGas(
        tag=name,
        molecular_mass=data["molecular_mass"] * units.gram,
        nasa_low_coeffs=tuple(data["nasa_low_coeffs"]),
        nasa_high_coeffs=tuple(data["nasa_high_coeffs"]),
    )


def __getattr__(name: str):
    """Intercepts module-level attribute access."""
    if name.startswith("_"):
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    # First, check if the user is asking for a custom mixture
    if name in _CUSTOM_MIXTURES:
        return _CUSTOM_MIXTURES[name]()

    # If not, fall back to the thermodynamic database
    return _get_gas(name)


def __dir__():
    """Allows IDEs to see both JSON species and custom mixtures."""
    available_species = []
    if _DB_PATH.exists():
        available_species.extend(_load_database().keys())

    available_species.extend(_CUSTOM_MIXTURES.keys())
    return available_species
