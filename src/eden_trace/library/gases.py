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
from typing import Optional

# package imports
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from eden_trace.utils import get_trace_root, register

# Trace imports
from eden_trace.library import units

# ----------------------------------------------------------------------------------------------------------------------
#  Thermo Database
# ----------------------------------------------------------------------------------------------------------------------

_DB_PATH = get_trace_root() / "library/data/thermo_database.json"

@lru_cache(maxsize=1)
def _load_database():
    """Loads the entire JSON database into memory on the first call."""
    if not _DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {_DB_PATH}")
    with open(_DB_PATH, "r") as f:
        return json.load(f)

db = _load_database()
SPECIES_LIST = list(db.keys())
SPECIES_INDEX = {name:idx for idx, name in enumerate(SPECIES_LIST)}

_MOL_LIST = []
_LOW_LIST = []
_MID_LIST = []
_HIGH_LIST = []

for name in SPECIES_LIST:
    data = db[name]
    _MOL_LIST.append(data["molecular_mass"])
    _LOW_LIST.append(data["nasa_low_coeffs"])
    _MID_LIST.append(data["T_mid"])
    _HIGH_LIST.append(data["nasa_high_coeffs"])

MOL_MASS    = jnp.array(_MOL_LIST, dtype=jnp.float64) * units.gram
NASA_LOW    = jnp.array(_LOW_LIST, dtype=jnp.float64)
NASA_MID    = jnp.array(_MID_LIST, dtype=jnp.float64)
NASA_HIGH   = jnp.array(_HIGH_LIST, dtype=jnp.float64)

R_UNIV = 8.314462
R_SPEC = R_UNIV / MOL_MASS

def _eval_Cp(T):
    T_arr = jnp.asarray(T)
    if T_arr.ndim > 1 and T_arr.shape[-1] == 1:
        T_arr = T_arr[..., 0]
        
    T_vec = jnp.stack([jnp.ones_like(T_arr), T_arr, T_arr**2, T_arr**3, T_arr**4], axis=-1)

    cp_low  = R_SPEC * jnp.dot(T_vec, NASA_LOW[:, :5].T)
    cp_high = R_SPEC * jnp.dot(T_vec, NASA_HIGH[:, :5].T)

    return jnp.where(jnp.expand_dims(T_arr, axis=-1) > NASA_MID, cp_high, cp_low)

def _eval_h(T):
    """
    Evaluates absolute enthalpy for all species simultaneously.
    NASA_LOW/HIGH shape: (N_species, 7)
    """
    # Build the T-multiplier vector for enthalpy [T, T^2/2, T^3/3, T^4/4, T^5/5, 1, 0]
    T_arr = jnp.asarray(T)
    if T_arr.ndim > 1 and T_arr.shape[-1] == 1:
        T_arr = T_arr[..., 0]

    T_vec = jnp.stack([
        T_arr, 
        (T_arr**2) / 2.0, 
        (T_arr**3) / 3.0, 
        (T_arr**4) / 4.0, 
        (T_arr**5) / 5.0, 
        jnp.ones_like(T_arr), 
        jnp.zeros_like(T_arr)
    ], axis=-1) # Shape (7, T.shape)
    
    # Dot product across the coefficients for low and high temperature ranges
    # T_vec @ COEFFS yields a (N_species, T.shape) array
    h_low   = R_SPEC * jnp.dot(T_vec, NASA_LOW.T)
    h_high  = R_SPEC * jnp.dot(T_vec, NASA_HIGH.T)
    
    return jnp.where(jnp.expand_dims(T_arr, axis=-1) > NASA_MID, h_high, h_low)

def _eval_s0(T):
    T_arr = jnp.asarray(T)
    if T_arr.ndim > 1 and T_arr.shape[-1] == 1:
        T_arr = T_arr[..., 0]

    T_vec = jnp.stack([
        jnp.log(T_arr),
        T_arr,
        (T_arr**2) / 2.0,
        (T_arr**3) / 3.0,
        (T_arr**4) / 4.0,
        jnp.zeros_like(T_arr),
        jnp.ones_like(T_arr)
    ], axis=-1)
    
    # FIX: Use T_vec @ NASA.T so shapes perfectly align to (..., N_species)
    s_low   = R_SPEC * jnp.dot(T_vec, NASA_LOW.T)
    s_high  = R_SPEC * jnp.dot(T_vec, NASA_HIGH.T)
    
    return jnp.where(jnp.expand_dims(T_arr, axis=-1) > NASA_MID, s_high, s_low)

@register
class Gas(eqx.Module):
    mass_fractions: jax.Array

    def __init__(self, mass_fractions:Optional[jax.Array]=None, fractions_dict: Optional[dict] = None):
        # If explicitly given an array (used by pure species and JAX internals)
        if mass_fractions is not None:
            self.mass_fractions = jnp.asarray(mass_fractions, dtype=jnp.float64)
            
        # If given a dictionary
        elif fractions_dict is not None:
            fractions = np.zeros(len(SPECIES_LIST), dtype=np.float64)
            for name, val in fractions_dict.items():
                fractions[SPECIES_INDEX[name]] = val
            self.mass_fractions = jnp.asarray(fractions)
            
        # Fallback to empty gas
        else:
            self.mass_fractions = jnp.zeros(len(SPECIES_LIST), dtype=jnp.float64)

    def __repr__(self) -> str:
        if isinstance(self.mass_fractions, jax.core.Tracer):
            return "Gas(Traced Composition)"

        # Convert to standard numpy. 
        # This is fast and prevents JAX from trying to trace the formatting logic.
        mf = np.asarray(self.mass_fractions)

        # Filter out numerical noise (anything below 0.0001%)
        active_indices = np.where(mf > 1e-6)[-1]

        if len(active_indices) == 0:
            return "Gas(Empty)"

        comp_str = "; ".join(f"{SPECIES_LIST[i]}: {mf[i]:.1%}" for i in active_indices)
        
        return f"Gas({comp_str})"

    @property
    def mole_fractions(self):
        inv_mm = 1.0 / MOL_MASS
        mm_mix = 1.0 / jnp.sum(self.mass_fractions * inv_mm, axis=-1, keepdims=True)
        return self.mass_fractions * mm_mix * inv_mm

    @property
    def R_specific(self):
        return jnp.sum(self.mass_fractions * R_SPEC, axis=-1, keepdims=True)

    def compute_density(self, T: float | jnp.ndarray = 298.15, P: float | jnp.ndarray = 101325.0):
        return P / (self.R_specific * T)

    def compute_Cp(self, T: float | jax.Array = 298.15):
        cp_all = _eval_Cp(T)  
        return jnp.sum(self.mass_fractions * cp_all, axis=-1, keepdims=True)

    def compute_absolute_enthalpy(self, T: float | jax.Array = 298.15):
        h_all = _eval_h(T)    
        return jnp.sum(self.mass_fractions * h_all, axis=-1, keepdims=True)

    def compute_enthalpy(self, T: float | jax.Array = 298.15):
        h_abs = self.compute_absolute_enthalpy(T)
        h_ref = self.compute_absolute_enthalpy(298.15)
        return h_abs - h_ref

    def invert_enthalpy(self, h_target: float | jnp.ndarray, T_guess: float | jnp.ndarray=1000.0, max_iter: int = 5):
        def step(T, _):
            h = self.compute_enthalpy(T)
            cp = self.compute_Cp(T)
            
            # Because h, cp, and h_target are all guaranteed to be shaped (-1, 1), 
            # this calculation is 100% immune to broadcasting mismatches.
            T = T - (h - h_target) / cp
            return T, None
    
        T_guess_arr = jnp.asarray(T_guess).reshape((-1, 1))
        T_final, _ = jax.lax.scan(step, T_guess_arr, jnp.arange(max_iter))
        return T_final

    def compute_entropy(self, T: float | jax.Array = 298.15, P: float | jax.Array = 101325.0):
        s0_all = _eval_s0(T) 
        s0_mixed = jnp.sum(self.mass_fractions * s0_all, axis=-1, keepdims=True)
        
        P_ref = 101325.0
        return s0_mixed - self.R_specific * jnp.log(P / P_ref)

    def compute_gamma(self, T: float | jnp.ndarray = 298.15):
        cp = self.compute_Cp(T)
        gamma = cp / (cp - self.R_specific)
        return gamma

    def compute_speed_of_sound(self, T: float | jnp.ndarray = 298.0):
        g = self.compute_gamma(T)
        return jnp.sqrt(g * self.R_specific * T)

    def compute_absolute_viscosity(self, T: float | jnp.ndarray = 298.0):
        return 1.8e-5

@lru_cache(maxsize=None)
def _get_gas(name: str):
    """Fetches the gas from the cached database and builds the Equinox module."""

    if name not in SPECIES_INDEX:
            raise AttributeError(f"Species '{name}' not found in the thermo database.")

    f_dict = {name: 1.0}
    return Gas(fractions_dict=f_dict)


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
    available_species.extend(SPECIES_LIST)
    available_species.extend(_CUSTOM_MIXTURES.keys())
    return available_species

# ----------------------------------------------------------------------------------------------------------------------
#  Custom Mixes
# ----------------------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _build_air():
    """Private builder for standard air."""
    air =  Gas(
        fractions_dict={
            "O2":0.2314,
            "AR":0.0128,
            "CO2":0.0006,
            "N2":0.7552
        },
    )
    return air

_CUSTOM_MIXTURES = {
    "Air": _build_air,
}

def BurnedJetA(FAR: float | jax.Array) -> Gas:
    """
    Dynamically generates combustion product mass fractions for Jet-A.
    Supports scalars, 1D, 2D (Batch, Time), or higher-dimensional FAR inputs.
    """
    # Ensure it's at least a JAX array
    FAR_arr = jnp.atleast_1d(FAR)

    if FAR_arr.ndim > 1 and FAR_arr.shape[-1] == 1:
        FAR_arr = FAR_arr[..., 0]

    m_O2_air = 0.2314
    m_N2_air = 0.7552
    m_Ar_air = 0.0128
    m_CO2_air = 0.0006

    O2_consumed = 3.396
    CO2_produced = 3.155
    H2O_produced = 1.242

    m_total = 1.0 + FAR_arr

    target_shape = FAR_arr.shape + (len(SPECIES_LIST),)
    fractions = jnp.zeros(target_shape, dtype=jnp.float64)

    # 4. Populate using Ellipsis (...) to handle ANY number of batch dimensions!
    fractions = fractions.at[..., SPECIES_INDEX["O2"]].set(
        jnp.maximum(m_O2_air - (O2_consumed * FAR_arr), 0.0) / m_total
    )
    fractions = fractions.at[..., SPECIES_INDEX["CO2"]].set(
        (m_CO2_air + (CO2_produced * FAR_arr)) / m_total
    )
    fractions = fractions.at[..., SPECIES_INDEX["H2O"]].set(
        (H2O_produced * FAR_arr) / m_total
    )
    fractions = fractions.at[..., SPECIES_INDEX["AR"]].set(
        m_Ar_air / m_total
    )
    fractions = fractions.at[..., SPECIES_INDEX["N2"]].set(
        m_N2_air / m_total
    )

    return Gas(mass_fractions=fractions)

# ----------------------------------------------------------------------------------------------------------------------
#  CHEMKIN Harvester
# ----------------------------------------------------------------------------------------------------------------------

# Standard atomic weights in g/mol
ATOMIC_MASSES = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "AR": 39.948,
    "S": 32.065 
}

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