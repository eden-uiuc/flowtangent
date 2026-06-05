# RCAIDE/Framework/Analyses/Aerodynamics/VLM.py
# (c) Copyright 2026 Aerospace Research Community LLC
#
# Created: May 2025, RCAIDE Team
# Modified: Mar 2026, J. Smart

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Optional, Iterable, Any, Literal

if TYPE_CHECKING:
    from RCAIDE.Framework import State, System, Settings

import warnings
from pathlib import Path
from itertools import product
from collections import defaultdict

# package imports
import jax
import jax.numpy as jnp
import equinox as eqx

import numpy as np  # Strictly for database serialization

import sklearn
import zarr

from tqdm import trange

try:
    import pynvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False

# RCAIDE imports
import RCAIDE.utils as ru

from RCAIDE.Library import Units
from RCAIDE.Library.Methods.Aerodynamics.Transonic import peaked_CL_spline, ensemble_CL_spline

from RCAIDE.Framework import State, Process, ProcessStep, Aircraft
from RCAIDE.Framework.Missions.Conditions import Numerics

from RCAIDE.Framework.Methods.Aerodynamics.VORJAX import (check_freestream,
                                                          compute_coefficients, compute_induced_velocity,
                                                          compute_panel_pressures, compute_boundary_conditions,
                                                          compute_vortex_strength,
                                                          initialize_VLM_data, discretize_surfaces,
                                                          apply_aerodynamic_forces)

from RCAIDE.Framework.Methods.Aerodynamics import (expand_component_coefficients,
                                                   compute_parasite_drag,
                                                   compute_viscous_induced_drag)

# ----------------------------------------------------------------------------------------------------------------------
#  VLM Settings
# ----------------------------------------------------------------------------------------------------------------------


class SupersonicSettings(eqx.Module):
    
    begin_blend_mach:               float = 0.5
    end_blend_mach:                 float = 2.0
    
    peak_CL_multiplier:             float = 1.15
    peak_mach_number:               Optional[float] = None
    _transonic_CL_blender:          Callable = eqx.field(static=True, default=ensemble_CL_spline)
    
    begin_drag_rise_mach_number:    float = 0.95
    end_drag_rise_mach_number:      float = 1.2
    
    transonic_drag_multiplier:      float = 1.25
    volume_wave_drag_scaling:       float = 3.2
    
    cross_section_type:             str  = eqx.field(static=True, default='Fixed')
    wave_drag_type:                 str  = eqx.field(static=True, default='Raymer')

    def __post_init__(self):
        if self.peak_mach_number is not None:
            object.__setattr__(self, "_transonic_CL_blender", peaked_CL_spline)
    
    def transonic_CL_blender(self, M, val_sub, val_sup):
        return self._transonic_CL_blender(
            M, 
            self.begin_blend_mach,
            self.peak_mach_number,
            self.end_blend_mach,
            val_sub, val_sup,
            peak_multiplier=self.peak_CL_multiplier
            )

class CorrectionFactors(eqx.Module):

    suction:    bool = eqx.field(static=True, default=True)
    shock:      bool = eqx.field(static=True, default=True)

    fuselage_lift: float = 1.14
    trim_drag: float = 1.02

    viscous_lift_drag: float = 0.38
    lift_to_drag: float = 0.0
    CL_max: float = 1.0


class FormFactors(eqx.Module):

    span_efficiency: float = 1.0
    oswald: float = 1.0

    wing: float = 1.1
    fuselage: float = 2.3
    pylon: float = 0.2


class Surrogate(eqx.Module):

    surrogate:              Optional[Any] = eqx.field(static=True, default_factory=sklearn.gaussian_process.GaussianProcessRegressor)
    
    blend_transonic:        bool          = True

    angle_of_attack:        jnp.ndarray  = eqx.field(default_factory=lambda: jnp.linspace(-5., 15., 40) * Units.deg)
    sideslip_angle:         jnp.ndarray  = eqx.field(default_factory=lambda: jnp.linspace(0.0, 15., 30) * Units.deg)
    Mach:                   jnp.ndarray  = eqx.field(default_factory=lambda: jnp.linspace(0., 0.85, 20))
    
    aileron_deflection:     jnp.ndarray  = eqx.field(default_factory=lambda: jnp.array([30, 10.0, 1E-12]) * Units.deg)
    elevator_deflection:    jnp.ndarray  = eqx.field(default_factory=lambda: jnp.array([30, 10.0, 1E-12]) * Units.deg)
    rudder_deflection:      jnp.ndarray  = eqx.field(default_factory=lambda: jnp.array([30, 10.0, 1E-12]) * Units.deg)
    flap_deflection:        jnp.ndarray  = eqx.field(default_factory=lambda: jnp.array([30, 10.0, 1E-12]) * Units.deg)
    slat_deflection:        jnp.ndarray  = eqx.field(default_factory=lambda: jnp.array([30, 10.0, 1E-12]) * Units.deg)

    u:                      jnp.ndarray  = eqx.field(default_factory=lambda: jnp.array([0.2, 0.1, 1E-12]))
    v:                      jnp.ndarray  = eqx.field(default_factory=lambda: jnp.array([0.2, 0.1, 1E-12]))
    w:                      jnp.ndarray  = eqx.field(default_factory=lambda: jnp.array([0.2, 0.1, 1E-12]))

    pitch_rate:             jnp.ndarray  = eqx.field(default_factory=lambda:jnp.array([0.3, 0.15, 0.0])  * Units.rad / Units.s)
    roll_rate:              jnp.ndarray  = eqx.field(default_factory=lambda:jnp.array([0.3, 0.15, 0.0])  * Units.rad / Units.s)
    yaw_rate:               jnp.ndarray  = eqx.field(default_factory=lambda:jnp.array([0.3, 0.15, 0.0])  * Units.rad / Units.s)

    def fit(self, *args, **kwargs):
        return self.surrogate.fit(*args, **kwargs)
    
    def predict(self, *args, **kwargs):
        return self.surrogate.predict(*args, **kwargs)


class Vortices(eqx.Module):

    model_fuselage:             bool = eqx.field(static=True, default=False)
    verbose:                    bool = eqx.field(static=True, default=False)
    
    # Discretization Inputs (Optional, so the user can choose which to define)
    spanwise_cosine:    bool = eqx.field(static=True, default=True)
    chordwise_cosine:   bool = eqx.field(static=True, default=False) # Currently unsupported

    n_spanwise:         Optional[Iterable[int] | int] = eqx.field(static=True, default=8)  # Min value is number of wing segments (possibly more for control surfaces)
    n_chordwise:        Optional[Iterable[int] | int] = eqx.field(static=True, default=3)  # Min value 3 to allow front and rear control surfaces
    
    # Can set separate values for each wing/fuselage (ex. [8, 4] for [wing, stab] and [4, 2] for [fuselage, nacelle]), else uses global value above
    wings_n_spanwise:   Optional[Iterable[int] | int] = eqx.field(static=True, default=None)
    wings_n_chordwise:  Optional[Iterable[int] | int] = eqx.field(static=True, default=None)

    bodies_n_spanwise:  Optional[Iterable[int] | int] = eqx.field(static=True, default=None)
    bodies_n_chordwise: Optional[Iterable[int] | int] = eqx.field(static=True, default=None)

    def __post_init__(self):
        """Validates discretization inputs and resolves global vs separate routing."""

        if self.chordwise_cosine:
            warnings.warn(f"Chordwise cosine spacing is currently unsupported. Defaulting to linear spacing.")
            object.__setattr__(self, "chordwise_cosine", False)
        
        # Check if the user explicitly provided separate definitions
        separate_provided = any([
            self.wings_n_spanwise is not None,
            self.wings_n_chordwise is not None,
            self.bodies_n_spanwise is not None,
            self.bodies_n_chordwise is not None
        ])

        if separate_provided:
            # Validate that all separate variables were provided
            missing_separate = any(x is None for x in [
                self.wings_n_spanwise, self.wings_n_chordwise,
                self.bodies_n_spanwise, self.bodies_n_chordwise
            ])
            if missing_separate:
                raise ValueError('If using separate surface discretization, all n_sw and n_cw values must be defined.')

        else:
            # User didn't provide separate settings, so we fallback to the global defaults
            if not self.n_spanwise or not self.n_chordwise:
                raise ValueError('If using global surface discretization, both n_sw and n_cw must be defined.')

            # Route the global settings to the specific component fields
            object.__setattr__(self, 'wings_n_spanwise', self.n_spanwise)
            object.__setattr__(self, 'wings_n_chordwise', self.n_chordwise)
            object.__setattr__(self, 'bodies_n_spanwise', self.n_spanwise)
            object.__setattr__(self, 'bodies_n_chordwise', self.n_chordwise)


class VLMSettings(eqx.Module):

    model_fuselage:             bool    = eqx.field(static=True, default=False)
    trim_aircraft:              bool    = eqx.field(static=True, default=False)

    recalculate_wetted_area:    bool    = eqx.field(static=True, default=False)
    model_propeller_wake:       bool    = eqx.field(static=True, default=False)
    near_field_drag:            bool    = eqx.field(static=True, default=False)

    CL_max:                     float   = jnp.inf
    CD_increment:               float   = 0.0
    spoiler_drag_increment:     float   = 0.0

    # Sub-Settings

    vortices:       Vortices                = eqx.field(default_factory=Vortices)

    supersonic:     SupersonicSettings      = eqx.field(default_factory=SupersonicSettings)
    corrections:    CorrectionFactors       = eqx.field(default_factory=CorrectionFactors)
    form_factors:   FormFactors             = eqx.field(default_factory=FormFactors)
    surrogate:      Surrogate               = eqx.field(default_factory=Surrogate)

# ----------------------------------------------------------------------------------------------------------------------
#  VLM Initialization
# ----------------------------------------------------------------------------------------------------------------------

def _default_VORJAX_init_steps():
    return(
        ProcessStep(expand_component_coefficients, "Initialize Component Bookkeeping"),
        ProcessStep(initialize_VLM_data, "Initialize Data Structures"),
        ProcessStep(discretize_surfaces, "Discretize Surfaces"),
    )

class InitializeVORJAX(Process):
    
    tag: str = eqx.field(static=True, default="Initialize VORJAX")
    steps: tuple = eqx.field(default_factory=_default_VORJAX_init_steps)

# ----------------------------------------------------------
#  VORJAX Compute Process
# ----------------------------------------------------------

def _default_VORJAX_compute_steps():
    return(
        # Lift and Induced Drag
        ProcessStep(check_freestream, "Freestream Validation"),
        ProcessStep(compute_boundary_conditions, "Calculate Boundary Conditions"),
        ProcessStep(compute_induced_velocity, "Calculate VICs"),
        ProcessStep(compute_vortex_strength, "Compute Vortex Strength"),
        ProcessStep(compute_panel_pressures, "Compute Pressure Coefficients"),
        ProcessStep(compute_coefficients, "Compute Aerodynamic Coefficients"),
        ProcessStep(apply_aerodynamic_forces, "Apply Aerodynamic Forces"),
    )


class ComputeVORJAX(Process):

    tag: str = eqx.field(static=True, default="Compute VORJAX")

    steps: tuple = eqx.field(default_factory=_default_VORJAX_compute_steps)


class VORJAX(Process):
    tag: str = eqx.field(static=True, default="Aerodynamics")
    steps: tuple = eqx.field(default_factory=lambda: (InitializeVORJAX(), ComputeVORJAX()))

    # TODO: Add full drag, trimming, stability analysis


#-----------------------------------------------------------
# Batch VORJAX Analysis
#-----------------------------------------------------------

class BatchVORJAX:

    def __init__(self):
        # Path mapping and default settings.
        self._INPUT_MAPPINGS = {
            "mach":         (ru.PathTuple(("freestream", "mach_number")), [0.0]),
            "alpha":        (ru.PathTuple(("aerodynamics", "angles", "alpha")), [0.0]),
            "beta":         (ru.PathTuple(("aerodynamics", "angles", "beta")), [0.0]),
            "roll_rate":    (ru.PathTuple(("stability", "static", "roll_rate")), [0.0]),
            "pitch_rate":   (ru.PathTuple(("stability", "static", "pitch_rate")), [0.0]),
            "yaw_rate":     (ru.PathTuple(("stability", "static", "yaw_rate")), [0.0]),
            "density":      (ru.PathTuple(("freestream", "density")), [1.225]),
            "gamma":        (ru.PathTuple(("freestream", "gamma")), [1.4]),
            "temperature":  (ru.PathTuple(("freestream", "temperature")), [288.15]),
        }

        self._OUTPUT_MAPPINGS = {
            "CL":           ru.PathTuple(("aerodynamics", "coefficients", "lift", "total")),
            "CD":           ru.PathTuple(("aerodynamics", "coefficients", "drag", "total")),
            "CX":           ru.PathTuple(("aerodynamics", "coefficients", "X",)),
            "CY":           ru.PathTuple(("aerodynamics", "coefficients", "Y",)),
            "CZ":           ru.PathTuple(("aerodynamics", "coefficients", "Z",)),
            "C_l":          ru.PathTuple(("aerodynamics", "coefficients", "moments", "roll")),
            "C_m":          ru.PathTuple(("aerodynamics", "coefficients", "moments", "pitch")),
            "C_n":          ru.PathTuple(("aerodynamics", "coefficients", "moments", "yaw")),
        }

        self._compute_process = ComputeVORJAX()
        self._compiled_step = eqx.filter_jit(self._compute_process.run)

    def run(
        self,
        system: Aircraft,
        settings: Settings,
        mode="zip",
        batch_size: Optional[int]=None,
        db_path: Optional[str | Path] = None,
        **kwargs
    ):

        # Set up base state
        state       = State(numerics=Numerics(number_of_control_points=1, calculate_integration=False))
        initials    = eqx.tree_at(lambda s: s.initials, state, None, is_leaf=lambda x: x is None)
        base_state  = eqx.tree_at(lambda s: s.initials, state, initials, is_leaf=lambda x: x is None)

        active_keys = []
        target_map = []
        raw_arrays = []

        # Validate inputs, convert to JAX arrays
        for k, v in kwargs.items():
            if k.lower() not in self._INPUT_MAPPINGS:
                warnings.warn(f"Unrecognized variable {k} ignored. "
                              f"Allowed variables: {list(self._INPUT_MAPPINGS.keys())}")
            else:
                active_keys.append(k.lower())
                target_map.append(self._INPUT_MAPPINGS[k.lower()][0])
                raw_arrays.append(jnp.atleast_1d(v))

        if len(active_keys) == 0:
            raise ValueError("No valid inputs provided.")
        for k, v in self._INPUT_MAPPINGS.items():
            if k not in active_keys:
                active_keys.append(k)
                target_map.append(v[0])
                raw_arrays.append(jnp.atleast_1d(v[1]))

        # Get all flight states
        if mode == "zip":
            processed_arrays = jnp.broadcast_arrays(*raw_arrays)
        elif mode == "mesh":
            grids = jnp.meshgrid(*raw_arrays, indexing="ij")
            processed_arrays = [g.ravel().reshape(-1, 1) for g in grids]
        else:
            raise ValueError(f"Invalid mode {mode}. Supported modes: 'zip', 'mesh'.")

        # Set batch size if not provided
        total_states = len(processed_arrays[0])
        if settings.verbose:
            print(f"Total states for VORJAX analysis: {total_states}.")


        if batch_size is None:
            n_s = settings.analysis.aerodynamics.vortices.n_spanwise
            n_c = settings.analysis.aerodynamics.vortices.n_chordwise
            n_panels = n_s * n_c

            # 1 kB per AIC coefficient is a rough multiplier estimate for peak memory footprint
            bytes_per_state = 1024 * n_panels ** 2

            if HAS_NVML:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(settings.JAX_device_index)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_vram_bytes = info.total
                pynvml.nvmlShutdown()
            else:
                warnings.warn("VORJAX batch size undefined, but unable to check GPU memory. "
                              "Please install pynvml for GPU memory monitoring or set batch size manually. "
                              "Defaulting to 8 GB assumption.", category=UserWarning)
                total_vram_bytes = 8 * 1024 ** 3  # Assuming 8 GB of VRAM

            # Calculate batch size
            jax_usable_vram = total_vram_bytes * 0.90  # JAX defaults to 90% pre-allocation
            target_vram = jax_usable_vram * 0.75  # Allocate 75% of usable VRAM for batch
            max_batch_size = int(target_vram // bytes_per_state)
            hw_batch_size = 2 ** int(jnp.log2(max_batch_size))

            if total_states <= hw_batch_size:
                batch_size = int(2 ** jnp.ceil(jnp.log2(total_states)))
            else:
                batch_size = hw_batch_size
            if settings.verbose:
                print(f"Optimal batch size for VORJAX analysis: {batch_size}.")

        # Prepare database if provided
        if db_path is not None:
            
            db_root = zarr.open_group(db_path, mode='a')
            
            def create_db_key(key: str, v_np):
                shape = (0,) + v_np.shape[1:]
                chunk = (batch_size,) + v_np.shape[1:]

                db_root.create_array(
                    name=key,
                    shape=shape,
                    chunks=chunk,
                    dtype=v_np.dtype
                )


        all_coeffs = {k: [] for k in self._OUTPUT_MAPPINGS.keys()}
        jac_arr = None

        # Initialize VORJAX once
        init_results = InitializeVORJAX().run(base_state.expand_rows(batch_size), system, settings)
        state = init_results[0]
        system = init_results[1]
        settings = init_results[2]

        # Batch over computation
        for i in trange(0, total_states, batch_size, desc="Running VORJAX Analysis"):
            batch_arrays = tuple(arr[i:i+batch_size].reshape(-1, 1) for arr in processed_arrays)
            actual_size  = len(batch_arrays[0])

            if actual_size < batch_size:
                pad_length = ((0, batch_size - actual_size), (0, 0))
                batch_arrays = tuple(jnp.pad(arr, pad_length, mode="edge") for arr in batch_arrays)

            batch_state = eqx.tree_at(lambda s: ru.get_all_targets(s, target_map), state, batch_arrays)
            res = self._compiled_step(batch_state, system, settings)

            coeff_arrs = ru.get_all_targets(res[0], self._OUTPUT_MAPPINGS.values())
            coeff_arrs = jax.tree.map(lambda x: x[:actual_size], coeff_arrs)

            for j, key in enumerate(self._OUTPUT_MAPPINGS.keys()):
                v_np = np.asarray(coeff_arrs[j])                    # Convert to numpy array for serialization
                if db_path is not None:
                    if key not in db_root: create_db_key(key, v_np) # Handle new keys
                    db_root[key].append(v_np, axis=0)               # Append to existing key

                all_coeffs[key].append(v_np)
            
            if settings.analysis.gradient_map is not None:
                g_map = settings.analysis.gradient_map
                jac_arr : jnp.ndarray = res[3]  #type: ignore
                all_grads = defaultdict(list)
                for out_i, output in enumerate(g_map.state_outputs):
                    for inp_i, input in enumerate(g_map.state_inputs):
                        key = f"d{output.tag}_d{input.tag}"
                        v_np = np.asarray(jac_arr[:actual_size, out_i, inp_i])         # Convert to numpy array for serialization
                        if db_path is not None:
                            if key not in db_root: create_db_key(key, v_np) # Handle new keys
                            db_root[key].append(v_np, axis=0)               # Append to existing key
                        
                        all_grads[key].append(v_np)


        if jac_arr is not None:
            return_val = (all_coeffs | all_grads,)
        else:
            return_val = (all_coeffs,)
        if db_path is not None:
            return_val += (db_root,)

        return return_val


# ----------------------------------------------------------
#  Surrogate VLM Process
# ----------------------------------------------------------

# TODO: Surrogate VLM initialization, steps, and analysis