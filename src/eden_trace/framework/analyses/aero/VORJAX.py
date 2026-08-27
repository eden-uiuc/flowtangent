# Trace/Framework/Analyses/Aerodynamics/VLM.py
# (c) Copyright 2026 Aerospace Research Community LLC
#
# Created: May 2025, Trace Team
# Modified: Mar 2026, J. Smart

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional

if TYPE_CHECKING:
    pass

import warnings
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp

# package imports
import sklearn

# Trace imports
from eden_trace.utils import DataPath, init_field

from eden_trace.library import units
from eden_trace.library.methods.aero.Transonic import ensemble_CL_spline, peaked_CL_spline

from eden_trace.framework import Process, ProcessStep
from eden_trace.framework.analyses import BatchedAnalysis
from eden_trace.framework.methods.aero.VORJAX import (
    apply_aerodynamic_forces,
    check_freestream,
    compute_boundary_conditions,
    compute_coefficients,
    compute_induced_velocity,
    compute_panel_pressures,
    compute_vortex_strength,
    discretize_surfaces,
    initialize_VORJAX_data,
)
from eden_trace.framework.simulation.initialize import initialize_aerodynamics

# ----------------------------------------------------------------------------------------------------------------------
#  VLM Settings
# ----------------------------------------------------------------------------------------------------------------------


class SupersonicSettings(eqx.Module):
    begin_blend_mach: float = 0.5
    end_blend_mach: float = 2.0

    peak_CL_multiplier: float = 1.15
    peak_mach_number: Optional[float] = None
    _transonic_CL_blender: Callable = init_field(ensemble_CL_spline, as_value=True, static=True)

    begin_drag_rise_mach_number: float = 0.95
    end_drag_rise_mach_number: float = 1.2

    transonic_drag_multiplier: float = 1.25
    volume_wave_drag_scaling: float = 3.2

    cross_section_type: str = init_field("Fixed", static=True)
    wave_drag_type: str = init_field("Raymer", static=True)

    def __post_init__(self):
        if self.peak_mach_number is not None:
            object.__setattr__(self, "_transonic_CL_blender", init_field(peaked_CL_spline, as_value=True, static=True))

    def transonic_CL_blender(self, M, val_sub, val_sup):
        return self._transonic_CL_blender(
            M,
            self.begin_blend_mach,
            self.peak_mach_number,
            self.end_blend_mach,
            val_sub,
            val_sup,
            peak_multiplier=self.peak_CL_multiplier,
        )


class CorrectionFactors(eqx.Module):
    suction: bool = init_field(True, static=True)
    shock: bool = init_field(True, static=True)

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
    surrogate: Optional[Any] = init_field(sklearn.gaussian_process.GaussianProcessRegressor, static=True)

    blend_transonic: bool = True

    angle_of_attack: jnp.ndarray = init_field(lambda: jnp.linspace(-5.0, 15.0, 40) * units.deg)
    sideslip_angle: jnp.ndarray = init_field(lambda: jnp.linspace(0.0, 15.0, 30) * units.deg)
    mach: jnp.ndarray = init_field(lambda: jnp.linspace(0.0, 0.85, 20))

    aileron_deflection: jnp.ndarray = init_field(lambda: jnp.array([30, 10.0, 1e-12]) * units.deg)
    elevator_deflection: jnp.ndarray = init_field(lambda: jnp.array([30, 10.0, 1e-12]) * units.deg)
    rudder_deflection: jnp.ndarray = init_field(lambda: jnp.array([30, 10.0, 1e-12]) * units.deg)
    flap_deflection: jnp.ndarray = init_field(lambda: jnp.array([30, 10.0, 1e-12]) * units.deg)
    slat_deflection: jnp.ndarray = init_field(lambda: jnp.array([30, 10.0, 1e-12]) * units.deg)

    u: jnp.ndarray = init_field(lambda: jnp.array([0.2, 0.1, 1e-12]))
    v: jnp.ndarray = init_field(lambda: jnp.array([0.2, 0.1, 1e-12]))
    w: jnp.ndarray = init_field(lambda: jnp.array([0.2, 0.1, 1e-12]))

    pitch_rate: jnp.ndarray = init_field(lambda: jnp.array([0.3, 0.15, 0.0]) * units.rad / units.s)
    roll_rate: jnp.ndarray = init_field(lambda: jnp.array([0.3, 0.15, 0.0]) * units.rad / units.s)
    yaw_rate: jnp.ndarray = init_field(lambda: jnp.array([0.3, 0.15, 0.0]) * units.rad / units.s)

    def fit(self, *args, **kwargs):
        return self.surrogate.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.surrogate.predict(*args, **kwargs)


class Vortices(eqx.Module):
    model_fuselage: bool = init_field(False, static=True)
    verbose: bool = init_field(False, static=True)

    # Discretization Inputs (Optional, so the user can choose which to define)
    spanwise_cosine: bool = init_field(True, static=True)
    chordwise_cosine: bool = init_field(False, static=True)  # Currently unsupported

    n_spanwise: Optional[Iterable[int] | int] = init_field(
        8, static=True
    )  # Min value is number of wing segments (possibly more for control surfaces)
    n_chordwise: Optional[Iterable[int] | int] = init_field(
        3, static=True
    )  # Min value 3 to allow front and rear control surfaces

    # Can set separate values for each wing/fuselage (ex. [8, 4] for [wing, stab] and [4, 2] for [fuselage, nacelle]), else uses global value above
    wings_n_spanwise: Optional[Iterable[int] | int] = init_field(None, static=True)
    wings_n_chordwise: Optional[Iterable[int] | int] = init_field(None, static=True)

    bodies_n_spanwise: Optional[Iterable[int] | int] = init_field(None, static=True)
    bodies_n_chordwise: Optional[Iterable[int] | int] = init_field(None, static=True)

    def __post_init__(self):
        """Validates discretization inputs and resolves global vs separate routing."""

        if self.chordwise_cosine:
            warnings.warn("Chordwise cosine spacing is currently unsupported. Defaulting to linear spacing.")
            object.__setattr__(self, "chordwise_cosine", False)

        # Check if the user explicitly provided separate definitions
        separate_provided = any(
            [
                self.wings_n_spanwise is not None,
                self.wings_n_chordwise is not None,
                self.bodies_n_spanwise is not None,
                self.bodies_n_chordwise is not None,
            ]
        )

        if separate_provided:
            # Validate that all separate variables were provided
            missing_separate = any(
                x is None
                for x in [
                    self.wings_n_spanwise,
                    self.wings_n_chordwise,
                    self.bodies_n_spanwise,
                    self.bodies_n_chordwise,
                ]
            )
            if missing_separate:
                raise ValueError("If using separate surface discretization, all n_sw and n_cw values must be defined.")

        else:
            # User didn't provide separate settings, so we fallback to the global defaults
            if not self.n_spanwise or not self.n_chordwise:
                raise ValueError("If using global surface discretization, both n_sw and n_cw must be defined.")

            # Route the global settings to the specific component fields
            object.__setattr__(self, "wings_n_spanwise", self.n_spanwise)
            object.__setattr__(self, "wings_n_chordwise", self.n_chordwise)
            object.__setattr__(self, "bodies_n_spanwise", self.n_spanwise)
            object.__setattr__(self, "bodies_n_chordwise", self.n_chordwise)


class VORJAX_Settings(eqx.Module):
    model_fuselage: bool = init_field(False, static=True)
    trim_aircraft: bool = init_field(False, static=True)

    recalculate_wetted_area: bool = init_field(False, static=True)
    model_propeller_wake: bool = init_field(False, static=True)
    near_field_drag: bool = init_field(False, static=True)

    CL_max: float = jnp.inf
    CD_increment: float = 0.0
    spoiler_drag_increment: float = 0.0

    # Sub-Settings

    vortices: Vortices = init_field(Vortices)

    supersonic: SupersonicSettings = init_field(SupersonicSettings)
    corrections: CorrectionFactors = init_field(CorrectionFactors)
    form_factors: FormFactors = init_field(FormFactors)
    surrogate: Surrogate = init_field(Surrogate)


# ----------------------------------------------------------------------------------------------------------------------
#  VLM Initialization
# ----------------------------------------------------------------------------------------------------------------------


def _default_VORJAX_init_steps():
    return (
        ProcessStep(function=initialize_aerodynamics, tag="Initialize Component Bookkeeping"),
        ProcessStep(function=initialize_VORJAX_data, tag="Initialize Data Structures"),
        ProcessStep(function=discretize_surfaces, tag="Discretize Surfaces"),
    )


class InitializeVORJAX(Process):
    tag: str = init_field("Initialize VORJAX", static=True)
    steps: tuple = init_field(_default_VORJAX_init_steps)


# ----------------------------------------------------------
#  VORJAX Compute Process
# ----------------------------------------------------------


def _default_VORJAX_compute_steps():
    return (
        # Lift and Induced Drag
        ProcessStep(function=check_freestream, tag="Freestream Validation"),
        ProcessStep(function=compute_boundary_conditions, tag="Calculate Boundary Conditions"),
        ProcessStep(function=compute_induced_velocity, tag="Calculate VICs"),
        ProcessStep(function=compute_vortex_strength, tag="Compute Vortex Strength"),
        ProcessStep(function=compute_panel_pressures, tag="Compute Pressure Coefficients"),
        ProcessStep(function=compute_coefficients, tag="Compute Aerodynamic Coefficients"),
        ProcessStep(function=apply_aerodynamic_forces, tag="Apply Aerodynamic Forces"),
    )


class ComputeVORJAX(Process):
    tag: str = init_field("Compute VORJAX", static=True)

    steps: tuple = init_field(_default_VORJAX_compute_steps)


class VORJAX(Process):
    tag: str = init_field("Aerodynamics", static=True)
    steps: tuple = init_field(lambda: (InitializeVORJAX(), ComputeVORJAX()))

    # TODO: Add full drag, trimming, stability analysis


# -----------------------------------------------------------
# Batched VORJAX Analysis
# -----------------------------------------------------------

VORJAX_Inputs = {
    "mach": (DataPath(("freestream", "mach_number")), [0.0]),
    "alpha": (DataPath(("aerodynamics", "angles", "alpha")), [0.0]),
    "beta": (DataPath(("aerodynamics", "angles", "beta")), [0.0]),
    "roll_rate": (DataPath(("stability", "static", "roll_rate")), [0.0]),
    "pitch_rate": (DataPath(("stability", "static", "pitch_rate")), [0.0]),
    "yaw_rate": (DataPath(("stability", "static", "yaw_rate")), [0.0]),
    "density": (DataPath(("freestream", "density")), [1.225]),
    "gamma": (DataPath(("freestream", "gamma")), [1.4]),
    "temperature": (DataPath(("freestream", "temperature")), [288.15]),
}

VORJAX_Outputs = {
    "CL": DataPath(("aerodynamics", "coefficients", "lift", "total")),
    "CD": DataPath(("aerodynamics", "coefficients", "drag", "total")),
    "CX": DataPath(
        (
            "aerodynamics",
            "coefficients",
            "X",
        )
    ),
    "CY": DataPath(
        (
            "aerodynamics",
            "coefficients",
            "Y",
        )
    ),
    "CZ": DataPath(
        (
            "aerodynamics",
            "coefficients",
            "Z",
        )
    ),
    "C_l": DataPath(("aerodynamics", "coefficients", "moments", "roll")),
    "C_m": DataPath(("aerodynamics", "coefficients", "moments", "pitch")),
    "C_n": DataPath(("aerodynamics", "coefficients", "moments", "yaw")),
}


class BatchVORJAX(BatchedAnalysis):
    def __init__(
        self,
        tag: str = "Batched VORJAX",
        initialize: Process = InitializeVORJAX(),
        compute: Process = ComputeVORJAX(),
        inputs: dict = VORJAX_Inputs,
        outputs: dict = VORJAX_Outputs,
        db_path: str | Path | None = None,
    ):
        super().__init__(tag, initialize, compute, inputs, outputs, db_path)


if __name__ == "__main__":
    print(*[VORJAX_Inputs[i][0] for i in ["alpha", "mach"]])

# ----------------------------------------------------------
#  Surrogate VLM Process
# ----------------------------------------------------------

# TODO: Surrogate VLM initialization, steps, and analysis
