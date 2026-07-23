# Trace/Framework/Settings.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Jul 2024, Trace Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from typing import Optional, Literal

import os

# package imports
import equinox as eqx
import jax

# Trace imports
from eden_trace.utils import init_field
from eden_trace.framework import GradientMap

# ----------------------------------------------------------------------------------------------------------------------
#  Settings
# ----------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#  Analysis Settings
# ------------------------------------------------------------------------------

# Mass Analysis ----------------------------------------------------------------


class ReductionFactors(eqx.Module):
    main_wing: float = 0.0
    fuselage: float = 0.0
    empennage: float = 0.0
    systems: float = 0.0


class SizingFractions(eqx.Module):
    rudder_sizing: float = 0.25


class MassAnalysisSettings(eqx.Module):
    reduction_factors: ReductionFactors = init_field(ReductionFactors)
    sizing_fractions: SizingFractions = init_field(SizingFractions)

# Energy Analysis --------------------------------------------------------------

class EnergyAnalysisSettings(eqx.Module):
    report_units: Literal["SI", "Imperial"] = init_field("SI", static=True)


class AnalysisSettings(eqx.Module):
    aerodynamics: Optional[eqx.Module] = None
    energy: EnergyAnalysisSettings = init_field(EnergyAnalysisSettings)
    mass: MassAnalysisSettings = init_field(MassAnalysisSettings)

    gradient_map: Optional[GradientMap] = init_field(None, static=True)


# ----------------------------------------------------------------------------------------------------------------------
#  Mission Settings
# ----------------------------------------------------------------------------------------------------------------------

class NumericalSettings(eqx.Module):
    
    relative_tolerance: float = init_field(1e-5, static=True)
    absolute_tolerance: float = init_field(1e-5, static=True)
    
    max_evaluations: int = init_field(30, static=True)
    step_size: float | None = init_field(None, static=True)

    number_of_control_points: int = init_field(1, static=True)
    maximum_graph_complexity: int = init_field(2e5, static=True)


# ----------------------------------------------------------------------------------------------------------------------
#  Full Settings
# ----------------------------------------------------------------------------------------------------------------------


class Settings(eqx.Module):
    tag: str = init_field("Settings", static=True)

    analysis: AnalysisSettings = init_field(AnalysisSettings)
    numerical: NumericalSettings = init_field(NumericalSettings)

    DEBUG_MODE: bool = init_field(False, static=True)
    verbose: bool = init_field(False, static=True)
    JAX_device_index: int = init_field(0, static=True)

    _DEV_MODE: bool = init_field(False, static=True)

    def __post_init__(self):
        if self._DEV_MODE:
            os.environ["XLA_FLAGS"] = "--xla_backend_optimization_level=0"
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        if self.DEBUG_MODE:
            jax.config.update("jax_disable_jit", True)
            jax.config.update("jax_debug_nans", True)
            object.__setattr__(self, "verbose", True)
        else:
            # Manually reset flags
            jax.config.update("jax_disable_jit", False)
            jax.config.update("jax_debug_nans", False)
