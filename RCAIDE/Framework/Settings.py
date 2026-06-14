# RCAIDE/Framework/Settings.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Jul 2024, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from typing import Optional, Literal

import jax
import equinox as eqx

# package imports
from jaxopt import ScipyRootFinding, Broyden, GaussNewton

# RCAIDE imports
from RCAIDE.utils import init_field
from RCAIDE.Framework import GradientMap

# ----------------------------------------------------------------------------------------------------------------------
#  Settings
# ----------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------
#  Analysis Settings
# ---------------------------------------------------------

# Mass Analysis

class ReductionFactors(eqx.Module):

    main_wing:  float = 0.0
    fuselage:   float = 0.0
    empennage:  float = 0.0
    systems:    float = 0.0

class SizingFractions(eqx.Module):
    rudder_sizing: float = 0.25

class MassAnalysisSettings(eqx.Module):

    reduction_factors: ReductionFactors = init_field(ReductionFactors)

class AnalysisSettings(eqx.Module):

    aerodynamics: Optional[eqx.Module] = None
    mass: Optional[MassAnalysisSettings] = init_field(MassAnalysisSettings)

    gradient_map: Optional[GradientMap] = init_field(None, static=True)

# ----------------------------------------------------------------------------------------------------------------------
#  Mission Settings
# ----------------------------------------------------------------------------------------------------------------------


RootFinders = Literal["ScipyRootFinding", "Broyden", "GaussNewton"]

class MissionSettings(eqx.Module):

    root_finder: RootFinders = init_field("GaussNewton", static=True)

# ----------------------------------------------------------------------------------------------------------------------
#  Full Settings
# ----------------------------------------------------------------------------------------------------------------------

class Settings(eqx.Module):

    tag:                str                 = init_field('Settings', static=True)

    analysis:           AnalysisSettings    = init_field(AnalysisSettings)
    mission:            MissionSettings     = init_field(MissionSettings)

    DEBUG_MODE:         bool                = init_field(False, static=True)
    verbose:            bool                = init_field(False, static=True)
    JAX_device_index:   int                 = init_field(0, static=True)


    def __post_init__(self):
        if self.DEBUG_MODE:
            jax.config.update("jax_disable_jit", True)
            jax.config.update("jax_debug_nans", True)
            object.__setattr__(self, "verbose", True)
        else:
            # Manually reset flags
            jax.config.update("jax_disable_jit", False)
            jax.config.update("jax_debug_nans", False)

