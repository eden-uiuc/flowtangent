# RCAIDE/Framework/Settings.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Jul 2024, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from typing import Optional

import jax
import equinox as eqx

# package imports
from jaxopt import ScipyRootFinding, Broyden, GaussNewton

# RCAIDE imports
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

    reduction_factors: ReductionFactors = eqx.field(default_factory=ReductionFactors)

class AnalysisSettings(eqx.Module):

    aerodynamics: Optional[eqx.Module] = None
    mass: Optional[MassAnalysisSettings] = eqx.field(default_factory=MassAnalysisSettings)

    gradient_map: Optional[GradientMap] = eqx.field(static=True, default=None)

# ----------------------------------------------------------------------------------------------------------------------
#  Mission Settings
# ----------------------------------------------------------------------------------------------------------------------


RootFinders = ScipyRootFinding | Broyden | GaussNewton

class MissionSettings(eqx.Module):

    root_finder:    RootFinders         = eqx.field(static=True, default=GaussNewton)

# ----------------------------------------------------------------------------------------------------------------------
#  Full Settings
# ----------------------------------------------------------------------------------------------------------------------

class Settings(eqx.Module):

    tag:                str                 = eqx.field(static=True, default='Settings')

    analysis:           AnalysisSettings    = eqx.field(default_factory=AnalysisSettings)
    mission:            MissionSettings     = eqx.field(default_factory=MissionSettings)

    DEBUG_MODE:         bool                = eqx.field(static=True, default=False)
    verbose:            bool                = eqx.field(static=True, default=False)
    JAX_device_index:   int                 = eqx.field(static=True, default=0)


    def __post_init__(self):
        if self.DEBUG_MODE:
            jax.config.update("jax_disable_jit", True)
            jax.config.update("jax_debug_nans", True)
            object.__setattr__(self, "verbose", True)
        else:
            # Manually reset flags
            jax.config.update("jax_disable_jit", False)
            jax.config.update("jax_debug_nans", False)

