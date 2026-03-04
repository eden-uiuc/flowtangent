# RCAIDE/Framework/Settings.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Jul 2024, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

import equinox as eqx
from typing import Callable

# package imports
from jaxopt import ScipyRootFinding

# RCAIDE imports

# ----------------------------------------------------------------------------------------------------------------------
#  Settings
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
#  Analysis Settings
# ----------------------------------------------------------------------------------------------------------------------

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

    aerodynamics: eqx.Module | None = None
    mass: MassAnalysisSettings = eqx.field(default_factory=MassAnalysisSettings)

# ----------------------------------------------------------------------------------------------------------------------
#  Mission Settings
# ----------------------------------------------------------------------------------------------------------------------

class MissionSettings(eqx.Module):

    verbose:    bool = eqx.field(static=True, default=False)
    debugging:  bool = eqx.field(static=True, default=False)

# ----------------------------------------------------------------------------------------------------------------------
#  Full Settings
# ----------------------------------------------------------------------------------------------------------------------

class Settings(eqx.Module):

    tag: str                            = eqx.field(static=True, default='Settings')
    root_finder:    Callable            = eqx.field(static=True, default=ScipyRootFinding)

    analysis:       AnalysisSettings    = eqx.field(default_factory=AnalysisSettings)
    mission:        MissionSettings     = eqx.field(default_factory=MissionSettings)

