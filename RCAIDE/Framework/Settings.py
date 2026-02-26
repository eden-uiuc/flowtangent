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


class AnalysisSettings(eqx.Module):

    aerodynamics: eqx.Module | None = None


class Settings(eqx.Module):

    tag: str                    = eqx.field(static=True, default='Settings')
    # Mission Settings
    root_finder: Callable       = eqx.field(static=True, default=ScipyRootFinding)

    analysis: AnalysisSettings  = eqx.field(default_factory=AnalysisSettings)

