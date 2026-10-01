# flowtangent/Framework/Analyses/__init__.py
# (c) Copyright 2023 Aerospace Research Community LLC

"""FlowTangent Package Setup"""

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from ._batched import BatchedAnalysis
from ._implicit import ImplicitAnalysis
from .energy import PACTAnalysis

from . import energy
from . import aero
from . import _mass as mass

from ._settings import (
    NumericalSettings,
    JacobianMap,
    JacobianSettings,
    AnalysisSettings,
    EnergyAnalysisSettings,
    MassAnalysisSettings,
)


from .aero._vorjax import VORJAX, VORJAXSettings

__all__ = [
    # Analysis Types
    "BatchedAnalysis",
    "ImplicitAnalysis",
    "PACTAnalysis",
    # Specific Analyses
    "VORJAX",
    # Settings
    "NumericalSettings",
    "JacobianMap",
    "JacobianSettings",
    "AnalysisSettings",
    "EnergyAnalysisSettings",
    "MassAnalysisSettings",
    "VORJAXSettings",
    # Submodules
    "energy",
    "aero",
    "mass",
]
