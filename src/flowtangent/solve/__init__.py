"""FlowTangent Package Setup"""

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from ._batched import BatchedAnalysis, ShardedDatasetGenerator
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
    "ShardedDatasetGenerator",
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
