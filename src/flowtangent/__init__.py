# flowtangent/__init__.py
# (c) Copyright 2023 Aerospace Research Community LLC

"""Flowtangent Package Setup"""

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

# 1. Early Boot (Must happen first)
from .utils.backend import numerical_environment, initialize_jax_cache

numerical_environment()
initialize_jax_cache()

# Framework Hoists
from .core._state import State
from .core._systems import System, Aircraft
from .core._component import Component
from .core._processes import Process, ProcessStep, array_barrier

# Utility Hoists
from .utils import (
    update,
    TreePath,
    ArraySlice,
    field,
    static_field,
    method_field,
    null_step,
    Module,
    configure_environment,
)

from .data import units

from .solve import (
    BatchedAnalysis,
    ImplicitAnalysis,
    PACTAnalysis,
    JacobianMap,
    NumericalSettings,
    AnalysisSettings,
)

from .components import (Wing, Fuselage)

# 4. Short-Name Namespace Routing
from . import functional as F  # noqa: N812
from . import components as comp
from . import data
from . import solve as solve
from . import sim as sim
from . import utils as utils
from . import plots as plots

from .core._settings import Settings

__all__ = [
    # FlowTangent Classes
    "Module",
    "State",
    "System",
    "Settings",
    "Aircraft",
    "Settings",
    "Component",
    "Process",
    "ProcessStep",
    "TreePath",
    "ArraySlice",
    "JacobianMap",
    "NumericalSettings",
    "AnalysisSettings",
    # Key utilities
    "update",
    "field",
    "static_field",
    "method_field",
    "null_step",
    "units",
    "array_barrier",
    "configure_environment",
    # Analyses
    "BatchedAnalysis",
    "ImplicitAnalysis",
    "PACTAnalysis",
    # Components
    "Wing",
    "Fuselage",
    # Submodules
    "F",
    "comp",
    "data",
    "solve",
    "sim",
    "utils",
    "plots",
]
