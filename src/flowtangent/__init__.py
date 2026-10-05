# flowtangent/__init__.py
# (c) Copyright 2023 Aerospace Research Community LLC

"""FlowTangent Package Setup"""

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

# Data Hoists
from .data import units
# from .components.airfoils import (_data as airfoils, load_foil as load_airfoil)
from .components.energy.maps import (_data as turbo_maps, load_map as load_turbo_map)
from .data import gases

# Analysis hoists
from .solve import (
    BatchedAnalysis,
    ImplicitAnalysis,
    PACTAnalysis,
    JacobianMap,
    NumericalSettings,
    AnalysisSettings,
)

from .components import Wing, Fuselage

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
    "array_barrier",
    "configure_environment"
    # Data sources
    "units",
    "airfoils",
    "load_airfoil",
    "turbo_maps",
    "load_turbo_map",
    "gases",
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
