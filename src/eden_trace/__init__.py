# Trace/__init__.py
# (c) Copyright 2023 Aerospace Research Community LLC

"""Trace Package Setup"""

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from .utils.backend import initialize_jax_cache

initialize_jax_cache(
    cache_dir="~/.eden_trace/jax_cache",
    max_size_gb=2.0,
    max_age_days=30,
)

# 1. Early Boot (Must happen first)
from .utils.backend import numerical_environment, initialize_jax_cache
numerical_environment()

# 2. Core Framework Hoists
from .core.settings import Settings
from .core.state_container import State, System
from .core.base_component import Component, Node

# 3. Critical Utility Hoists (The 80% Rule applied to utils)
from .utils import (
    update,          # ft.update(obj, ...)
    TreePath,        # ft.TreePath(...)
    field,           # ft.field(...)
    static_field,    # ft.static_field(...)
)

# 4. Short-Name Namespace Routing
from . import functional as F
from . import components as comp
from . import environment as env
from . import datasets as data
from . import analyses as solve
from . import simulation as sim
from . import state
from . import utils