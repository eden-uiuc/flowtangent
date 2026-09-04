# flowtangent/__init__.py
# (c) Copyright 2023 Aerospace Research Community LLC

"""Flowtangent Package Setup"""

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from .utils.backend import initialize_jax_cache

initialize_jax_cache(
    cache_dir="~/.flowtangent/jax_cache",
    max_size_gb=2.0,
    max_age_days=30,
)

# 1. Early Boot (Must happen first)
from .utils.backend import numerical_environment, initialize_jax_cache
numerical_environment()

# Framework Hoists
from .core.settings import Settings
from .core.state_container import State, System
from .core.base_component import Component, Node

# Utility Hoists
from .utils import (
    update,         
    TreePath,       
    field,          
    static_field,   
    method_field,   
    null_step,
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