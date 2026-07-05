# RCAIDE/__init__.py
# (c) Copyright 2023 Aerospace Research Community LLC

"""RCAIDE Package Setup"""

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from .utils import initialize_jax_cache

initialize_jax_cache(
    cache_dir="~/.rcaide/jax_cache",
    max_size_gb=2.0,
    max_age_days=30,
)

