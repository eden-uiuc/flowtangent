# src/eden_trace/utils/backend.py
## NO 3RD PARTY PACKAGE IMPORTS - SETS UP JAX/EQUINOX FLAGS AND MUST RUN FIRST

import os
import sys
import time

def numerical_environment():
    """Configures hardware and NUMA affinity before JAX initializes."""
    os.environ["JAX_ENABLE_X64"] = "True"
    os.environ['OPENMDAO_REPORTS'] = '0'
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    os.environ["JAX_PERSISTENT_CACHE_DISABLE"] = "1"
    os.environ["JAX_PLATFORM_NAME"] = "gpu"
    
    if sys.platform == "linux":
        cpu_count = os.cpu_count() or 1
        if cpu_count > 16:
            try:
                node_0_cores = set(range(16))
                os.sched_setaffinity(0, node_0_cores)
                os.environ["OMP_PROC_BIND"] = "true"
                os.environ["OMP_PLACES"] = "cores"
                print(f"Hardware Config: NUMA affinity set to Node 0 (16 cores).")
            except Exception as e:
                print(f"Hardware Config Warning: Could not set CPU affinity: {e}")

    cache_path = os.path.expanduser("~/.eden_trace/jax_cache")
    os.makedirs(cache_path, exist_ok=True)
    os.environ["JAX_COMPILATION_CACHE_DIR"] = cache_path

def initialize_jax_cache(cache_dir="~/.eden_trace/jax_cache", max_size_gb=2.0, max_age_days=30):
    import jax  # Lazy import safe here
    cache_path = os.path.expanduser(cache_dir)
    os.makedirs(cache_path, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", cache_path)
    try:
        _prune_cache(cache_path, max_size_gb, max_age_days)
    except Exception as e:
        print(f"Trace Warning: Failed to prune JAX compilation cache - {e}")

def _prune_cache(cache_path, max_size_gb, max_age_days):
    # (Paste your exact _prune_cache logic here from the old utils.py)
    pass 

def configure_environment(settings):
    """Configures global JAX and XLA compiler flags at runtime."""
    import jax  # Lazy import safe here
    if settings._DEV_MODE:
        os.environ["JAX_LOGGING_LEVEL"] = "DEBUG"
        os.environ["JAX_DEBUG_LOG_MODULES"] = "jax._src.compiler, jax._src.lru_cache"
        os.environ["JAX_EXPLAIN_CACHE_MISSES"] = "1"
        
    if settings.DEBUG_MODE:
        jax.config.update("jax_disable_jit", True)
        jax.config.update("jax_debug_nans", True)
        print("TRACE WARNING: Debug mode is active. JIT disabled and NaN debugging enabled.")
    else:
        jax.config.update("jax_disable_jit", False)
        jax.config.update("jax_debug_nans", False)