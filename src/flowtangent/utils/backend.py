# src/flowtangent/utils/backend.py
## NO 3RD PARTY PACKAGE IMPORTS - SETS UP JAX/EQUINOX FLAGS AND MUST RUN FIRST

import os
import time
import contextlib


def numerical_environment():
    """Configures JAX and TensorFlow environment variables before JAX initializes.

    This function must be called before importing JAX, Equinox, or any
    hardware-accelerated libraries. It enforces 64-bit precision, prevents
    XLA from preallocating all available GPU memory, and silences standard
    TensorFlow C++ warnings.
    """
    os.environ["JAX_ENABLE_X64"] = "True"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    cache_path = os.path.expanduser("~/.flowtangent/jax_cache")
    os.makedirs(cache_path, exist_ok=True)
    os.environ["JAX_COMPILATION_CACHE_DIR"] = cache_path


def initialize_jax_cache(cache_dir="~/.flowtangent/jax_cache", max_size_gb=2.0, max_age_days=30):
    """Initializes and prunes the persistent JAX compilation cache.

    Setting up a compilation cache significantly reduces startup times for
    frequently run graphs. This function ensures the cache directory exists,
    binds it to the JAX configuration, and triggers a maintenance pass to
    clean up stale or oversized cache files.

    Args:
        cache_dir (str): The local directory path for storing compiled XLA binaries.
            Defaults to `~/.flowtangent/jax_cache`.
        max_size_gb (float): The maximum allowed size of the cache directory in
            gigabytes. Defaults to `2.0`.
        max_age_days (int): The maximum number of days to retain an untouched
            cache file. Defaults to `30`.
    """
    import jax  # Lazy import safe here

    cache_path = os.path.expanduser(cache_dir)
    os.makedirs(cache_path, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", cache_path)
    try:
        _prune_cache(cache_path, max_size_gb, max_age_days)
    except Exception as e:
        print(f"FlowTangent Warning: Failed to prune JAX compilation cache - {e}")


def _prune_cache(cache_path, max_size_gb, max_age_days):
    """Enforces size and age limits on the compilation cache directory.

    Args:
        cache_path (str): The expanded absolute path to the cache directory.
        max_size_gb (float): Target maximum directory size in gigabytes.
        max_age_days (int): File age limit in days.
    """
    if not os.path.exists(cache_path):
        return

    current_time = time.time()
    max_age_seconds = max_age_days * 86400
    max_size_bytes = max_size_gb * (1024**3)

    total_size = 0
    files = []

    # First pass: collect files and immediately remove expired ones
    for filename in os.listdir(cache_path):
        filepath = os.path.join(cache_path, filename)
        if not os.path.isfile(filepath):
            continue

        try:
            file_stat = os.stat(filepath)
        except FileNotFoundError:
            continue
        file_age = current_time - file_stat.st_mtime

        if file_age > max_age_seconds:
            with contextlib.suppress(FileNotFoundError):
                os.remove(filepath)
        else:
            files.append((filepath, file_stat.st_mtime, file_stat.st_size))
            total_size += file_stat.st_size

    # Second pass: if total size exceeds limit, remove oldest files until compliant
    if total_size > max_size_bytes:
        # Sort by modification time (oldest first)
        files.sort(key=lambda x: x[1])
        for filepath, _, size in files:
            with contextlib.suppress(FileNotFoundError):
                os.remove(filepath)
            total_size -= size
            if total_size <= max_size_bytes:
                break


def configure_environment(settings):
    """Configures global JAX and XLA compiler flags at runtime.

    Modifies the active JAX session based on the provided settings object.
    This allows switching to debug modes, enabling NaN detection, or viewing
    compiler trace logs without restarting the Python interpreter.

    Args:
        settings: The global configuration object containing framework flags
            (must provide `_DEV_MODE` and `DEBUG_MODE` attributes).
    """
    import jax  # Lazy import safe here

    if settings._DEV_MODE:
        os.environ["JAX_LOGGING_LEVEL"] = "DEBUG"
        os.environ["JAX_DEBUG_LOG_MODULES"] = "jax._src.compiler, jax._src.lru_cache"
        os.environ["JAX_EXPLAIN_CACHE_MISSES"] = "1"

    if settings.DEBUG_MODE:
        jax.config.update("jax_disable_jit", True)
        jax.config.update("jax_debug_nans", True)
        print("FLOWTANGENT WARNING: Debug mode is active. JIT disabled and NaN debugging enabled.")
    else:
        jax.config.update("jax_disable_jit", False)
        jax.config.update("jax_debug_nans", False)
