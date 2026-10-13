# src/flowtangent/utils/backend.py
## NO 3RD PARTY PACKAGE IMPORTS - SETS UP JAX/EQUINOX FLAGS AND MUST RUN FIRST

import contextlib
import os
import time


def numerical_environment():
    """Set numerical environment defaults before JAX initialization.

    Enable 64-bit JAX arithmetic, disable GPU memory preallocation, and suppress
    TensorFlow C++ messages below level 3. Set the persistent compilation cache
    directory to `~/.flowtangent/jax_cache`, creating it if necessary.

    ### Process Architecture
    Overwrite the environment flags, create the cache directory, and publish its path.
    Call before importing JAX or Equinox for the flags to take effect.

    ### Usage Example
    ```python
    import flowtangent
    # Importing FlowTangent calls numerical_environment().
    ```

    Returns:
        None: Updates process environment variables and may create directories.

    Raises:
        OSError: If the cache directory cannot be created.
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
    r"""Delete expired cache files and trim remaining files by age.

    Only regular files directly inside the cache directory count toward the limits;
    subdirectories are ignored. Age and removal order use modification time, not
    access time. A missing cache path is a no-op.

    ### Mathematical Formulation
    Delete files with $t_{now}-t_{mtime}>86400\,d$. Then remove the oldest remaining
    files while their total size exceeds $g\,1024^3$ bytes.

    ### Process Architecture
    Remove expired files first, then remove retained files oldest first to meet the size limit.

    ### Usage Example
    ```python
    import tempfile
    from pathlib import Path
    from flowtangent.utils.backend import _prune_cache

    with tempfile.TemporaryDirectory() as cache:
        Path(cache, "entry").write_bytes(b"compiled")
        _prune_cache(cache, max_size_gb=0, max_age_days=30)
    ```

    Args:
        cache_path (str | os.PathLike): Directory to prune; no path expansion is performed.
        max_size_gb (float): Size limit in GiB (`1024**3` bytes). Zero removes
            positive-size files; zero-byte files may remain.
        max_age_days (int | float): Modification-age limit in days. Files exactly
            at the threshold are retained unless removed to meet the size limit.

    Returns:
        None: Deletes qualifying files from disk. Files disappearing during
            metadata lookup or removal are ignored.

    Raises:
        OSError: If listing the directory fails, or metadata lookup or deletion
            fails for a reason other than a file disappearing. `initialize_jax_cache`
            catches pruning errors; direct callers receive them.
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
