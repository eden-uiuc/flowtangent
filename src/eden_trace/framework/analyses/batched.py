# RCAIDE/Framework/Analyses/Batched.py
# (c) Copyright 2026 Aerospace Research Community LLC
#
# Created: Jun 2026, J. Smart
# Modified: Jun 2026, J. Smart

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional, Tuple

if TYPE_CHECKING:
    from src.eden_trace.framework import State, System, Settings

import logging
import os
import shutil
import tempfile
import time
from collections import defaultdict
from itertools import product
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import zarr
from numcodecs import Blosc
from tqdm import tqdm, trange

import src.eden_trace.utils as ru

from src.eden_trace.framework import Process, State


# ----------------------------------------------------------------------------------------------------------------------
#  Batch Analysis
# ----------------------------------------------------------------------------------------------------------------------
class BatchAnalysis:
    def __init__(
        self,
        tag: str = "Batched Analysis",
        initialize: Process = Process(),
        compute: Process = Process(),
        inputs: dict = {},
        outputs: dict = {},
        db_path: Optional[str | Path] = None,
    ):

        self.tag = tag

        # Path mapping and default settings.
        self.input_mappings = inputs
        self.output_mappings = outputs

        self.initialization_process = initialize
        self.compute_process = compute
        self._compiled_step = eqx.filter_jit(self.compute_process.run)

        self.db_path = db_path

    def run(
        self,
        system: System,
        settings: Settings,
        mode="zip",
        batch_size: Optional[int] = None,
        handle: Optional[str] = None,
        **kwargs,
    ):

        if handle is not None:  # Inherit logger from dataset generator
            logger = logging.getLogger(handle)
        else:  # Self logging
            logger = logging.getLogger(self.tag + "_Logger")

            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)

            formatter = logging.Formatter("[%(asctime)s] - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
            ch.setFormatter(formatter)
            logger.addHandler(ch)

        # Set up base state
        from src.eden_trace.framework.conditions import Time

        state = State(time=Time(number_of_control_points=1, calculate_integration=False))
        initials = eqx.tree_at(lambda s: s.initials, state, None, is_leaf=lambda x: x is None)
        base_state = eqx.tree_at(lambda s: s.initials, state, initials, is_leaf=lambda x: x is None)

        input_keys = []
        active_keys = []
        target_map = []
        raw_arrays = []

        # Validate inputs, convert to JAX arrays
        all_outputs = {k: [] for k in self.output_mappings.keys()}
        all_grads = defaultdict(list)
        all_inputs = defaultdict(list)
        jac_arr = None

        for k, v in kwargs.items():
            if k.lower() not in self.input_mappings:
                logger.warning(
                    f"Unrecognized variable {k} ignored. Allowed variables: {list(self.input_mappings.keys())}"
                )
            else:
                input_keys.append(k.lower())
                active_keys.append(k.lower())
                target_map.append(self.input_mappings[k.lower()][0])
                raw_arrays.append(jnp.atleast_1d(v))

        if len(active_keys) == 0:
            raise ValueError("No valid inputs provided.")
        for k, v in self.input_mappings.items():
            if k not in active_keys:
                active_keys.append(k)
                target_map.append(v[0])
                raw_arrays.append(jnp.atleast_1d(v[1]))

        # Get all flight states
        if mode == "zip":
            processed_arrays = jnp.broadcast_arrays(*raw_arrays)
        elif mode == "mesh":
            grids = jnp.meshgrid(*raw_arrays, indexing="ij")
            processed_arrays = [g.ravel().reshape(-1, 1) for g in grids]
        else:
            raise ValueError(f"Invalid mode {mode}. Supported modes: 'zip', 'mesh'.")

        total_states = len(processed_arrays[0])

        # Prepare for grads if provided
        if settings.analysis.gradient_map is not None:
            g_map = settings.analysis.gradient_map
            inp = g_map.state_inputs
            out = g_map.state_outputs

            grad_pairs = product(out, inp)
            grad_keys = [f"d{p[0].tag}_d{p[1].tag}" for p in grad_pairs]
            grad_idxs = list(product(range(len(out)), range(len(inp))))

        # Initialize Analysis once
        init_results = self.initialization_process.run(base_state.expand_rows(batch_size), system, settings)
        state = init_results[0]
        system = init_results[1]
        settings = init_results[2]

        # Batch over computation
        if handle is not None:
            pbar = range(0, total_states, batch_size)
        else:
            pbar = trange(0, total_states, batch_size, desc=f"Running {self.tag} Analysis")
        for i in pbar:
            batch_arrays = tuple(arr[i : i + batch_size].reshape(-1, 1) for arr in processed_arrays)
            actual_size = len(batch_arrays[0])

            clean_batch_arrays = jax.device_get(batch_arrays)
            for j, ik in enumerate(input_keys):
                all_inputs[ik].append(clean_batch_arrays[j])

            if actual_size < batch_size:
                pad_length = ((0, batch_size - actual_size), (0, 0))
                batch_arrays = tuple(jnp.pad(arr, pad_length, mode="edge") for arr in batch_arrays)

            batch_state = eqx.tree_at(lambda s: ru.get_all_targets(s, target_map), state, batch_arrays)

            try:
                res = self._compiled_step(batch_state, system, settings)

                raw_output_arrs = jax.device_get(ru.get_all_targets(res[0], self.output_mappings.values()))
                clean_output_arrs = [arr[:actual_size] for arr in raw_output_arrs]

                for l, key in enumerate(self.output_mappings.keys()):
                    all_outputs[key].append(clean_output_arrs[l])

                if settings.analysis.gradient_map is not None:
                    jac_arr = jax.device_get(res[3])
                    for i, key in enumerate(grad_keys):
                        out_idx, in_idx = grad_idxs[i]
                        v_np = jac_arr[:actual_size, out_idx, in_idx]
                        all_grads[key].append(v_np)

            except Exception:
                logger.error(f"Failed at states {i} to {i + batch_size}. Injecting NaNs...", exc_info=True)
                nan_array = np.full((actual_size, 1), np.nan, dtype=np.float64)

                for key in self.output_mappings.keys():
                    all_outputs[key].append(nan_array)

                if settings.analysis.gradient_map is not None:
                    for key in grad_keys:
                        all_grads[key].append(nan_array)

        merged_results = all_inputs | all_outputs | all_grads

        if self.db_path is not None:
            db_root = zarr.open_group(self.db_path, mode="a", zarr_format=2)
            for key, list_of_arrays in merged_results.items():
                # Concatenate the individual batch arrays
                full_array = np.concatenate(list_of_arrays, axis=0)

                if key not in db_root:
                    db_root.create_array(
                        name=key,
                        shape=(0,) + full_array.shape[1:],
                        chunks=(100_000,) + full_array.shape[1:],
                        dtype=full_array.dtype,
                        compressor=Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE),
                    )
                db_root[key].append(full_array, axis=0)

        return merged_results


# ----------------------------------------------------------------------------------------------------------------------
#  Sharded Dataset Generator
# ----------------------------------------------------------------------------------------------------------------------


class JAXCompileFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()

        # 1. Identify if this is a compilation/tracing log
        is_compile_log = any(
            keyword in msg
            for keyword in ["Compiling", "tracing + transforming", "Finished jaxpr to MLIR", "Finished XLA compilation"]
        )

        # 2. If it IS a compile log, apply our strict whitelist & formatting
        if is_compile_log:
            # Block it if it's not the main solve
            if "jit(run)" not in msg:
                return False

            # If it is the main solve, truncate the massive PyTree dump
            if "with global shapes and types" in msg:
                parts = msg.split("with global shapes and types")
                prefix = parts[0] + "with global shapes and types"
                suffix = parts[1][:30] if len(parts) > 1 else ""

                record.msg = f"{prefix} {suffix} ... [PyTree Truncated]"
                record.args = ()

            return True

        # 3. If it's NOT a compile log (e.g., GPU memory warning), let it through untouched
        return True


class ShardManager:
    def __init__(self, cache_dir, storage_dir, max_rows=3_000_000, handle="rcaide_dataset.manager"):
        self.local_dir = Path(cache_dir)
        self.hdd_dir = Path(storage_dir)
        self.max_rows = max_rows
        self.prefix = handle.split(".")[0]

        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.hdd_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(handle)
        self.current_shard_idx = self._find_resume_shard()
        self.current_rows = 0
        self.active_root = None
        self.compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)

        self._open_active_shard()

    def _find_resume_shard(self):
        existing = list(self.hdd_dir.glob(f"{self.prefix}_shard_*.zarr"))
        if not existing:
            return 0
        indices = [int(p.stem.split("_")[-1]) for p in existing]
        return max(indices) + 1

    def _open_active_shard(self):
        shard_name = f"{self.prefix}_shard_{self.current_shard_idx:04d}.zarr"
        self.active_path = self.local_dir / shard_name

        if self.active_path.exists() and self.current_rows == 0:
            shutil.rmtree(self.active_path)

        self.active_root = zarr.open_group(str(self.active_path), mode="a", zarr_format=2)

    def offload_and_rollover(self):
        self.logger.info(f"Sealing Shard {self.current_shard_idx:04d}...")
        zarr.consolidate_metadata(str(self.active_path))

        hdd_path = self.hdd_dir / self.active_path.name
        shutil.move(str(self.active_path), str(hdd_path))

        self.current_shard_idx += 1
        self.current_rows = 0
        self._open_active_shard()

    def append_data(self, data_dict):
        batch_size = len(next(iter(data_dict.values())))
        if self.current_rows + batch_size > self.max_rows:
            self.offload_and_rollover()

        for key, arr in data_dict.items():
            if key not in self.active_root:
                self.active_root.create_array(
                    name=key,
                    shape=(0,) + arr.shape[1:],
                    chunks=(100_000,) + arr.shape[1:],
                    dtype=arr.dtype,
                    compressor=self.compressor,
                )
            self.active_root[key].append(arr, axis=0)

        self.current_rows += batch_size


class ShardedDatasetGenerator:
    """
    Orchestrates batched runs for any RCAIDE BatchProcess.
    Slices total design space into manageable shards, executes them locally,
    and offloads them to medium-term storage.
    """

    def __init__(
        self,
        batch_analysis: Any,
        cache_dir: str | Path,
        storage_dir: str | Path,
        shard_size: int = 3_000_000,
        tag: str = "DataGenerator",
    ):

        self.cache_dir = Path(cache_dir)
        self.storage_dir = Path(storage_dir)
        self.shard_size = shard_size

        self.batch_process = batch_analysis

        self.tag = tag
        self.dataset_prefix = "_".join(tag.split(" ")).lower()

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self._setup_logger()
        self.shard_manager = ShardManager(cache_dir, storage_dir, shard_size, self.dataset_prefix + ".manager")

    def _setup_logger(self):
        self.logger = logging.getLogger(self.dataset_prefix)
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            # Logfile Handler
            fh = logging.FileHandler(self.storage_dir / f"{self.dataset_prefix}.log")
            fh.setLevel(logging.INFO)

            # Console Handler
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)

            # Format: [2026-06-05 10:45:12] - INFO - Generating epoch 3...
            formatter = logging.Formatter("[%(asctime)s] - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)

            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

        jax_logger = logging.getLogger("jax")
        jax_logger.propagate = False
        jax_logger.handlers.clear()

        jax_filter = JAXCompileFilter()
        for handler in self.logger.handlers:
            handler.addFilter(jax_filter)
            jax_logger.addHandler(handler)

        if getattr(jax.config, "jax_log_compiles", False):
            jax_logger.setLevel(logging.INFO)
        else:
            jax_logger.setLevel(logging.WARNING)

    def run(
        self,
        settings,
        state_kwargs: Dict[str, np.ndarray],
        state_mode: str = "zip",
        system: Optional[Any] = None,
        system_iter: Optional[Iterable[Tuple[Any, Dict[str, float]]]] = None,
        total_systems: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        # Resolve inputs upfront to determine total states
        raw_states = [np.atleast_1d(v) for v in state_kwargs.values()]
        state_keys = list(state_kwargs.keys())

        if state_mode == "zip":
            proc_states = np.broadcast_arrays(*raw_states)
        elif state_mode == "mesh":
            grids = np.meshgrid(*raw_states, indexing="ij")
            proc_states = [grid.ravel() for grid in grids]
        else:
            raise ValueError("State mode must be 'zip' or 'mesh'")

        states_per_system = len(proc_states[0])
        flat_state_kwargs = {k: v.reshape(-1, 1) for k, v in zip(state_keys, proc_states)}

        if system_iter is None:
            if system is None:
                raise ValueError("Must provide either 'system' or 'system_iter'.")
            system_iter = [(system, {})]

        if getattr(jax.config, "jax_log_compiles", False):
            os.system("cls" if os.name == "nt" else "clear")

        self.logger.info("=== INITIALIZING SHARDED GENERATION ===")
        self.logger.info(f"Initialized Generalized Generator. {states_per_system} states per geometry.")

        # System Loop
        with tqdm(desc="Processing Systems", total=total_systems) as pbar:
            for s_idx, (sys, meta) in enumerate(system_iter):
                try:
                    res = self.batch_process.run(
                        system=sys,
                        settings=settings,
                        mode="zip",
                        batch_size=batch_size,
                        handle=self.dataset_prefix + ".analysis",
                        **flat_state_kwargs,
                    )

                    for k, v in meta.items():
                        res[k] = np.full((states_per_system, 1), v, dtype=np.float64)

                    conformed_dict = {}
                    for key, val in res.items():
                        if isinstance(val, list):
                            conformed_dict[key] = np.concatenate(val, axis=0)
                        else:
                            conformed_dict[key] = np.asarray(val)

                    self.shard_manager.append_data(conformed_dict)

                except Exception:
                    self.logger.error(f"Failuire on system {s_idx}. Skipping.", exc_info=True)
                    continue

                pbar.update(1)
                if s_idx == 0:
                    pbar.start_t = time.time()
                    pbar.last_print_t = time.time()

        self.shard_manager.offload_and_rollover()
        shutil.rmtree(self.cache_dir)
        self.logger.info(f"{self.tag} Complete.")


# -----------------------------------------------------------------------------------------------------------------------
# Compression Benchmarking
# -----------------------------------------------------------------------------------------------------------------------


def benchmark_zarr_compression(num_states=1_000_000, chunk_size=100_000):
    print(f"Generating {num_states} simulated aerodynamic states...")

    # 1. Simulate aerodynamic data (smooth gradients, floats)
    # Bitshuffle works best on data where values don't change randomly
    mach = np.linspace(0.1, 2.0, num_states)
    alpha = np.sin(np.linspace(0, 10, num_states)) * 15.0
    dCL_dAlpha = np.cos(alpha) * 0.1  # Simulated smooth gradient

    # Pack into a standard 2D array: (states, features)
    data = np.column_stack([mach, alpha, dCL_dAlpha])
    raw_bytes = data.nbytes
    print(f"Uncompressed Data Size: {raw_bytes / (1024**2):.2f} MB")

    # 2. Define the competitors
    compressors = {
        "Uncompressed": None,
        "LZ4 (Speed focus)": Blosc(cname="lz4", clevel=5, shuffle=Blosc.NOSHUFFLE),
        "LZ4 + BitShuffle": Blosc(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE),
        "Zstd Lvl 1 + BitShuffle": Blosc(cname="zstd", clevel=1, shuffle=Blosc.BITSHUFFLE),
        "Zstd Lvl 5 + BitShuffle (Default)": Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE),
        "Zstd Lvl 9 + BitShuffle (Max)": Blosc(cname="zstd", clevel=9, shuffle=Blosc.BITSHUFFLE),
        "Zstd Lvl 5 + ByteShuffle": Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE),
    }

    results = []

    # 3. Run the Benchmark
    with tempfile.TemporaryDirectory() as tmpdir:
        for name, comp in compressors.items():
            path = os.path.join(tmpdir, f"{name.replace(' ', '_')}.zarr")

            # --- WRITE TEST ---
            t0 = time.perf_counter()
            z = zarr.array(
                data, chunks=(chunk_size, data.shape[1]), compressor=comp, store=path, overwrite=True, zarr_format=2
            )
            write_time = time.perf_counter() - t0

            # --- SIZE CHECK ---
            # z.nbytes is uncompressed, z.nbytes_stored is on disk
            compressed_bytes = z.nbytes_stored()
            ratio = raw_bytes / compressed_bytes if compressed_bytes > 0 else 1.0

            # --- READ TEST ---
            t0 = time.perf_counter()
            _ = z[:]  # Read entire array into memory
            read_time = time.perf_counter() - t0

            results.append(
                {
                    "Compressor": name,
                    "Size (MB)": compressed_bytes / (1024**2),
                    "Ratio": ratio,
                    "Write Speed (MB/s)": (raw_bytes / (1024**2)) / write_time,
                    "Read Speed (MB/s)": (raw_bytes / (1024**2)) / read_time,
                }
            )

    # 4. Display Results
    df = pd.DataFrame(results).round(2)
    df = df.sort_values(by="Read Speed (MB/s)", ascending=False)
    print("\n--- Benchmark Results ---")
    print(df.to_string(index=False))


if __name__ == "__main__":
    benchmark_zarr_compression()
