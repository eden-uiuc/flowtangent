# flowtangent/Framework/Analyses/Batched.py
# (c) Copyright 2026 Aerospace Research Community LLC
#
# Created: Jun 2026, J. Smart
# Modified: Jun 2026, J. Smart

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from .. import Settings, System

import logging
import os
import shutil
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import zarr
from numcodecs import Blosc
from tqdm import tqdm, trange

from .. import Process, State, System
from ..utils import TreePath, field, get_all_targets, null_step, update
from ._implicit import ImplicitAnalysis

# ----------------------------------------------------------------------------------------------------------------------
#  Batch Analysis
# ----------------------------------------------------------------------------------------------------------------------


class BatchedAnalysis(Process):
    name: str = field("Batched Analysis")

    analyze: Process = field(Process)
    batch_inputs: tuple[TreePath, ...] = field(())

    def __init__(
        self,
        analyze: Process = Process(name="Batched Analysis"),
        batch_inputs: tuple[TreePath, ...] = (),
        name: str = "Batched Analysis",
        *,
        _initial_state: Optional[State] = None,
        _initial_system: Optional[System] = None,
        _initial_settings: Optional[Settings] = None,
        _filter_map: Optional[dict] = None,
    ):

        self.name = name
        self.function = null_step
        self.initial_step = 0

        self._initial_state = _initial_state
        self._initial_system = _initial_system
        self._initial_settings = _initial_settings

        # Handle mutable dictionary default safely
        self._filter_map = (
            _filter_map
            if _filter_map is not None
            else {
                "energy": r"state\.energy\.nodes\.\[*\].",
            }
        )

        self.analyze = analyze

        if not isinstance(self.analyze, ImplicitAnalysis):
            self.batch_inputs = batch_inputs
        else:
            vars = self.analyze.variables
            # fmt: off
            var_inputs = tuple(TreePath(
                path=v.state_path.path,
                value=jnp.atleast_3d(v.initial_value)) for v in vars)
            self.batch_inputs = self.batch_inputs + var_inputs
            # fmt: on

    @property
    def steps(self): # type: ignore
        return self.analyze.steps

    def _batch_inputs(self, mode="mesh"):

        batch_arrays = []

        raw_arrays = [jnp.atleast_1d(jnp.array(p.value)) for p in self.batch_inputs]

        if mode == "zip":
            input_size = raw_arrays[0].shape[0]
            for arr in raw_arrays:
                if arr.shape[0] != input_size:
                    raise ValueError("In 'zip' batch mode all input arrays must have the same size.")
                batch_arrays = raw_arrays

        elif mode == "mesh":
            num_inputs = len(raw_arrays)
            input_sizes = [arr.shape[0] for arr in raw_arrays]

            for i, arr in enumerate(raw_arrays):
                leading_shape = [1] * num_inputs
                leading_shape[i] = input_sizes[i]

                target_shape = tuple(leading_shape) + arr.shape[1:]
                reshaped_arr = arr.reshape(target_shape)

                full_leading_shape = tuple(input_sizes)
                full_target_shape = full_leading_shape + arr.shape[1:]
                broadcasted_arr = jnp.broadcast_to(reshaped_arr, full_target_shape)

                final_shape = (-1,) + arr.shape[1:]
                final_arr = broadcasted_arr.reshape(final_shape)

                batch_arrays.append(final_arr)

        else:
            raise ValueError("Batch mode must be 'zip' or 'mesh'.")

        total_size = batch_arrays[0].shape[0]
        state_inputs = []
        system_inputs = []

        for i, p in enumerate(self.batch_inputs):
            path_tup = p.path
            if path_tup[0] == "state":
                state_inputs.append(replace(p, path=path_tup[1:], value=batch_arrays[i]))
            elif path_tup[0] == "system":
                system_inputs.append(replace(p, path=path_tup[1:], value=batch_arrays[i]))
            else:
                # Default fallback: assume state variable if no prefix
                state_inputs.append(replace(p, value=batch_arrays[i]))

        return state_inputs, system_inputs, total_size

    @staticmethod
    def _update_inputs(pytree: State | System, idx: int, batch_size: int, inputs: Sequence[TreePath]):
        input_arrays = tuple(si.value[idx : idx + batch_size] for si in inputs)
        actual_size = input_arrays[0].shape[0]

        if actual_size < batch_size:
            pads = [((0, batch_size - actual_size),) + ((0, 0),) * (arr.ndim - 1) for arr in input_arrays]
            padded_arrays = [jnp.pad(arr, pads[i], mode="edge") for i, arr in enumerate(input_arrays)]
        else:
            padded_arrays = input_arrays

        return update(pytree, lambda p: get_all_targets(p, inputs), padded_arrays)

    def __call__(self, state: State, system: System, settings: Settings) -> Tuple[State, System, Settings]:

        batch_size = settings.numerical.batch_size
        batch_mode = settings.numerical.batch_mode

        state_inputs, _, total_states = self._batch_inputs(batch_mode)

        batch_state = state.expand_batch(batch_size)
        batch_axes = batch_state.get_vmap_axes()

        batch_analyze = eqx.filter_jit(eqx.filter_vmap(self.analyze.__call__, in_axes=(batch_axes, None, None)))

        if settings.logging.handle is not None:
            pbar = range(0, total_states, batch_size)
        else:
            pbar = trange(0, total_states, batch_size, desc=self.name, leave=False)

        if not settings.DEBUG_MODE and not settings._DEV_MODE:
            analyis_settings = replace(settings, verbose=False)
        else:
            analyis_settings = settings

        batch_states = []
        for batch_idx in pbar:
            updated_state = self._update_inputs(batch_state, batch_idx, batch_size, state_inputs)
            b_st, _, _ = batch_analyze(updated_state, system, analyis_settings)
            actual_size = min(batch_size, total_states - batch_idx)
            if actual_size < batch_size:
                b_st = b_st.truncate(actual_size)
            batch_states.append(b_st)

        f_st = State.concatenate(batch_states)

        return f_st, system, settings


# ----------------------------------------------------------------------------------------------------------------------
#  Sharded Dataset Generator
# ----------------------------------------------------------------------------------------------------------------------


class ShardManager:
    def __init__(self, cache_dir, storage_dir, max_rows=3_000_000, handle="Flowtangent_dataset.manager"):
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
    Orchestrates batched runs for any Flowtangent BatchProcess.
    Slices total design space into manageable shards, executes them locally,
    and offloads them to medium-term storage.
    """

    def __init__(
        self,
        batch_analysis: Any,
        cache_dir: str | Path,
        storage_dir: str | Path,
        shard_size: int = 3_000_000,
        name: str = "DataGenerator",
    ):

        self.cache_dir = Path(cache_dir)
        self.storage_dir = Path(storage_dir)
        self.shard_size = shard_size

        self.batch_process = batch_analysis

        self.name = name
        self.dataset_prefix = "_".join(name.split(" ")).lower()

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
        self.logger.info(f"{self.name} Complete.")


# ----------------------------------------------------------------------------------------------------------------------
# Compression Benchmarking
# ----------------------------------------------------------------------------------------------------------------------


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
