# RCAIDE/Framework/Analyses/Batched.py
# (c) Copyright 2026 Aerospace Research Community LLC
#
# Created: Jun 2026, J. Smart
# Modified: Jun 2026, J. Smart

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from typing import TYPE_CHECKING, Any, Optional
from pathlib import Path

import jax
import jax.numpy as jnp
import equinox as eqx

import numpy as np

import os
import shutil
import zarr

from tqdm import trange

# --- Framework Imports (Strictly for Type Hinting to avoid Circular Imports) ---
if TYPE_CHECKING:
    from RCAIDE.Framework.State import State
    from RCAIDE.Framework.System import System
    from RCAIDE.Framework.Settings import Settings

# ----------------------------------------------------------------------------------------------------------------------
#  Batch Analysis Utilities
# ----------------------------------------------------------------------------------------------------------------------

class ShardedDatasetGenerator:
    """
    Orchestrates batched runs for any RCAIDE BatchProcess.
    Slices total design space into manageable shards, executes them locally, 
    and offloads them to medium-term storage.
    """
    def __init__(
        self, 
        batch_process: Any, 
        local_dir: str | Path, 
        storage_dir: str | Path, 
        epoch_size: int = 1_000,
        tag: str = "RCAIDE Dataset"
    ):
        self.batch_process = batch_process
        self.local_dir = Path(local_dir)
        self.hdd_dir = Path(storage_dir)
        self.epoch_size = epoch_size

        self.tag = tag
        self.dataset_prefix = '_'.join(tag.split(' ')).lower()
        
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.hdd_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self, 
        system, 
        settings, 
        mode: str = "zip", 
        batch_size: Optional[int] = None, 
        **kwargs
    ):
        # 1. Resolve inputs upfront to determine total states
        # We must resolve 'mesh' or 'zip' HERE so we can slice across epochs properly.
        raw_arrays = [np.atleast_1d(v) for v in kwargs.values()]
        keys = list(kwargs.keys())
        
        if mode == 'zip':
            processed_arrays = np.broadcast_arrays(*raw_arrays)
        elif mode == 'mesh':
            grids = np.meshgrid(*raw_arrays, indexing='ij')
            processed_arrays = [grid.ravel() for grid in grids]
        else:
            raise ValueError("Mode must be 'zip' or 'mesh'")

        total_states = len(processed_arrays[0])
        if settings.verbose:
            print("Total Dataset States")
        total_epochs = int(np.ceil(total_states / self.epoch_size))
        
        print(f"Starting Sharded Generation: {total_states} total states across {total_epochs} epochs, {self.epoch_size} each.")

        # 2. The Outer Epoch Loop
        for epoch_idx, start_idx in enumerate(trange(0, total_states, self.epoch_size, desc=f"Generating {self.tag}")):
            end_idx = min(start_idx + self.epoch_size, total_states)
            
            # Slice kwargs specifically for this epoch
            epoch_kwargs = {}
            for key, arr in zip(keys, processed_arrays):
                epoch_kwargs[key] = arr[start_idx:end_idx]

            # Define temporary local path and final HDD path
            shard_name = f"{self.dataset_prefix}_shard_{epoch_idx:04d}.zarr"
            local_zarr = self.local_dir / shard_name
            hdd_zarr = self.hdd_dir / shard_name

            # 3. Hand off to the RCAIDE BatchProcess
            # CRITICAL: We pass mode="zip" here, regardless of what the user originally asked for.
            # We already expanded the mesh above, so the BatchProcess just needs to iterate 1-to-1.
            self.batch_process.run(
                system=system,
                settings=settings,
                mode="zip", 
                batch_size=batch_size,
                db_path=local_zarr,
                **epoch_kwargs
            )

            # 4. Consolidate and Offload
            self._finalize_and_offload(local_zarr, hdd_zarr)

        print("\nAll epochs generated and offloaded successfully.")

    def _finalize_and_offload(self, local_path: Path, hdd_path: Path):
        """Seals the Zarr database and moves it to slower storage."""
        print(f"Consolidating Zarr metadata for {local_path.name}...")
        # Consolidating makes opening the dataset much faster for ML dataloaders
        zarr.consolidate_metadata(str(local_path))
        
        print(f"Offloading to HDD: {hdd_path}...")
        shutil.move(str(local_path), str(hdd_path))
        print("Offload complete.")
