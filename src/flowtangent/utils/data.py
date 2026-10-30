# type: ignore
# Dask and other data libraries aren't type hinted an so create nuisance errors.
# src/flowtangent/utils/data.py
"""FlowTangent data loading and dataset management utilities.

This module provides a unified data loading architecture that bridges the gap
between PyTorch's robust multi-processing data streams and JAX's functional
compilation requirements. It introduces a universal `Dataset` PyTree that duck-types
as a PyTorch Dataset, offering out-of-the-box lazy evaluation for massive on-disk
formats (Zarr, HDF5, Parquet, NetCDF) via dynamic file proxies.

Crucially, it replaces PyTorch's default collation with `numpy_collate` to guarantee
that downstream JAX optimizers never receive incompatible Torch tensors. It also
provides mechanisms for dynamic dataset expansion—like virtual columns and row
appends—to support active learning loops without mutating the original source files on disk.
"""

import copy
import math
import warnings
from beartype.typing import Callable
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional, Union

import dask.array as da
import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import netCDF4 as nc  # noqa: N813
import numpy as np
import polars as pl
import pyarrow.csv as pcsv
import pyarrow.parquet as pq
import scipy.io as sio
import torch
import xarray as xr
import zarr
from torch.utils.data import (
    ChainDataset,
    ConcatDataset,
    IterableDataset,
    DataLoader as TorchDataLoader,
    # Samplers
    Sampler,
    BatchSampler,
    SequentialSampler,
    StackDataset,
    Subset,
    RandomSampler,
    SubsetRandomSampler,
    WeightedRandomSampler,
    # Utils
    get_worker_info,
    random_split,
)

from tqdm.dask import TqdmCallback

from .base import Module, static_field
from .typing import _

# ----------------------------------------------------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------------------------------------------------


def numpy_collate(batch: list[Any]) -> Any:
    """Recursively converts a PyTorch batch into stacked NumPy arrays.

    Replaces `torch.utils.data.default_collate` for NumPy-based training inputs.
    Preserves nested dictionaries, tuples, and lists; tensors are detached and
    moved to the CPU before conversion.

    Args:
        batch (list[Any]): Nonempty samples of compatible structure. Leaves may
            be ND NumPy/JAX arrays, tensors, numbers, booleans, or NumPy scalars.

    Returns:
        Any: A NumPy array, or a nested dictionary, tuple, or list of collated values.

    Raises:
        IndexError: If `batch` is empty.
        TypeError: If a leaf's first sample has an unsupported type.
        ValueError: If array samples cannot be stacked because their shapes differ.
        KeyError: If a later dictionary lacks a key from the first sample.
    """
    elem = batch[0]

    if isinstance(elem, torch.Tensor):
        return np.stack([x.detach().cpu().numpy() for x in batch])
    elif isinstance(elem, (np.ndarray, jax.Array)):
        return np.stack([np.asarray(x) for x in batch])
    elif isinstance(elem, (int, float, bool, np.generic)):
        return np.array(batch)
    elif isinstance(elem, (tuple, list)):
        transposed = zip(*batch)
        return type(elem)(numpy_collate(samples) for samples in transposed)
    elif isinstance(elem, dict):
        return {key: numpy_collate([d[key] for d in batch]) for key in elem}
    else:
        raise TypeError(f"numpy_collate cannot handle batch of type {type(elem)}")


class DataLoader(TorchDataLoader):
    """A standalone facade wrapper around PyTorch's `DataLoader`.

    Forces the use of `numpy_collate` by default to ensure batches are returned
    as pure NumPy arrays instead of PyTorch Tensors, providing drop-in compatibility
    between PyTorch multi-processing data streams and JAX optimizers.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load. Defaults to 1.
        shuffle (bool, optional): Set to True to have the data reshuffled at every epoch. Defaults to False.
        collate_fn (callable, optional): Merges a list of samples to form a mini-batch. Defaults to `numpy_collate`.
        **kwargs: Additional keyword arguments passed directly to `torch.utils.data.DataLoader`.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, collate_fn=numpy_collate, **kwargs):
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, **kwargs)


class LatentDataLoader:
    """Wraps a physical DataLoader to encode a physical batch into a latent batch on the fly.

    Duck-types as a standard `DataLoader` for downstream training loops.
    Useful for training surrogate models directly on the encoded latent space
    without explicitly saving intermediate encoded datasets to disk.

    Args:
        dataloader (DataLoader): The primary DataLoader supplying physical states `(X, Y)`.
        manifold (Any): A FlowTangent autoencoder/manifold model equipped with an `.encode()` method.
    """

    def __init__(self, dataloader, manifold):
        self.dataloader = dataloader
        self.manifold = manifold
        self.dataset = dataloader.dataset  # Satisfies duck-typing checks

    def __iter__(self):
        """Yields batches mapped through the manifold encoder."""
        for x_batch, y_batch in self.dataloader:
            # Encode physical batch to latent batch
            z_batch = jax.vmap(self.manifold.encode)(x_batch)
            yield z_batch, y_batch

    def __len__(self):
        """Returns the number of batches in the underlying dataloader."""
        return len(self.dataloader)


class SlicedDataLoader:
    """Dynamically slices the feature dimension of batches streaming from a DataLoader.

    Duck-types as a standard DataLoader. Primarily used in multi-GPU or distributed
    training loops where data parallelization splits batches across devices.

    Args:
        dataloader (DataLoader): The base DataLoader stream.
        split_indices (int | Sequence[int]): The array indices at which to split the feature dimension.
        chunk_idx (int): Which specific split section to yield to this worker.
    """

    def __init__(self, dataloader, split_indices, chunk_idx):
        self.dataloader = dataloader
        self.split_indices = split_indices
        self.chunk_idx = chunk_idx
        self.dataset = getattr(dataloader, "dataset", None)

    def __iter__(self):
        """Yields specific feature-chunks of the underlying batch stream."""
        for batch in self.dataloader:
            if isinstance(batch, (tuple, list)):
                x_batch, y_batch = batch
                x_parts = jnp.split(x_batch, self.split_indices, axis=-1)
                yield x_parts[self.chunk_idx], y_batch
            else:
                x_parts = jnp.split(batch, self.split_indices, axis=-1)
                yield x_parts[self.chunk_idx]

    def __len__(self):
        """Returns the number of batches in the underlying dataloader."""
        return len(self.dataloader)


def slice_data(data: Any, split_indices: Union[int, list[int]], chunk_idx: int) -> Any:
    """Selects a feature chunk from arrays or wraps a loader to slice its batches.

    Splits the final feature axis while preserving targets in an `(X, Y)` pair.
    Objects with both `__iter__` and `dataset` attributes are treated as loaders.

    ### Mathematical Formulation
    For feature blocks $(X_0, ..., X_{K-1}) = split(X)$ along the final axis,
    the selected features are $X_c$, where $c$ is `chunk_idx`.

    ### Process Architecture
    Return a lazy slicing wrapper for loaders; otherwise split the feature array
    immediately and return the selected chunk with any original targets.

    ### Usage Example
    ```python
    import jax.numpy as jnp
    from flowtangent.utils.data import slice_data

    x, y = jnp.ones((4, 6)), jnp.zeros((4,))
    chunk, targets = slice_data((x, y), 3, 1)
    assert chunk.shape == (4, 2)
    assert targets is y
    ```

    Args:
        data (Any): A loader, an `(X, Y)` tuple/list, or an ND feature array with
            at least one axis. Targets are returned unchanged.
        split_indices (Union[int, list[int]]): A positive number of equal-sized
            sections, or split positions along the final axis, as for `jnp.split`.
        chunk_idx (int): Chunk index; negative indices count from the last chunk.

    Returns:
        Any: A `SlicedDataLoader`, a feature array chunk, or `(chunk, Y)` for paired data.

    Raises:
        ValueError: If an integer section count does not evenly divide the feature axis.
        IndexError: If `chunk_idx` is out of range or a data pair lacks its targets.
            For loaders, splitting/indexing errors occur during iteration.
        ZeroDivisionError: If the integer section count is zero; for loaders,
            this occurs during iteration.
    """
    if hasattr(data, "__iter__") and hasattr(data, "dataset"):
        return SlicedDataLoader(data, split_indices, chunk_idx)
    # If it's a raw tuple of (X, Y) or a raw X array
    if isinstance(data, (tuple, list)):
        return jnp.split(data[0], split_indices, axis=-1)[chunk_idx], data[1]
    return jnp.split(data, split_indices, axis=-1)[chunk_idx]


LoaderType = Union[DataLoader, LatentDataLoader, SlicedDataLoader]
"""Union[DataLoader, LatentDataLoader, SlicedDataLoader]: Type alias for standard dataloader structures."""

# ----------------------------------------------------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------------------------------------------------


class Dataset(Module):
    """Universal FlowTangent Dataset factory and abstraction layer.

    Duck-types as a PyTorch Dataset (implementing `__len__` and `__getitem__`)
    while remaining a valid, immutable Equinox PyTree module. It utilizes proxy
    classes to lazily evaluate massive on-disk datasets (Zarr, HDF5, Parquet, NetCDF),
    and handles dynamic row appends (for active learning) through a non-destructive
    virtual overlay mechanism.
    """

    _source: Any = static_field(_)
    _proxy: Any = static_field(_)
    _proxy_vars: tuple[str, ...] = static_field(())

    _virtual_columns: dict[str, jax.Array] = _
    _indices: Optional[jax.Array] = _

    _transforms: tuple[Callable, ...] = static_field(())

    _appended_data: dict = _
    _appended_len: int = 0

    def __check_init__(self):
        """Initializes empty containers for the virtual data overlay post-instantiation."""
        super(Dataset, self).__check_init__()
        if getattr(self, "_virtual_columns", _) is _:
            object.__setattr__(self, "_virtual_columns", {})
        if getattr(self, "_appended_data", _) is _:
            object.__setattr__(self, "_appended_data", {})
        if getattr(self, "_indices", _) is _:
            object.__setattr__(self, "_indices", None)
        if getattr(self, "_appended_data", _) is _:
            object.__setattr__(self, "_appended_data", {})

        if getattr(self, "_proxy_vars", None) == ():
            if self._proxy is not _:
                p_vars = [
                    k for k in dir(self._proxy) if not k.startswith("_") and hasattr(getattr(self._proxy, k), "shape")
                ]
                object.__setattr__(self, "_proxy_vars", tuple(p_vars))

    @property
    def full_vars(self) -> list[str]:
        """list[str]: Returns a sorted list of all column names available in the disk proxy and virtual overlay."""
        virtual_keys = list(self._virtual_columns.keys())
        return sorted(list(set(list(self._proxy_vars) + virtual_keys)))

    def transform_variable(self, transform_fn: Callable):
        """
        Returns a shallow copy of the dataset with a new transformation function applied.
        """
        new_tfs = self._transforms + (transform_fn,)
        new_ds = copy.copy(self)
        object.__setattr__(new_ds, "_transforms", new_tfs)
        return new_ds

    def add_variable(self, name: str, data: Optional[jax.Array] = None, quiet: bool = False) -> "Dataset":
        """Registers a new variable in the virtual overlay without modifying the disk source.

        Args:
            name (str): The column name to register.
            data (Optional[jax.Array], optional): The array data. If None, it is filled with NaNs.
            quiet (bool, optional): Suppresses the warning when a user typos a column name. Defaults to False.

        Raises:
            ValueError: If the provided data array length does not match the dataset length.

        Returns:
            Dataset: A new instance of the Dataset containing the added virtual column.
        """
        if not quiet:
            warnings.warn(
                f"Creating virtual dataset column for new variable '{name}'. "
                "If this is a typo, check your Parameter names.",
                UserWarning,
                stacklevel=2,
            )

        if data is None:
            data = jnp.full((len(self),), jnp.nan)
        else:
            data = jnp.asarray(data)
            if len(data) != len(self):
                raise ValueError(f"Data length {len(data)} != dataset length {len(self)}.")

        new_virtuals = {**self._virtual_columns, name: data}
        return eqx.tree_at(lambda d: d._virtual_columns, self, new_virtuals)

    def compute_variable(self, name: str, expr_fn: Callable, chunk_size: int=1000, quiet: bool = False, **compute_kwargs):
        """
        Derives a new variable out-of-core.
        compute_kwargs are passed directly to the backend's ephemeral compute engine.
        """
        # 1. Ask the backend to spin up a temporary compute engine just for this math
        with self._proxy._compute_engine(chunk_size=chunk_size,**compute_kwargs) as compute_env:
            # 2. Execute the user's graph
            derived_data = expr_fn(compute_env)
            
            # 3. Force Dask/Polars to evaluate the graph and return a concrete numpy/JAX array
            if hasattr(derived_data, "compute"):
                concrete_array = jnp.asarray(derived_data.compute())
            else: # For eager engines
                concrete_array = jnp.asarray(derived_data)

        # 4. Attach to the virtual overlay (which returns a new Dataset PyTree)
        return self.add_variable(name, concrete_array, quiet=quiet)

    def append(self, row_dict: dict[str, Any]) -> "Dataset":
        """Appends a new row to the dataset via the virtual overlay.

        Useful for expanding datasets dynamically during Active Learning loops.
        Variables missing from `row_dict` that exist in the broader dataset are
        automatically padded with NaNs.

        Args:
            row_dict (dict[str, Any]): A dictionary mapping column names to values.

        Returns:
            Dataset: A new instance of the Dataset containing the appended row.
        """
        new_appended = dict(self._appended_data)

        base_columns = {}
        for var in self.full_vars:
            if var in self._virtual_columns:
                base_col = self._virtual_columns[var]
            elif self._proxy is not _:
                try:
                    base_col = getattr(self._proxy, var)
                except AttributeError:
                    continue
            else:
                continue
            if hasattr(base_col, "shape"):
                base_columns[var] = base_col

        for var, base_col in base_columns.items():
            # Extract the new value, defaulting to NaN if the oracle didn't provide it
            trailing_shape = base_col.shape[1:]
            if var in row_dict:
                val = jnp.asarray(row_dict[var]).reshape((1, *trailing_shape))
            else:
                val = jnp.full((1, *trailing_shape), jnp.nan)

            if var in new_appended:
                new_appended[var] = jnp.concatenate([new_appended[var], val], axis=0)
            else:
                new_appended[var] = val

        new = eqx.tree_at(lambda d: (d._appended_data, d._appended_len), self, (new_appended, self._appended_len + 1))
        if self._indices is not None:
            new_idx = object.__getattribute__(self, "_len") + self._appended_len
            new = eqx.tree_at(lambda d: d._indices, new, jnp.append(self._indices, new_idx))
        return new

    def filter(self, mask: jax.Array) -> "Dataset":
        """Creates a zero-copy filtered view of the dataset using a boolean mask or index array.

        Args:
            mask (jax.Array): A boolean mask array or an integer index array.

        Returns:
            Dataset: A new instance of the Dataset configured to yield only the masked subset.
        """
        # Convert boolean mask to integer indices
        if mask.dtype == jnp.bool_:
            new_indices = jnp.where(mask)[0]
        else:
            new_indices = mask

        # If we are already a subset, map the new indices through the existing ones
        if self._indices is not None:
            new_indices = self._indices[new_indices]

        return eqx.tree_at(lambda d: d._indices, self, new_indices)

    def __getattr__(self, name: str) -> Any:
        """Retrieves a column array by resolving virtual overrides, then disk proxies.

        Attribute access includes appended rows and respects the dataset's current
        filter. Proxy reads may load data from disk and populate backend caches.

        ### Usage Example
        ```python
        import jax.numpy as jnp
        import pandas as pd
        from flowtangent.utils.data import DataFrameDataset

        dataset = DataFrameDataset(
            pd.DataFrame({"x": [0.0, 1.0, 2.0]})
        )
        subset = dataset.filter(jnp.array([2]))
        assert subset.x.shape == (1,)
        ```

        Args:
            name (str): Column name; names starting with `_` use ordinary attribute lookup.

        Returns:
            Any: The column value, including appended rows and any row selection.
                Array columns may be ND, with rows along axis 0.

        Raises:
            AttributeError: If the variable cannot be found in the dataset.
            TypeError: If appended rows have incompatible trailing dimensions.
            OSError: If an underlying file read fails; backend errors propagate.
        """
        if name.startswith("_"):
            return object.__getattribute__(self, name)

        # 1. Fetch the base column (from virtual columns or the backend proxy)
        base_col = None
        if hasattr(self, "_virtual_columns") and name in self._virtual_columns:
            base_col = self._virtual_columns[name]
        elif hasattr(self, "_proxy") and self._proxy is not _:
            try:
                base_col = getattr(self._proxy, name)
            except AttributeError:
                pass

        if base_col is None:
            raise AttributeError(f"Dataset has no variable '{name}'")

        # 2. Concatenate any appended active-learning rows to the end
        if hasattr(self, "_appended_data") and name in self._appended_data:
            base_col = jnp.concatenate([jnp.asarray(base_col), self._appended_data[name]], axis=0)

        if self._indices is not None:
            
            # 1. Convert JAX array to standard NumPy array
            idx_np = np.asarray(self._indices)
            
            # 2. Handle Xarray DataArrays specifically
            if hasattr(base_col, "values"):
                arr = base_col.values 
                
                # 3. Handle virtual stacking by unraveling the indices
                if hasattr(self, "_proxy") and getattr(self._proxy, "_virtual_stack", None):
                    stack_dims = self._proxy._virtual_stack
                    dim_sizes = tuple(self._proxy._ds.sizes[d] for d in stack_dims)
                    
                    # Unravel the flat index array into N-D coordinate arrays
                    # e.g., returns (array([0, 1, ...]), array([41, 0, ...]))
                    unraveled = np.unravel_index(idx_np, dim_sizes)
                    idx_map = dict(zip(stack_dims, unraveled))
                    
                    # Slice using ONLY the dimensions this specific variable has!
                    # For panel_data: slices (run, alpha). For foil_name: slices just (run).
                    actual_indices = tuple(idx_map[d] for d in base_col.dims if d in idx_map)
                    
                    return arr[actual_indices]
                
                # Fallback for standard 1D Datasets
                return arr[idx_np]
                
            # Fallback for standard NumPy/JAX arrays (e.g., appended active-learning data)
            return base_col[idx_np]

        return base_col

    # --- PyTorch DataLoader Compatibility ---
    def __len__(self) -> int:
        """Returns the total number of accessible rows in the dataset."""
        if self._indices is not None:
            return len(self._indices)
        return object.__getattribute__(self, "_len") + self._appended_len

    def __getitem__(self, idx: int | slice) -> dict[str, Any]:
        """Retrieves a row dict (or sliced column dict) from the dataset.

        Row indices refer to the current filtered view, including appended rows.
        Integer access uses backend row reads; slices may load entire columns
        and populate backend caches.

        ### Usage Example
        ```python
        import pandas as pd
        from flowtangent.utils.data import DataFrameDataset

        dataset = DataFrameDataset(
            pd.DataFrame({"x": [0.0, 1.0, 2.0]})
        )
        assert dataset[-1]["x"] == 2.0
        assert dataset[:2]["x"].shape == (2,)
        ```

        Args:
            idx (int | slice): Row index or slice along axis 0. Negative integers
                are offset by `len(self)` once. Bounds handling is delegated to
                the backend arrays; there is no uniform out-of-range check.

        Returns:
            dict[str, Any]: Values for every name in `full_vars`. Integer access
                removes the row axis; slices retain it for array columns.

        Raises:
            IndexError: If an underlying backend rejects the row index.
            KeyError: If an appended row has no data for a requested column.
            AttributeError: If a column cannot be resolved by the backend.
            OSError: If an underlying file read fails; backend errors propagate.
        """
        # Fast-path for single integer indexing (PyTorch DataLoader)
        if isinstance(idx, int):
            row = {}
            total_len = len(self)
            if idx < 0:
                idx += total_len

            actual_idx = idx if self._indices is None else self._indices[idx]
            orig_len = object.__getattribute__(self, "_len")

            for var in self.full_vars:
                # 1. Is it an active-learning row?
                if actual_idx >= orig_len:
                    row[var] = self._appended_data[var][actual_idx - orig_len]

                # 2. Is it a virtual column?
                elif var in self._virtual_columns:
                    row[var] = self._virtual_columns[var][actual_idx]

                # 3. It must be on disk
                else:
                    row[var] = self._proxy.get_row(var, actual_idx)
        else:
            # Fallback for slices
            row = {}
            for var in self.full_vars:
                row[var] = getattr(self, var)[idx]

        for tf in self._transforms:
            row = tf(row)
        
        return row


class StreamingDataset(IterableDataset):
    """An out-of-core IterableDataset that yields pre-collated matrices directly.

    Args:
        dataset (Dataset): The underlying FlowTangent Dataset instance.
        in_names (list[str]): Column names to stack into the X_batch matrix.
        out_names (list[str]): Column names to stack into the Y_batch matrix.
        batch_size (int): The number of rows to slice from disk per read.
        shuffle (bool): Whether to randomize chunk order and rows.
    """
    def __init__(
        self, 
        dataset, 
        in_names: list[str], 
        out_names: list[str], 
        batch_size: int = 16384, 
        shuffle: bool = True,
        read_cols: Optional[list[str]] = None
    ):
        super().__init__()
        self.dataset = dataset
        self.in_names = in_names
        self.out_names = out_names
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.total_rows = len(dataset)

        if read_cols is not None:
            self.read_cols = read_cols
        else:
            self.read_cols = dataset.full_vars


    def __len__(self) -> int:
        return math.ceil(self.total_rows / self.batch_size)

    def _extract(self, lazy_arr, indices, actual_dims=None):
        """Backend-agnostic router for lazy array evaluation."""
        
        # Fast Path: Contiguous 1D Slice (Supported by literally everything)
        if not isinstance(indices, tuple) and len(indices) > 0:
            if indices[-1] - indices[0] == len(indices) - 1:
                res = lazy_arr[indices[0] : indices[-1] + 1]
                if hasattr(res, "compute"): return res.compute()
                if hasattr(res, "values"): return res.values
                return np.asarray(res)
            
        # Try Xarray Native Selection
        if hasattr(lazy_arr, "isel"):
            if isinstance(indices, tuple) and actual_dims is not None:
                # ND Unraveled Fancy Indexing
                idx_dict = {dim: xr.DataArray(idx) for dim, idx in zip(actual_dims, indices)}
                res = lazy_arr.isel(**idx_dict)
            else:
                # 1D Fancy Indexing
                res = lazy_arr.isel({lazy_arr.dims[0]: xr.DataArray(indices)})
                
            if hasattr(res, "compute"): return res.compute()
            if hasattr(res, "values"): return res.values
            return np.asarray(res)
            
        # Try Dask Native Selection
        if hasattr(lazy_arr, "vindex"):
            res = lazy_arr.vindex[indices]
            if hasattr(res, "compute"): return res.compute()
            return np.asarray(res)

        # Try Universal Python Fancy Indexing
        try:
            res = lazy_arr[indices]
            if hasattr(res, "compute"): return res.compute()
            if hasattr(res, "values"): return res.values
            return np.asarray(res)
        except Exception:
            pass # Backend rejected standard fancy indexing
            
        # Bulletproof Fallback: Bounded memory block fetch (for strict Zarr/HDF5)
        if not isinstance(indices, tuple):
            min_idx, max_idx = indices[0], indices[-1]
            block = lazy_arr[min_idx : max_idx + 1]
            
            if hasattr(block, "compute"): block = block.compute()
            elif hasattr(block, "values"): block = block.values
            else: block = np.asarray(block)
            
            # Localize indices and slice in memory
            return block[indices - min_idx]
            
        raise TypeError(f"Could not extract ND indices from backend array of type {type(lazy_arr)}")

    def __iter__(self):
        chunk_starts = list(range(0, self.total_rows, self.batch_size))
        if self.shuffle:
            np.random.shuffle(chunk_starts)
            
        proxy = getattr(self.dataset, "_proxy", None)
        has_proxy = proxy is not None and getattr(proxy, "_compute_engine", None) is not None
        virtual_stack = getattr(proxy, "_virtual_stack", None) if has_proxy else None
        
        orig_len = getattr(self.dataset, "_len", self.total_rows)
        indices = getattr(self.dataset, "_indices", None)
        virtuals = getattr(self.dataset, "_virtual_columns", {})
        appended = getattr(self.dataset, "_appended_data", {})

        @contextmanager
        def get_backend_arrays():
            if has_proxy:
                with proxy._compute_engine(chunk_size=self.batch_size) as arrs:
                    yield arrs
            elif proxy is not None:
                # Fallback: Yield raw proxy columns if no compute engine exists
                yield {k: getattr(proxy, k) for k in dir(proxy) if not k.startswith("_")}
            else:
                yield {}

        with get_backend_arrays() as backend_arrays:
            for start in chunk_starts:
                end = min(start + self.batch_size, self.total_rows)
                
                # 1. Resolve flat indices
                if indices is not None:
                    batch_indices = np.asarray(indices[start:end])
                else:
                    batch_indices = np.arange(start, end)
                    
                base_mask = batch_indices < orig_len
                app_mask = batch_indices >= orig_len
                
                base_idx = batch_indices[base_mask]
                app_idx = batch_indices[app_mask] - orig_len
                
                raw_dict = {}
                for name in self.read_cols:
                    base_data = None
                    
                    # --- A. In-Memory Virtuals ---
                    if name in virtuals:
                        if len(base_idx) > 0:
                            base_data = virtuals[name][base_idx]
                            
                    # --- B. Out-of-Core Backend OR Proxy Reads ---
                    else:
                        lazy_arr = None
                        if name in backend_arrays:
                            lazy_arr = backend_arrays[name]
                        elif has_proxy and hasattr(proxy, name):
                            # Fallback for coordinates/variables the compute engine skipped
                            lazy_arr = getattr(proxy, name)
                            
                        if lazy_arr is not None and len(base_idx) > 0:
                            # Handle ND XArray/Dask unraveling
                            if virtual_stack is not None:
                                dim_sizes = tuple(proxy._ds.sizes[d] for d in virtual_stack)
                                unraveled = np.unravel_index(base_idx, dim_sizes)
                                idx_map = dict(zip(virtual_stack, unraveled))
                                
                                # raw_col = getattr(proxy, name)
                                actual_dims = getattr(lazy_arr, "dims", None)
                                
                                if actual_dims is not None:
                                    actual_indices = tuple(idx_map[d] for d in actual_dims if d in idx_map)
                                    base_data = self._extract(lazy_arr, actual_indices, actual_dims)
                                else:
                                    base_data = self._extract(lazy_arr, base_idx)
                            else:
                                base_data = self._extract(lazy_arr, base_idx)
                                
                    # --- C. Active-Learning & Recombination ---
                    app_data = appended[name][app_idx] if (len(app_idx) > 0 and name in appended) else None
                        
                    if base_data is not None and app_data is not None:
                        raw_dict[name] = np.concatenate([base_data, app_data], axis=0)
                    elif base_data is not None:
                        raw_dict[name] = base_data
                    elif app_data is not None:
                        raw_dict[name] = app_data
                        
                # 2. Functional Transforms & Stacking
                for tf in getattr(self.dataset, "_transforms", ()):
                    raw_dict = tf(raw_dict)
                    
                X_batch = jnp.column_stack([jnp.asarray(raw_dict[name]) for name in self.in_names])
                Y_batch = jnp.column_stack([jnp.asarray(raw_dict[name]) for name in self.out_names])
                
                # 3. Micro-Shuffle
                if self.shuffle:
                    perm = np.random.permutation(X_batch.shape[0])
                    X_batch, Y_batch = X_batch[perm], Y_batch[perm]
                    
                yield X_batch, Y_batch


def load_dataset(source: Union[str, Path, Any], *args, **kwargs) -> Dataset:
    """Intelligently detects a data source format and initializes the correct `Dataset` subclass.

    Supports reading natively from `.zarr`, `.h5`/`.hdf5`, `.parquet`, `.csv`,
    `.nc`/`.netcdf`, and `.mat` files, as well as Pandas or Polars DataFrames.

    Args:
        source (Union[str, Path, Any]): A file path or DataFrame object.
        *args: Additional positional arguments passed to the specific `Dataset` implementation.
        **kwargs: Additional keyword arguments passed to the specific `Dataset` implementation.

    Raises:
        ValueError: If instantiated with `None` or an unsupported file extension.
        TypeError: If the source object is not recognized (e.g. not a path or DataFrame).

    Returns:
        Dataset: The initialized, type-specific subclass of `ft.Dataset`.
    """

    if source is None:
        raise ValueError("ft.Dataset requires a source when instantiated directly.")

    if isinstance(source, (str, Path)):
        src_str = str(source).lower()
        ext = src_str.split(".")[-1]

        ext_dict = {
            "zarr": ZarrDataset,
            "h5": HDF5Dataset,
            "hdf5": HDF5Dataset,
            "parquet": ParquetDataset,
            "csv": CSVDataset,
            "nc": NetCDFDataset,
            "nc4": NetCDFDataset,
            "netcdf": NetCDFDataset,
            "mat": MATDataset,
        }

        if ext in ext_dict:
            # Instantiate manually to bypass Equinox initialization strictness
            return ext_dict[ext](source, *args, **kwargs)
        else:
            raise ValueError(f"Unsupported data format extension: {source}")

    elif hasattr(source, "columns") and hasattr(source, "iloc"):
        return DataFrameDataset(source, *args, **kwargs)

    raise TypeError(f"Unsupported data source type: {type(source)}")


# ----------------------------------------------------------------------------------------------------------------------
# Dataset Proxies
# ----------------------------------------------------------------------------------------------------------------------


class MemoryProxy:
    """A generic proxy layer for tabular formats loaded fully into memory (CSV, MAT, DataFrames)."""

    def __init__(self, data_dict: dict):
        self._data = data_dict

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"MemoryProxy has no attribute '{name}'")

    def __dir__(self):
        return list(self._data.keys())

    def get_row(self, name: str, idx: int):
        if name not in self._data:
            raise AttributeError(f"MemoryProxy has no attribute '{name}'")
        return self._data[name][idx]

    @contextmanager
    def _compute_engine(self, **kwargs):
        yield self._data


class ParquetProxy:
    """A lazy-evaluation proxy layer for Parquet files utilizing dynamic column caching."""

    def __init__(self, file_path: str | Path, columns: list):
        self._file_path = file_path
        self._columns = columns
        # Cache to store columns as they are requested
        self._column_cache = {}

    def get_row(self, name: str, idx: int):
        if name not in self._columns:
            raise AttributeError(f"ParquetProxy has no column '{name}'")

        # Load the column from disk only if it hasn't been accessed yet
        if name not in self._column_cache:
            table = pq.read_table(self._file_path, columns=[name])
            self._column_cache[name] = table.column(name)

        return self._column_cache[name][idx].as_py()

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError
        if name in self._columns:
            if name not in self._column_cache:
                table = pq.read_table(self._file_path, columns=[name])
                self._column_cache[name] = table.column(name)
            return self._column_cache[name].to_numpy()
        raise AttributeError(f"ParquetProxy has no column '{name}'")

    def __dir__(self):
        return self._columns

    @contextmanager
    def _compute_engine(self, **kwargs):
        df = pl.scan_parquet(self._file_path)
        yield df

class ZarrGroupProxy:
    """Lazy evaluation proxy for chunked arrays within Zarr storage architectures."""

    def __init__(self, zarr_path: str, group_path: str = ""):
        self._zarr_path = zarr_path
        self._group_path = group_path

    def _get_zarr_obj(self):
        root = zarr.open_group(self._zarr_path, mode="r")
        return root[self._group_path] if self._group_path else root

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

        obj = self._get_zarr_obj()
        if name not in obj:
            raise AttributeError(f"Zarr group '{self._group_path or 'root'}' has no subgroup or array '{name}'.")

        target = obj[name]
        next_path = f"{self._group_path}/{name}" if self._group_path else name

        if isinstance(target, zarr.Group):
            return ZarrGroupProxy(self._zarr_path, next_path)
        return target

    def __dir__(self):
        return list(self._get_zarr_obj().keys())

    def get_row(self, name: str, idx: int):
        obj = self._get_zarr_obj()
        if name not in obj:
            raise AttributeError(f"Zarr group '{self._group_path or 'root'}' has no array '{name}'.")
        # Slicing the Zarr array object reads only that specific chunk from disk
        return obj[name][idx]

    @contextmanager
    def _compute_engine(self, chunk_size: int = 1000, **kwargs):
        obj = self._get_zarr_obj()
        arrays = {}
        
        # Build a dictionary of Dask arrays to bypass Xarray metadata requirements
        for key in obj.keys():
            if isinstance(obj[key], zarr.Array):
                # Chunk along the primary axis (0), leaving trailing dimensions whole
                chunks = (chunk_size,) + obj[key].shape[1:]
                arrays[key] = da.from_zarr(obj[key], chunks=chunks)
                
        yield arrays

class HDF5GroupProxy:
    """Lazy evaluation proxy for arrays within HDF5 files."""

    def __init__(self, file_path: str | Path, group_path: str = ""):
        self._file_path = str(file_path)
        self._group_path = group_path

    def _get_obj(self, f):
        return f[self._group_path] if self._group_path else f

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError

        with h5py.File(self._file_path, "r") as f:
            group = self._get_obj(f)
            if name not in group:
                raise AttributeError

            target = group[name]
            next_path = f"{self._group_path}/{name}" if self._group_path else name

            if isinstance(target, h5py.Group):
                return HDF5GroupProxy(self._file_path, next_path)
            elif isinstance(target, h5py.Dataset):
                return target

    def __dir__(self):
        with h5py.File(self._file_path, "r") as f:
            group = self._get_obj(f)
            return list(group.keys())

    def get_row(self, name: str, idx: int):
        with h5py.File(self._file_path, "r") as f:
            group = self._get_obj(f)
            if name not in group:
                raise AttributeError
            # Target is an h5py.Dataset, slicing it reads only that index from disk
            return group[name][idx]

    @contextmanager
    def _compute_engine(self, chunk_size: int = 1000, **kwargs):
        with h5py.File(self._file_path, "r") as f:
            group = self._get_obj(f)
            arrays = {}
            for key in group.keys():
                if isinstance(group[key], h5py.Dataset):
                    chunks = (chunk_size,) + group[key].shape[1:]
                    arrays[key] = da.from_array(group[key], chunks=chunks)
            yield arrays

class NetCDFGroupProxy:
    """Lazy evaluation proxy for arrays within NetCDF4 files."""

    def __init__(self, file_path: str | Path, group_path: str = ""):
        self._file_path = file_path
        self._group_path = group_path

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError

        with nc.Dataset(self._file_path, "r") as ds:
            grp = ds[self._group_path] if self._group_path else ds

            if name in grp.groups:
                next_path = f"{self._group_path}/{name}" if self._group_path else name
                return NetCDFGroupProxy(self._file_path, next_path)
            elif name in grp.variables:
                return grp.variables[name]
            raise AttributeError

    def __dir__(self):
        with nc.Dataset(self._file_path, "r") as ds:
            grp = ds[self._group_path] if self._group_path else ds
            return list(grp.groups.keys()) + list(grp.variables.keys())

    def get_row(self, name: str, idx: int):
        with nc.Dataset(self._file_path, "r") as ds:
            grp = ds[self._group_path] if self._group_path else ds
            if name not in grp.variables:
                raise AttributeError
            # Slicing the netCDF variable reads only that index from disk
            return grp.variables[name][idx]

    @contextmanager
    def _compute_engine(self, chunk_size: int = 1000, **kwargs):
        with nc.Dataset(self._file_path, "r") as ds:
            grp = ds[self._group_path] if self._group_path else ds
            arrays = {}
            for key in grp.variables.keys():
                chunks = (chunk_size,) + grp.variables[key].shape[1:]
                arrays[key] = da.from_array(grp.variables[key], chunks=chunks)
            yield arrays

class XArrayProxy:
    """Lazy evaluation proxy utilizing Xarray for labeled N-dimensional coordinates."""

    def __init__(self, file_path: str | Path, engine: str = "zarr", **kwargs):
        self._file_path = file_path
        self._engine = engine
        self._kwargs = kwargs
        
        self._ds = xr.open_dataset(file_path, engine=engine, chunks=None, **kwargs)
        
        # Build a routing map to flatten dimension coordinates into tabular variables
        self._var_map = {}
        
        for var_name, da in self._ds.data_vars.items():
            # 1. Always expose the top-level macro array itself (e.g., 'polar_data')
            self._var_map[var_name] = var_name
            
            # 2. Inspect dimensions to see if they have string coordinate labels
            for dim in da.dims:
                if dim in self._ds.coords:
                    coord_vals = self._ds.coords[dim].values
                    # If the coordinate contains string labels (like 'CL', 'CD', 'Mach')
                    if coord_vals.dtype.kind in ('U', 'S', 'O'):
                        for label in coord_vals:
                            # Map the label to its parent array and dimension
                            # e.g., 'CL' -> ('polar_data', 'polar_var')
                            self._var_map[str(label)] = (var_name, dim)

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError
            
        if name not in self._var_map:
            raise AttributeError(f"XArrayProxy has no variable or coordinate label '{name}'")
            
        mapping = self._var_map[name]
        
        # If it's a sub-variable mapped to a dimension label, use Xarray's .sel()
        if isinstance(mapping, tuple):
            array_name, dim_name = mapping
            return self._ds[array_name].sel({dim_name: name})
            
        # Otherwise, return the top-level macro array
        return self._ds[mapping]

    def __dir__(self):
        return list(self._ds.data_vars.keys()) + list(self._var_map.keys())

    def get_row(self, name: str, idx: int):
        if name not in self._var_map:
            raise AttributeError(f"XArrayProxy has no variable '{name}'")
            
        mapping = self._var_map[name]
        
        # Resolve the target lazy array
        if isinstance(mapping, tuple):
            array_name, dim_name = mapping
            target_array = self._ds[array_name].sel({dim_name: name})
        else:
            target_array = self._ds[mapping]

        # --- VIRTUAL STACKING SLICE ---
        if hasattr(self, "_virtual_stack") and self._virtual_stack:
            # 1. Get the sizes of the stacked dimensions (e.g., [358400, 42])
            dim_sizes = tuple(self._ds.sizes[d] for d in self._virtual_stack)
            
            # 2. Unravel the flat index (e.g., 43 -> (1, 1))
            unraveled_indices = np.unravel_index(idx, dim_sizes)
            
            # 3. Create a dictionary of the slice mapping: {'run': 1, 'alpha': 1}
            isel_kwargs = dict(zip(self._virtual_stack, unraveled_indices))
            
            # 4. Filter the slice down to only the dimensions this specific array actually has!
            # E.g. foil_name only has 'run', so we safely ignore the 'alpha' index.
            valid_isel_kwargs = {k: v for k, v in isel_kwargs.items() if k in target_array.dims}
            
            val = target_array.isel(**valid_isel_kwargs).values
            
        # --- STANDARD 1D SLICE ---
        else:
            first_dim = list(target_array.dims)[0]
            val = target_array.isel({first_dim: idx}).values
            
        # Handle Xarray MultiIndex coordinate scalar wrapping
        if isinstance(val, tuple): 
            return np.array(val)
        return val
    
    @contextmanager
    def _compute_engine(self, chunk_size: int = 1000, **kwargs):
        # 1. Grab the primary dimension name dynamically to use as the chunking axis
        first_var = list(self._ds.data_vars.values())[0]
        primary_dim = list(first_var.dims)[0]
        
        # 2. Combine base kwargs with any passed specifically to the compute engine
        compute_kwargs = {**self._kwargs, **kwargs}
        
        # 3. Open a fresh Xarray dataset natively mapped to Dask
        lazy_ds = xr.open_dataset(
            self._file_path, 
            engine=self._engine, 
            chunks={primary_dim: chunk_size}, 
            **compute_kwargs
        )
        
        try:
            yield lazy_ds
        finally:
            lazy_ds.close()

    def _build_var_map(self):
        self._var_map = {}
        
        # 1. Map explicit dimensions/coordinates first (e.g., the integer indices)
        for coord_name in self._ds.coords:
            self._var_map[coord_name] = coord_name

        # 2. Map MultiIndex levels (e.g., 'run', 'alpha' after stacking)
        for coord_name, index in self._ds.indexes.items():
            if isinstance(index, pd.MultiIndex):
                for level_name in index.names:
                    self._var_map[level_name] = level_name

        # 3. Map Features last so they win any name collisions
        # If polar_data has a feature named "alpha", it overwrites the integer dimension index.
        for var_name, da in self._ds.data_vars.items():
            self._var_map[var_name] = var_name
            for dim in da.dims:
                if dim in self._ds.coords:
                    coord_vals = self._ds.coords[dim].values
                    if coord_vals.dtype.kind in ('U', 'S', 'O') and not isinstance(self._ds.indexes.get(dim), pd.MultiIndex):
                        for label in coord_vals:
                            self._var_map[str(label)] = (var_name, dim)

    def create_stacked_view(self, stack_dims: list[str], new_dim_name: str) -> "XArrayProxy":
        """Creates a new proxy with a virtual Cartesian view using index unraveling."""
        new_proxy = copy.copy(self)
        
        # Store the dimensions we are virtually stacking
        new_proxy._virtual_stack = stack_dims
        
        # Calculate the new total length (e.g., 358400 * 42)
        new_proxy._stacked_len = math.prod(self._ds.sizes[d] for d in stack_dims)
        
        return new_proxy

# ----------------------------------------------------------------------------------------------------------------------
# Datasets
# ----------------------------------------------------------------------------------------------------------------------

class ZarrDataset(Dataset):
    """Dataset implementation explicitly targeting `.zarr` directories.

    Args:
        source (str): File path to the `.zarr` directory.
        primary_array (str, optional): The canonical array used to derive dataset row length.
            Defaults to the first array found in the root if empty.
    """

    _len: int = _

    def __init__(self, source: str, primary_array: str = ""):
        object.__setattr__(self, "_source", source)
        object.__setattr__(self, "_proxy", ZarrGroupProxy(source))
        object.__setattr__(self, "_virtual_columns", {})

        root = zarr.open_group(source, mode="r")
        # Use first array if primary_array isn't specified
        primary_array = primary_array or list(root.keys())[0]
        object.__setattr__(self, "_len", root[primary_array].shape[0])


class DataFrameDataset(Dataset):
    """Dataset implementation explicitly targeting in-memory Pandas or Polars DataFrames.

    Args:
        df (Any): The DataFrame instance.
    """

    _len: int = _

    def __init__(self, df):
        columns = list(df.columns)
        data_arr = df.to_numpy(dtype=np.float32)

        data_dict = {col: data_arr[:, i] for i, col in enumerate(columns)}

        object.__setattr__(self, "_source", "dataframe")
        object.__setattr__(self, "_proxy", MemoryProxy(data_dict))
        object.__setattr__(self, "_virtual_columns", {})
        object.__setattr__(self, "_len", len(data_arr))


class HDF5Dataset(Dataset):
    """Dataset implementation explicitly targeting `.h5`/`.hdf5` files.

    Args:
        file_path (str): File path to the `.h5` file.
        primary_key (str, optional): The canonical array used to derive dataset row length.
            Defaults to "data" or the first dataset found if "data" is missing.
    """

    _len: int = _

    def __init__(self, file_path: str | Path, primary_key: str = "data"):
        with h5py.File(file_path, "r") as f:
            if primary_key not in f:
                datasets = [k for k in f.keys() if isinstance(f[k], h5py.Dataset)]
                if not datasets:
                    raise KeyError(f"No datasets found in HDF5 root: {list(f.keys())}")
                primary_key = datasets[0]
            length = f[primary_key].shape[0]

        object.__setattr__(self, "_source", file_path)
        object.__setattr__(self, "_proxy", HDF5GroupProxy(file_path))
        object.__setattr__(self, "_virtual_columns", {})
        object.__setattr__(self, "_len", length)


class ParquetDataset(Dataset):
    """Dataset implementation explicitly targeting `.parquet` files.

    Args:
        file_path (str): File path to the `.parquet` file.
    """

    _len: int = _

    def __init__(self, file_path: str | Path):
        pf = pq.ParquetFile(file_path)

        object.__setattr__(self, "_source", file_path)
        object.__setattr__(self, "_proxy", ParquetProxy(file_path, pf.schema.names))
        object.__setattr__(self, "_virtual_columns", {})
        object.__setattr__(self, "_len", pf.metadata.num_rows)


class ArrayDataset(Dataset):
    """Direct JAX replacement for PyTorch's `TensorDataset`.

    Requires all provided arrays to possess the same length in their first dimension.
    Does not use the string-based proxy routing of tabular datasets.

    Args:
        *arrays: Positional arguments of matching-length JAX or NumPy arrays.
    """

    arrays: tuple = _

    def __init__(self, *arrays):
        assert all(arrays[0].shape[0] == array.shape[0] for array in arrays), "Size mismatch"
        object.__setattr__(self, "arrays", arrays)
        # Bypassing _source/_proxy entirely since we use positional indexing

    def __getitem__(self, index: int | slice) -> tuple:
        """Retrieves an index or slice positionally across all internal arrays."""
        return tuple(array[index] for array in self.arrays)

    def __len__(self) -> int:
        """Returns the shared length of the arrays."""
        return self.arrays[0].shape[0]


class CSVDataset(Dataset):
    """Dataset implementation explicitly targeting `.csv` files.

    Utilizes PyArrow to swiftly read the CSV into memory.

    Args:
        file_path (str): File path to the `.csv` file.
    """

    _len: int = _

    def __init__(self, file_path: str | Path):
        table = pcsv.read_csv(file_path)
        columns = table.column_names
        data_dict = {col: table.column(col).to_numpy() for col in columns}

        object.__setattr__(self, "_source", file_path)
        object.__setattr__(self, "_proxy", MemoryProxy(data_dict))
        object.__setattr__(self, "_virtual_columns", {})
        object.__setattr__(self, "_len", table.num_rows)


class NetCDFDataset(Dataset):
    """Dataset implementation explicitly targeting `.nc`/`.nc4`/`.netcdf` files.

    Args:
        file_path (str): File path to the `.nc` file.
        primary_variable (str, optional): The canonical array used to derive dataset row length.
            Defaults to the first variable found if empty.
    """

    _len: int = _

    def __init__(self, file_path: str, primary_variable: str = None):
        with nc.Dataset(file_path, "r") as ds:
            vars_list = list(ds.variables.keys())
            primary_variable = primary_variable or vars_list[0]
            length = ds.variables[primary_variable].shape[0]

        object.__setattr__(self, "_source", file_path)
        object.__setattr__(self, "_proxy", NetCDFGroupProxy(file_path))
        object.__setattr__(self, "_virtual_columns", {})
        object.__setattr__(self, "_len", length)


class MATDataset(Dataset):
    """Dataset implementation explicitly targeting older `.mat` files.

    Attempts to load via `scipy.io` first, falling back to HDF5 mechanisms
    if the `.mat` file was saved in newer formats.

    Args:
        file_path (str): File path to the `.mat` file.
        primary_key (str, optional): The canonical array used to derive dataset row length.
    """

    _len: int = _

    def __init__(self, file_path: str | Path, primary_key: str = None):
        data_dict = {}
        try:
            mat = sio.loadmat(file_path)
            data_dict = {k: v for k, v in mat.items() if not k.startswith("__")}
        except NotImplementedError:
            with h5py.File(file_path, "r") as f:
                for k in f.keys():
                    if not k.startswith("#"):
                        data_dict[k] = f[k][...]

        if not data_dict:
            raise ValueError(f"No valid variables found in MAT file '{file_path}'")

        keys = list(data_dict.keys())
        primary_key = primary_key or keys[0]

        object.__setattr__(self, "_source", file_path)
        object.__setattr__(self, "_proxy", MemoryProxy(data_dict))
        object.__setattr__(self, "_virtual_columns", {})
        object.__setattr__(self, "_len", len(data_dict[primary_key]))


class XArrayDataset(Dataset):
    """Dataset implementation explicitly targeting datasets with complex dimensional coordinates."""
    
    _len: int = _

    def __init__(self, file_path: str | Path, engine: str = "zarr", _proxy: Any = None, **kwargs):
        # ds = xr.open_dataset(file_path, engine=engine)
        
        object.__setattr__(self, "_source", file_path)
        object.__setattr__(self, "_engine", engine)
        object.__setattr__(self, "_virtual_columns", {})

        if not _proxy:
            object.__setattr__(self, "_proxy", XArrayProxy(file_path, engine))
        else:
            object.__setattr__(self, "_proxy", _proxy)
        
        # Deduce length from the size of the first dimension of the first variable
        first_var = list(self._proxy._ds.data_vars.values())[0]
        first_dim = list(first_var.dims)[0]
        object.__setattr__(self, "_len", self._proxy._ds.sizes[first_dim])

    def to_tabular(self, stack_dims: list[str], new_dim_name: str = "row") -> "XArrayDataset":
        """Transforms an N-dimensional dataset into a flat, tabular view via virtual indexing."""
        stacked_proxy = self._proxy.create_stacked_view(stack_dims, new_dim_name)
        
        new_ds = replace(self, file_path=self._source, engine=self._engine, _proxy=stacked_proxy)
        return eqx.tree_at(lambda d: d._len, new_ds, stacked_proxy._stacked_len)

    def dropna(self, columns: list[str], chunk_size: int = 10_000, **compute_kwargs) -> "Dataset":
        """Returns a new Dataset with NaN rows filtered out using the out-of-core compute engine."""

        with self._proxy._compute_engine(chunk_size=chunk_size, **compute_kwargs) as lazy_ds:
            global_valid_mask = None
            
            # 1. Identify the axes that define our "rows"
            if hasattr(self._proxy, "_virtual_stack") and self._proxy._virtual_stack:
                row_dims = self._proxy._virtual_stack
            else:
                first_var = list(lazy_ds.data_vars.values())[0]
                row_dims = [list(first_var.dims)[0]]

            for col in columns:
                # 2. Resolve the actual backend variable name from the proxy map
                if hasattr(self._proxy, "_var_map") and col in self._proxy._var_map:
                    mapping = self._proxy._var_map[col]
                    real_name = mapping[0] if isinstance(mapping, tuple) else mapping
                else:
                    real_name = col
                    
                col_data = lazy_ds[real_name]
                
                # 3. Reduce over any dimension that ISN'T a row dimension (e.g., panels, vars)
                reduce_dims = [d for d in col_data.dims if d not in row_dims]
                
                if reduce_dims:
                    col_valid = ~col_data.isnull().any(dim=reduce_dims)
                else:
                    col_valid = ~col_data.isnull()
                    
                # Xarray's '&' operator automatically broadcasts mismatched dimensions!
                # (e.g., if one column only has 'run' and another has 'run' + 'alpha')
                if global_valid_mask is None:
                    global_valid_mask = col_valid
                else:
                    global_valid_mask = global_valid_mask & col_valid
                
            # 4. Compute the mask in its original N-Dimensional shape (e.g., [N_run, N_alpha])
            with TqdmCallback(desc="Dropping NaNs from dataset"):
                concrete_mask = np.asarray(global_valid_mask.compute())
            
        # 5. Flatten the N-D mask! This perfectly mirrors np.unravel_index in get_row()
        flat_mask = concrete_mask.ravel()
        
        valid_indices = jnp.array(np.where(flat_mask)[0])
        return self.filter(valid_indices)


__all__ = [
    "Dataset",
    "load_dataset",
    "IterableDataset",
    "StreamingDataset"
    "ArrayDataset",
    "StackDataset",
    "ConcatDataset",
    "ChainDataset",
    "Subset",
    "random_split",
    "DataLoader",
    "LatentDataLoader",
    "get_worker_info",
    "slice_data",
    "numpy_collate",
    "Sampler",
    "BatchSampler",
    "RandomSampler",
    "LoaderType",
    "SequentialSampler",
    "SubsetRandomSampler",
    "WeightedRandomSampler",
]