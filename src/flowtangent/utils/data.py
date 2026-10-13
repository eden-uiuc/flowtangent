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

import warnings
from pathlib import Path
from typing import Any, Optional, Union

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import netCDF4 as nc  # noqa: N813
import numpy as np
import pyarrow.csv as pcsv
import pyarrow.parquet as pq
import scipy.io as sio
import torch
import zarr
from torch.utils.data import (
    BatchSampler,
    ChainDataset,
    ConcatDataset,
    # Core Datasets
    IterableDataset,
    RandomSampler,
    # Samplers
    Sampler,
    SequentialSampler,
    StackDataset,
    Subset,
    SubsetRandomSampler,
    WeightedRandomSampler,
    # Utils
    get_worker_info,
    random_split,
)
from torch.utils.data import DataLoader as TorchDataLoader

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

    ### Mathematical Formulation
    For array samples $x_i$ of shape $S$, the result has shape $(B, *S)$ and
    $C[i] = x_i$, where $B$ is the batch size.

    ### Process Architecture
    Convert array or scalar samples, or recursively collate matching container
    entries. Sequence entries are paired only up to the shortest sample length;
    dictionary keys come from the first sample.

    ### Usage Example
    ```python
    from flowtangent.utils.data import numpy_collate

    batch = numpy_collate([{"x": 1.0}, {"x": 2.0}])
    assert batch["x"].shape == (2,)
    ```

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

        ### Mathematical Formulation
        For base column $C$, appended rows $A$, and optional indices $I$, return
        $concat(C, A)[I]$, omitting concatenation or indexing when absent.

        ### Process Architecture
        Resolve the virtual column or proxy value, append any overlay rows, then
        apply the current row indices.

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
            return base_col[self._indices]

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

        ### Mathematical Formulation
        For current-view columns $C_v$, the returned mapping is $R[v] = C_v[idx]$.

        ### Process Architecture
        Map integer indices through the current view and select appended, virtual,
        or proxy rows. For slices, retrieve current-view columns before slicing.

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
            return row

        # Fallback for slices
        row = {}
        for var in self.full_vars:
            row[var] = getattr(self, var)[idx]
        return row


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
# Universal Proxy Helpers
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


class ParquetProxy:
    """A lazy-evaluation proxy layer for Parquet files utilizing dynamic column caching."""

    def __init__(self, file_path: str, columns: list):
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


# ----------------------------------------------------------------------------------------------------------------------
# Dataset Implementations
# ----------------------------------------------------------------------------------------------------------------------

# Zarr -----------------------------------------------------------------------------------------------------------------


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
        return target[:]

    def __dir__(self):
        return list(self._get_zarr_obj().keys())

    def get_row(self, name: str, idx: int):
        obj = self._get_zarr_obj()
        if name not in obj:
            raise AttributeError(f"Zarr group '{self._group_path or 'root'}' has no array '{name}'.")
        # Slicing the Zarr array object reads only that specific chunk from disk
        return obj[name][idx]


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


# DataFrames (Pandas/Polars) -------------------------------------------------------------------------------------------


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


# HDF5 -----------------------------------------------------------------------------------------------------------------


class HDF5GroupProxy:
    """Lazy evaluation proxy for arrays within HDF5 files."""

    def __init__(self, file_path: str, group_path: str = ""):
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
                return target[...]

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


class HDF5Dataset(Dataset):
    """Dataset implementation explicitly targeting `.h5`/`.hdf5` files.

    Args:
        file_path (str): File path to the `.h5` file.
        primary_key (str, optional): The canonical array used to derive dataset row length.
            Defaults to "data" or the first dataset found if "data" is missing.
    """

    _len: int = _

    def __init__(self, file_path: str, primary_key: str = "data"):
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


# Parquet --------------------------------------------------------------------------------------------------------------


class ParquetDataset(Dataset):
    """Dataset implementation explicitly targeting `.parquet` files.

    Args:
        file_path (str): File path to the `.parquet` file.
    """

    _len: int = _

    def __init__(self, file_path: str):
        pf = pq.ParquetFile(file_path)

        object.__setattr__(self, "_source", file_path)
        object.__setattr__(self, "_proxy", ParquetProxy(file_path, pf.schema.names))
        object.__setattr__(self, "_virtual_columns", {})
        object.__setattr__(self, "_len", pf.metadata.num_rows)


# Tensor/Arrays --------------------------------------------------------------------------------------------------------


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


# CSV ------------------------------------------------------------------------------------------------------------------


class CSVDataset(Dataset):
    """Dataset implementation explicitly targeting `.csv` files.

    Utilizes PyArrow to swiftly read the CSV into memory.

    Args:
        file_path (str): File path to the `.csv` file.
    """

    _len: int = _

    def __init__(self, file_path: str):
        table = pcsv.read_csv(file_path)
        columns = table.column_names
        data_dict = {col: table.column(col).to_numpy() for col in columns}

        object.__setattr__(self, "_source", file_path)
        object.__setattr__(self, "_proxy", MemoryProxy(data_dict))
        object.__setattr__(self, "_virtual_columns", {})
        object.__setattr__(self, "_len", table.num_rows)


# NetCDF ---------------------------------------------------------------------------------------------------------------


class NetCDFGroupProxy:
    """Lazy evaluation proxy for arrays within NetCDF4 files."""

    def __init__(self, file_path: str, group_path: str = ""):
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
                return grp.variables[name][:]
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


# MATLAB ---------------------------------------------------------------------------------------------------------------


class MATDataset(Dataset):
    """Dataset implementation explicitly targeting older `.mat` files.

    Attempts to load via `scipy.io` first, falling back to HDF5 mechanisms
    if the `.mat` file was saved in newer formats.

    Args:
        file_path (str): File path to the `.mat` file.
        primary_key (str, optional): The canonical array used to derive dataset row length.
    """

    _len: int = _

    def __init__(self, file_path: str, primary_key: str = None):
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


__all__ = [
    "Dataset",
    "load_dataset",
    "IterableDataset",
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
