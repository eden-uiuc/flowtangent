import warnings
from pathlib import Path
from typing import Any, Optional, Union

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import netCDF4 as nc
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

#-----------------------------------------------------------------------------------------------------------------------
# Helper Functions
#-----------------------------------------------------------------------------------------------------------------------

def numpy_collate(batch):
    """
    Recursively converts to NumPy and stacks the batch. 
    Completely replaces PyTorch's default_collate to ensure downstream 
    JAX functions never accidentally receive a torch.Tensor.
    """
    elem = batch[0]

    if isinstance(elem, torch.Tensor):
        return np.stack([x.detach().cpu().numpy() for x in batch])
    elif isinstance(elem, np.ndarray):
        return np.stack(batch)
    elif isinstance(elem, (int, float)):
        return np.array(batch)
    elif isinstance(elem, (tuple, list)):
        transposed = zip(*batch)
        return type(elem)(numpy_collate(samples) for samples in transposed)
    elif isinstance(elem, dict):
        return {key: numpy_collate([d[key] for d in batch]) for key in elem}
    else:
        raise TypeError(f"numpy_collate cannot handle batch of type {type(elem)}")


class DataLoader(TorchDataLoader):
    """
    A standalone facade wrapper around PyTorch's DataLoader.
    Defaults to `numpy_collate` to ensure batches are returned as pure NumPy arrays 
    (or dictionaries of arrays) instead of PyTorch Tensors.
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, collate_fn=numpy_collate, **kwargs):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            **kwargs
        )

class LatentDataLoader:
    """
    Wraps a physical DataLoader to encode X_batch into Z_batch on the fly.
    Duck-types as a standard DataLoader for downstream training loops.
    """
    def __init__(self, dataloader, manifold):
        self.dataloader = dataloader
        self.manifold = manifold
        self.dataset = dataloader.dataset  # Satisfies duck-typing checks

    def __iter__(self):
        for x_batch, y_batch in self.dataloader:
            # Encode physical batch to latent batch
            z_batch = jax.vmap(self.manifold.encode)(x_batch)
            yield z_batch, y_batch

    def __len__(self):
        return len(self.dataloader)

class SlicedDataLoader:
    """
    Dynamically slices the feature dimension of batches streaming from a DataLoader.
    Duck-types as a standard DataLoader for neural training loops.
    """
    def __init__(self, dataloader, split_indices, chunk_idx):
        self.dataloader = dataloader
        self.split_indices = split_indices
        self.chunk_idx = chunk_idx
        self.dataset = getattr(dataloader, "dataset", None)

    def __iter__(self):
        for batch in self.dataloader:
            if isinstance(batch, (tuple, list)):
                x_batch, y_batch = batch
                x_parts = jnp.split(x_batch, self.split_indices, axis=-1)
                yield x_parts[self.chunk_idx], y_batch
            else:
                x_parts = jnp.split(batch, self.split_indices, axis=-1)
                yield x_parts[self.chunk_idx]

    def __len__(self):
        return len(self.dataloader)

def slice_data(data, split_indices, chunk_idx):
    """Routes data slicing depending on if it's a full array or a DataLoader."""
    if hasattr(data, "__iter__") and hasattr(data, "dataset"):
        return SlicedDataLoader(data, split_indices, chunk_idx)
    # If it's a raw tuple of (X, Y) or a raw X array
    if isinstance(data, (tuple, list)):
        return jnp.split(data[0], split_indices, axis=-1)[chunk_idx], data[1]
    return jnp.split(data, split_indices, axis=-1)[chunk_idx]

LoaderType = Union[DataLoader, LatentDataLoader, SlicedDataLoader]

#-----------------------------------------------------------------------------------------------------------------------
# Dataset
#-----------------------------------------------------------------------------------------------------------------------

class Dataset(Module):
    """
    Universal FlowTangent Dataset factory.
    Duck-types as a PyTorch Dataset (via __len__ and __getitem__) 
    while remaining a valid Equinox PyTree.
    """
    _source: Any = static_field(_)
    _proxy: Any = static_field(_)
    _virtual_columns: dict[str, jax.Array] = _
    _indices: Optional[jax.Array] = _

    def __new__(cls, source=None, *args, **kwargs):
        if cls is not Dataset:
            return super().__new__(cls)

        if source is None:
            raise ValueError("ft.Dataset requires a source when instantiated directly.")

        if isinstance(source, (str, Path)):
            src_str = str(source).lower()
            ext = src_str.split('.')[-1]

            ext_dict = {
                "zarr": ZarrDataset,
                "h5": HDF5Dataset,
                "hdf5": HDF5Dataset,
                "parquet": ParquetDataset,
                "csv": CSVDataset,
                "nc": NetCDFDataset,
                "nc4": NetCDFDataset,
                "netcdf": NetCDFDataset,
                "mat": MATDataset
            }

            if ext in ext_dict:
                # Instantiate manually to bypass Equinox initialization strictness
                proxy_cls = ext_dict[ext]
                instance = proxy_cls.__new__(proxy_cls)
                instance.__init__(source, *args, **kwargs)
                return instance
            else:
                raise ValueError(f"Unsupported data format extension: {source}")

        elif hasattr(source, "columns") and hasattr(source, "iloc"):
            instance = DataFrameDataset.__new__(DataFrameDataset)
            instance.__init__(source, *args, **kwargs)
            return instance

        raise TypeError(f"Unsupported data source type: {type(source)}")

    def __check_init__(self):
        super(Dataset, self).__check_init__()
        if getattr(self, "_virtual_columns", _) is _:
            object.__setattr__(self, "_virtual_columns", {})
        if getattr(self, "_appended_data", _) is _:
            object.__setattr__(self, "_appended_data", {})
        if getattr(self, "_indices", _) is _:
            object.__setattr__(self, "_indices", None)

    @property
    def full_vars(self) -> list[str]:
        # Implementation for crawling proxy + virtual columns (from previous step)
        backend_keys = []
        if self._proxy is not _:
            backend_keys = [k for k in dir(self._proxy) if not k.startswith("_")]
        virtual_keys = list(self._virtual_columns.keys())
        return sorted(list(set(backend_keys + virtual_keys)))

    def add_variable(self, name: str, data: Optional[jax.Array] = None, quiet: bool = False) -> "Dataset":
        """
        Registers a new variable in the virtual overlay. 
        Emits a warning to prevent silent typo bugs unless quiet=True.
        """
        if not quiet:
            warnings.warn(
                f"Creating virtual dataset column for new variable '{name}'. "
                "If this is a typo, check your Parameter names.",
                UserWarning,
                stacklevel=2
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
        """
        Appends a new row to the dataset via the virtual overlay.
        Missing variables are automatically filled with NaNs.
        Returns a functionally updated Dataset.
        """
        new_appended = dict(self._appended_data)

        for var in self.full_vars:
            # Extract the new value, defaulting to NaN if the oracle didn't provide it
            val = jnp.atleast_1d(row_dict.get(var, jnp.nan))

            if var in new_appended:
                new_appended[var] = jnp.concatenate([new_appended[var], val], axis=0)
            else:
                new_appended[var] = val

        return eqx.tree_at(
            lambda d: (d._appended_data, d._appended_len),
            self,
            (new_appended, self._appended_len + 1)
        )

    def filter(self, mask: jax.Array) -> "Dataset":
        """
        Creates a zero-copy view of the dataset using a boolean mask or index array.
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

    def __getattr__(self, name: str):
        if name.startswith("_"):
            return object.__getattribute__(self, name)

        # 1. Fetch the base column (from virtual columns or the backend proxy)
        base_col = None
        if hasattr(self, "_virtual_columns") and name in self._virtual_columns:
            base_col = self._virtual_columns[name]
        elif hasattr(self, "_backend_proxy") and self._backend_proxy is not _:
            try:
                base_col = getattr(self._backend_proxy, name)
            except AttributeError:
                pass

        if base_col is None:
            raise AttributeError(f"Dataset has no variable '{name}'")

        # 2. Concatenate any appended active-learning rows to the end
        if hasattr(self, "_appended_data") and name in self._appended_data:
            return jnp.concatenate([jnp.asarray(base_col), self._appended_data[name]], axis=0)

        return base_col

    # --- PyTorch DataLoader Compatibility ---
    def __len__(self):
        if self._indices is not None:
            return len(self._indices)
        return object.__getattribute__(self, "_len") + self._appended_len

    def __getitem__(self, idx):
        # Route the requested index through the virtual view
        if self._indices is not None:
            idx = self._indices[idx]

        row = {}
        for var in self.full_vars:
            row[var] = getattr(self, var)[idx]
        return row

#-----------------------------------------------------------------------------------------------------------------------
# Universal Proxy Helpers
#-----------------------------------------------------------------------------------------------------------------------

class MemoryProxy:
    """A generic proxy for tabular formats loaded fully into memory (CSV, MAT, DataFrames)."""
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


class ParquetProxy:
    """Lazy-evaluation proxy for Parquet files."""
    def __init__(self, file_path: str, columns: list):
        self._file_path = file_path
        self._columns = columns

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError
        if name in self._columns:
            table = pq.read_table(self._file_path, columns=[name])
            return table.column(name).to_numpy()
        raise AttributeError(f"ParquetProxy has no column '{name}'")

    def __dir__(self):
        return self._columns

#-----------------------------------------------------------------------------------------------------------------------
# Dataset Implementations
#-----------------------------------------------------------------------------------------------------------------------

# Zarr -----------------------------------------------------------------------------------------------------------------

class ZarrGroupProxy:
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

        if isinstance(target, zarr.hierarchy.Group):
            return ZarrGroupProxy(self._zarr_path, next_path)
        return target[:]

    def __dir__(self):
        return list(self._get_zarr_obj().keys())


class ZarrDataset(Dataset):
    _len: int = _

    def __init__(self, source: str, primary_array: str = ""):
        object.__setattr__(self, "_source", source)
        object.__setattr__(self, "_backend_proxy", ZarrGroupProxy(source))
        object.__setattr__(self, "_virtual_columns", {})

        root = zarr.open_group(source, mode="r")
        # Use first array if primary_array isn't specified
        primary_array = primary_array or list(root.keys())[0]
        object.__setattr__(self, "_len", root[primary_array].shape[0])

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        row = {}
        for var in self.full_vars:
            row[var] = getattr(self, var)[idx]
        return row

# DataFrames (Pandas/Polars) -------------------------------------------------------------------------------------------

class DataFrameDataset(Dataset):
    _len: int = _

    def __init__(self, df):
        columns = list(df.columns)
        data_arr = df.to_numpy(dtype=np.float32)

        data_dict = {col: data_arr[:, i] for i, col in enumerate(columns)}

        object.__setattr__(self, "_source", "dataframe")
        object.__setattr__(self, "_backend_proxy", MemoryProxy(data_dict))
        object.__setattr__(self, "_virtual_columns", {})
        object.__setattr__(self, "_len", len(data_arr))

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        row = {}
        for var in self.full_vars:
            row[var] = getattr(self, var)[idx]
        return row

# HDF5 -----------------------------------------------------------------------------------------------------------------

class HDF5GroupProxy:
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


class HDF5Dataset(Dataset):
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
        object.__setattr__(self, "_backend_proxy", HDF5GroupProxy(file_path))
        object.__setattr__(self, "_virtual_columns", {})
        object.__setattr__(self, "_len", length)

    def __len__(self):
        return self._len

    def __getitem__(self, idx: int):
        row = {}
        for var in self.full_vars:
            row[var] = getattr(self, var)[idx]
        return row

# Parquet --------------------------------------------------------------------------------------------------------------

class ParquetDataset(Dataset):
    _len: int = _

    def __init__(self, file_path: str):
        pf = pq.ParquetFile(file_path)

        object.__setattr__(self, "_source", file_path)
        object.__setattr__(self, "_backend_proxy", ParquetProxy(file_path, pf.schema.names))
        object.__setattr__(self, "_virtual_columns", {})
        object.__setattr__(self, "_len", pf.metadata.num_rows)

    def __len__(self):
        return self._len

    def __getitem__(self, idx: int):
        row = {}
        for var in self.full_vars:
            row[var] = getattr(self, var)[idx]
        return row

# Tensor/Arrays --------------------------------------------------------------------------------------------------------

class ArrayDataset(Dataset):
    """
    Direct JAX replacement for PyTorch's TensorDataset.
    Does not use the string-based proxy routing of the tabular datasets.
    """
    arrays: tuple = _

    def __init__(self, *arrays):
        assert all(arrays[0].shape[0] == array.shape[0] for array in arrays), "Size mismatch"
        object.__setattr__(self, "arrays", arrays)
        # Bypassing _source/_backend_proxy entirely since we use positional indexing

    def __getitem__(self, index):
        return tuple(array[index] for array in self.arrays)

    def __len__(self):
        return self.arrays[0].shape[0]

# CSV ------------------------------------------------------------------------------------------------------------------

class CSVDataset(Dataset):
    _len: int = _

    def __init__(self, file_path: str):
        table = pcsv.read_csv(file_path)
        columns = table.column_names
        data_dict = {col: table.column(col).to_numpy() for col in columns}

        object.__setattr__(self, "_source", file_path)
        object.__setattr__(self, "_backend_proxy", MemoryProxy(data_dict))
        object.__setattr__(self, "_virtual_columns", {})
        object.__setattr__(self, "_len", table.num_rows)

    def __len__(self):
        return self._len

    def __getitem__(self, idx: int):
        row = {}
        for var in self.full_vars:
            row[var] = getattr(self, var)[idx]
        return row

# NetCDF ---------------------------------------------------------------------------------------------------------------

class NetCDFGroupProxy:
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


class NetCDFDataset(Dataset):
    _len: int = _

    def __init__(self, file_path: str, primary_variable: str = None):
        with nc.Dataset(file_path, "r") as ds:
            vars_list = list(ds.variables.keys())
            primary_variable = primary_variable or vars_list[0]
            length = ds.variables[primary_variable].shape[0]

        object.__setattr__(self, "_source", file_path)
        object.__setattr__(self, "_backend_proxy", NetCDFGroupProxy(file_path))
        object.__setattr__(self, "_virtual_columns", {})
        object.__setattr__(self, "_len", length)

    def __len__(self):
        return self._len

    def __getitem__(self, idx: int):
        row = {}
        for var in self.full_vars:
            row[var] = getattr(self, var)[idx]
        return row

# MATLAB ---------------------------------------------------------------------------------------------------------------

class MATDataset(Dataset):
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
        object.__setattr__(self, "_backend_proxy", MemoryProxy(data_dict))
        object.__setattr__(self, "_virtual_columns", {})
        object.__setattr__(self, "_len", len(data_dict[primary_key]))

    def __len__(self):
        return self._len

    def __getitem__(self, idx: int):
        row = {}
        for var in self.full_vars:
            row[var] = getattr(self, var)[idx]
        return row

__all__ = [
    "Dataset", "IterableDataset", "ArrayDataset", "StackDataset", "ConcatDataset",
    "ChainDataset", "Subset", "random_split", "DataLoader", "LatentDataLoader", "get_worker_info",
    "slice_data", "numpy_collate", "Sampler", "BatchSampler", "RandomSampler", "LoaderType"
    "SequentialSampler", "SubsetRandomSampler", "WeightedRandomSampler"
]
