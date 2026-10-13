"""Synthetic data tests for collation, loaders, views, and file proxies."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import h5py
import jax.numpy as jnp
import netCDF4 as nc  # noqa: N813
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import scipy.io as sio
import torch
import zarr

from flowtangent.utils import data as data_utils
from flowtangent.utils.data import (
    ArrayDataset,
    DataFrameDataset,
    DataLoader,
    Dataset,
    LatentDataLoader,
    MATDataset,
    MemoryProxy,
    ParquetProxy,
    SlicedDataLoader,
    load_dataset,
    numpy_collate,
    slice_data,
)


class _OverlayDataset(Dataset):
    """Minimal in-memory Dataset using Equinox's generated initializer."""

    _len: int = 0


@pytest.fixture
def dataset():
    return DataFrameDataset(pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [10.0, 11.0, 12.0]}))


@pytest.fixture
def overlay_dataset():
    return _OverlayDataset(_proxy=MemoryProxy({"x": np.arange(3.0), "y": np.arange(10.0, 13.0)}), _len=3)


@pytest.mark.parametrize("make_array", [np.asarray, jnp.asarray, torch.tensor])
def test_numpy_collate_stacks_array_samples_into_numpy(make_array):
    batch = [make_array([[1.0, 2.0]]), make_array([[3.0, 4.0]])]
    result = numpy_collate(batch)

    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 1, 2)
    np.testing.assert_allclose(result, [[[1, 2]], [[3, 4]]])


def test_numpy_collate_detaches_tensors_requiring_gradients():
    tensor = torch.tensor([1.0, 2.0], requires_grad=True)

    np.testing.assert_allclose(numpy_collate([tensor, tensor * 2]), [[1, 2], [2, 4]])
    assert tensor.requires_grad
    assert tensor.grad is None


@pytest.mark.parametrize("samples", [[1, 2], [1.5, 2.5], [True, False], [np.int32(1), np.int32(2)]])
def test_numpy_collate_handles_numeric_and_boolean_scalars(samples):
    result = numpy_collate(samples)

    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, samples)


def test_numpy_collate_preserves_nested_container_types():
    samples = [
        {"features": (np.array([1.0]), [2, True])},
        {"features": (np.array([3.0]), [4, False])},
    ]

    result = numpy_collate(samples)

    assert isinstance(result, dict)
    assert isinstance(result["features"], tuple)
    assert isinstance(result["features"][1], list)
    np.testing.assert_allclose(result["features"][0], [[1], [3]])
    np.testing.assert_allclose(result["features"][1][0], [2, 4])
    np.testing.assert_array_equal(result["features"][1][1], [True, False])


def test_numpy_collate_uses_first_dictionary_keys_and_shortest_sequence():
    result = numpy_collate([{"x": [1, 2]}, {"x": [3], "extra": 4}])

    assert set(result) == {"x"}
    assert len(result["x"]) == 1
    np.testing.assert_allclose(result["x"][0], [1, 3])


@pytest.mark.parametrize(
    ("batch", "error"),
    [
        ([], IndexError),
        (["unsupported"], TypeError),
        ([np.ones(2), np.ones(3)], ValueError),
        ([{"x": 1}, {}], KeyError),
    ],
)
def test_numpy_collate_rejects_invalid_batches(batch, error):
    with pytest.raises(error):
        numpy_collate(batch)


def test_dataloader_defaults_to_numpy_and_keeps_partial_final_batch(dataset):
    loader = DataLoader(dataset, batch_size=2, num_workers=0)
    batches = list(loader)

    assert len(loader) == 2
    assert len(batches) == 2
    assert all(isinstance(batch["x"], np.ndarray) for batch in batches)
    np.testing.assert_allclose(batches[0]["x"], [0, 1])
    np.testing.assert_allclose(batches[1]["x"], [2])


def test_dataloader_honors_custom_collation_and_drop_last(dataset):
    collate = Mock(return_value="custom batch")
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate, drop_last=True, num_workers=0)

    assert list(loader) == ["custom batch"]
    collate.assert_called_once()
    assert len(collate.call_args.args[0]) == 2


def test_latent_loader_encodes_each_sample_and_preserves_targets():
    features = jnp.arange(12.0).reshape(3, 4)
    targets = jnp.array([10.0, 20.0, 30.0])
    source = DataLoader(ArrayDataset(features, targets), batch_size=2, num_workers=0)
    loader = LatentDataLoader(source, SimpleNamespace(encode=lambda sample: sample[:2] * 2))
    batches = list(loader)

    assert loader.dataset is source.dataset
    assert len(loader) == len(source) == 2
    np.testing.assert_allclose(batches[0][0], [[0, 2], [8, 10]])
    np.testing.assert_allclose(batches[0][1], [10, 20])
    np.testing.assert_allclose(batches[1][0], [[16, 18]])
    np.testing.assert_allclose(batches[1][1], [30])


@pytest.mark.parametrize("pair_type", [tuple, list])
def test_slice_data_pair_preserves_target_identity_and_splits_last_axis(pair_type):
    features = jnp.arange(24).reshape(2, 3, 4)
    targets = object()

    chunk, actual_targets = slice_data(pair_type((features, targets)), 2, -1)

    np.testing.assert_allclose(chunk, [[[2, 3], [6, 7], [10, 11]], [[14, 15], [18, 19], [22, 23]]])
    assert actual_targets is targets


def test_slice_data_explicit_split_positions_allow_empty_chunks():
    features = jnp.arange(12).reshape(2, 6)

    np.testing.assert_allclose(slice_data(features, [1, 4], 1), [[1, 2, 3], [7, 8, 9]])
    assert slice_data(features, [0, 6], 0).shape == (2, 0)


@pytest.mark.parametrize(
    "split,chunk,error", [(4, 0, ValueError), (2, 2, IndexError), (2, -3, IndexError), (0, 0, ZeroDivisionError)]
)
def test_slice_data_rejects_invalid_split_or_chunk(split, chunk, error):
    with pytest.raises(error):
        slice_data(jnp.ones((2, 6)), split, chunk)


def test_slice_data_rejects_pair_without_targets():
    with pytest.raises(IndexError):
        slice_data((jnp.ones((2, 4)),), 2, 0)


@pytest.mark.parametrize("paired", [False, True])
def test_sliced_loader_preserves_batch_count_dataset_and_targets(paired):
    features = np.arange(12).reshape(3, 4)
    arrays = (features, np.array([10, 20, 30])) if paired else (features,)
    # PyTorch returns a tuple for ArrayDataset; use an array-only stream for the other branch.
    if paired:
        source = DataLoader(ArrayDataset(*arrays), batch_size=2, num_workers=0)
    else:
        source = DataLoader(list(features), batch_size=2, num_workers=0)
    loader = slice_data(source, 2, 1)

    assert isinstance(loader, SlicedDataLoader)
    assert loader.dataset is source.dataset
    assert len(loader) == len(source) == 2
    batches = list(loader)
    np.testing.assert_allclose(batches[0][0] if paired else batches[0], [[2, 3], [6, 7]])
    np.testing.assert_allclose(batches[1][0] if paired else batches[1], [[10, 11]])
    if paired:
        np.testing.assert_allclose(batches[0][1], [10, 20])


def test_sliced_loader_defers_invalid_split_until_iteration():
    loader = slice_data(DataLoader(list(np.ones((2, 6))), num_workers=0), 4, 0)

    with pytest.raises(ValueError, match="array split does not result in an equal division"):
        next(iter(loader))


def test_dataset_column_and_row_access_matches_sorted_variables(dataset):
    assert dataset.full_vars == ["x", "y"]
    assert len(dataset) == 3
    np.testing.assert_allclose(dataset.x, [0, 1, 2])
    np.testing.assert_allclose(dataset[-1]["y"], 12)
    np.testing.assert_allclose(dataset[1:]["x"], [1, 2])
    with pytest.raises(AttributeError, match="Dataset has no variable 'missing'"):
        _ = dataset.missing
    with pytest.raises(AttributeError):
        _ = dataset._missing


def test_add_variable_warns_and_returns_independent_nan_column(dataset):
    with pytest.warns(UserWarning, match="Creating virtual dataset column.*'new'"):
        changed = dataset.add_variable("new")

    np.testing.assert_allclose(changed.new, [np.nan, np.nan, np.nan], equal_nan=True)
    assert changed.full_vars == ["new", "x", "y"]
    assert dataset.full_vars == ["x", "y"]


def test_add_variable_can_override_proxy_column_without_mutating_source(dataset):
    changed = dataset.add_variable("x", jnp.array([5.0, 6.0, 7.0]), quiet=True)

    np.testing.assert_allclose(changed.x, [5, 6, 7])
    np.testing.assert_allclose(changed[1]["x"], 6)
    np.testing.assert_allclose(dataset.x, [0, 1, 2])
    assert changed.full_vars == ["x", "y"]


def test_add_variable_rejects_length_mismatch(dataset):
    with pytest.raises(ValueError, match="Data length 2 != dataset length 3"):
        dataset.add_variable("new", jnp.zeros(2), quiet=True)


@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="Dataset constructors leave _appended_len outside instance leaves; append fails in eqx.tree_at",
)
def test_dataframe_append_exposes_new_row_without_mutating_source(dataset):
    changed = dataset.append({"x": 3.0, "y": 13.0})

    assert len(changed) == 4
    np.testing.assert_allclose(changed.x, [0, 1, 2, 3])
    assert len(dataset) == 3


def test_append_pads_missing_columns_and_keeps_original_unchanged(overlay_dataset):
    dataset = overlay_dataset
    changed = dataset.append({"x": 3.0}).append({"x": 4.0, "y": 14.0})

    assert len(changed) == 5
    np.testing.assert_allclose(changed.x, [0, 1, 2, 3, 4])
    np.testing.assert_allclose(changed.y, [10, 11, 12, np.nan, 14], equal_nan=True)
    np.testing.assert_allclose(changed[-1]["y"], 14)
    assert len(dataset) == 3
    assert dataset._appended_data == {}


def test_append_preserves_multidimensional_virtual_columns(overlay_dataset):
    dataset = overlay_dataset
    original = dataset.add_variable("vector", jnp.arange(6.0).reshape(3, 2), quiet=True)
    changed = original.append({"vector": [6.0, 7.0]})

    np.testing.assert_allclose(changed.vector, [[0, 1], [2, 3], [4, 5], [6, 7]])
    np.testing.assert_allclose(changed[-1]["vector"], [6, 7])
    assert original.vector.shape == (3, 2)


@pytest.mark.parametrize("mask", [jnp.array([True, False, True]), jnp.array([2, 0])])
def test_filter_respects_boolean_or_integer_selection_without_mutation(dataset, mask):
    filtered = dataset.filter(mask)
    expected = [0, 2] if mask.dtype == jnp.bool_ else [2, 0]

    assert len(filtered) == 2
    np.testing.assert_allclose(filtered.x, expected)
    np.testing.assert_allclose(filtered[:]["x"], expected)
    np.testing.assert_allclose(filtered[-1]["x"], expected[-1])
    np.testing.assert_allclose(dataset.x, [0, 1, 2])


def test_chained_filter_maps_indices_through_existing_view(dataset):
    filtered = dataset.filter(jnp.array([2, 0])).filter(jnp.array([False, True]))

    assert len(filtered) == 1
    np.testing.assert_allclose(filtered.x, [0])
    np.testing.assert_allclose(filtered[0]["y"], 10)


def test_append_to_filtered_dataset_exposes_new_row_at_end_of_view(overlay_dataset):
    dataset = overlay_dataset
    changed = dataset.filter(jnp.array([2, 0])).append({"x": 3.0, "y": 13.0})

    assert len(changed) == 3
    np.testing.assert_allclose(changed.x, [2, 0, 3])
    np.testing.assert_allclose(changed[-1]["y"], 13)
    assert len(dataset) == 3


def test_empty_filter_retains_column_shapes(dataset):
    empty = dataset.filter(jnp.array([False, False, False]))

    assert len(empty) == 0
    assert empty.x.shape == (0,)
    assert empty[:]["y"].shape == (0,)


def test_array_dataset_indexes_arrays_positionally_and_retains_slices():
    dataset = ArrayDataset(jnp.arange(6).reshape(3, 2), np.array([10, 20, 30]))

    assert len(dataset) == 3
    assert isinstance(dataset[-1], tuple)
    np.testing.assert_allclose(dataset[-1][0], [4, 5])
    np.testing.assert_allclose(dataset[1:][1], [20, 30])
    assert dataset[:0][0].shape == (0, 2)


def test_array_dataset_accepts_zero_rows():
    assert len(ArrayDataset(jnp.empty((0, 2)), jnp.empty((0,)))) == 0


def test_array_dataset_rejects_inconsistent_lengths():
    with pytest.raises(AssertionError, match="Size mismatch"):
        ArrayDataset(jnp.zeros((2, 3)), jnp.zeros(3))


@pytest.mark.parametrize("source,error", [(None, ValueError), ("file.unknown", ValueError), (object(), TypeError)])
def test_load_dataset_rejects_unsupported_sources(source, error):
    with pytest.raises(error):
        load_dataset(source)


@pytest.mark.parametrize(
    ("extension", "class_name"),
    [
        ("zarr", "ZarrDataset"),
        ("h5", "HDF5Dataset"),
        ("hdf5", "HDF5Dataset"),
        ("parquet", "ParquetDataset"),
        ("csv", "CSVDataset"),
        ("nc", "NetCDFDataset"),
        ("nc4", "NetCDFDataset"),
        ("netcdf", "NetCDFDataset"),
        ("mat", "MATDataset"),
    ],
)
def test_load_dataset_routes_extensions_case_insensitively_and_forwards_args(monkeypatch, extension, class_name):
    constructor = Mock()
    monkeypatch.setattr(data_utils, class_name, constructor)
    source = Path(f"synthetic.{extension.upper()}")

    result = load_dataset(source, "primary", example=True)

    constructor.assert_called_once_with(source, "primary", example=True)
    assert result is constructor.return_value


def test_load_dataset_accepts_in_memory_dataframe():
    result = load_dataset(pd.DataFrame({"x": [1.0, 2.0]}))

    assert isinstance(result, DataFrameDataset)
    np.testing.assert_allclose(result.x, [1, 2])


def test_memory_proxy_exposes_columns_and_rejects_missing_names():
    values = np.array([1, 2])
    proxy = MemoryProxy({"x": values})

    assert dir(proxy) == ["x"]
    assert proxy.x is values
    assert proxy.get_row("x", -1) == 2
    with pytest.raises(AttributeError):
        proxy.get_row("missing", 0)
    with pytest.raises(AttributeError):
        _ = proxy.missing


def test_parquet_proxy_loads_only_requested_column_once(monkeypatch):
    table = pa.table({"x": [1.0, 2.0]})
    read = Mock(return_value=table)
    monkeypatch.setattr(data_utils.pq, "read_table", read)
    proxy = ParquetProxy("synthetic.parquet", ["x", "y"])

    read.assert_not_called()
    assert proxy.get_row("x", 1) == 2.0
    np.testing.assert_allclose(proxy.x, [1, 2])
    read.assert_called_once_with("synthetic.parquet", columns=["x"])
    assert dir(proxy) == ["x", "y"]
    with pytest.raises(AttributeError, match="no column 'missing'"):
        proxy.get_row("missing", 0)
    read.assert_called_once()


@pytest.fixture(params=["csv", "parquet", "h5", "zarr", "nc", "mat"])
def tabular_file(tmp_path, request):
    """Small synthetic files exercise readers without external data or services."""
    path = tmp_path / f"sample.{request.param}"
    values = np.array([1.0, 2.0, 3.0])
    if request.param == "csv":
        path.write_text("x,y\n1,10\n2,20\n3,30\n")
    elif request.param == "parquet":
        pq.write_table(pa.table({"x": values, "y": values * 10}), path)
    elif request.param == "h5":
        with h5py.File(path, "w") as file:
            file.create_dataset("x", data=values)
            file.create_dataset("y", data=values * 10)
    elif request.param == "zarr":
        group = zarr.open_group(str(path), mode="w")
        group.create_array("x", data=values)
        group.create_array("y", data=values * 10)
    elif request.param == "nc":
        with nc.Dataset(path, "w") as file:
            file.createDimension("row", 3)
            file.createVariable("x", "f8", ("row",))[:] = values
            file.createVariable("y", "f8", ("row",))[:] = values * 10
    else:
        sio.savemat(path, {"x": values[:, None], "y": (values * 10)[:, None]})
    return path


def test_file_datasets_load_columns_rows_and_non_destructive_overlays(tabular_file):
    dataset = load_dataset(str(tabular_file))

    assert len(dataset) == 3
    assert dataset.full_vars == ["x", "y"]
    np.testing.assert_allclose(np.asarray(dataset.x).reshape(-1), [1, 2, 3])
    np.testing.assert_allclose(np.asarray(dataset[-1]["y"]).reshape(-1), [30])

    changed = dataset.add_variable("virtual", jnp.array([4.0, 5.0, 6.0]), quiet=True)

    assert len(changed) == 3
    np.testing.assert_allclose(changed.virtual, [4, 5, 6])
    assert changed.full_vars == ["virtual", "x", "y"]
    reloaded = load_dataset(str(tabular_file))
    assert len(reloaded) == 3
    assert reloaded.full_vars == ["x", "y"]
    np.testing.assert_allclose(np.asarray(reloaded.x).reshape(-1), [1, 2, 3])


def test_mat_dataset_ignores_metadata_and_rejects_files_without_variables(monkeypatch):
    monkeypatch.setattr(data_utils.sio, "loadmat", lambda path: {"__header__": "synthetic"})

    with pytest.raises(ValueError, match="No valid variables found"):
        MATDataset("synthetic.mat")


def test_mat_dataset_falls_back_to_hdf5_for_new_format(tmp_path, monkeypatch):
    path = tmp_path / "new.mat"
    with h5py.File(path, "w") as file:
        file.create_dataset("x", data=np.array([[1.0], [2.0]]))
        file.create_dataset("#metadata", data=np.array([99]))
    monkeypatch.setattr(data_utils.sio, "loadmat", Mock(side_effect=NotImplementedError))

    dataset = MATDataset(str(path))

    assert dataset.full_vars == ["x"]
    assert len(dataset) == 2
    np.testing.assert_allclose(dataset.x, [[1], [2]])
