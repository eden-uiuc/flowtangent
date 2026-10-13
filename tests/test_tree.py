"""Immutable tree updates, equality, deltas, and diagnostic output."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flowtangent.utils.tree import (
    TreePath,
    apply_tree_delta,
    combine,
    compute_tree_delta,
    get_actual_path,
    get_all_parents,
    get_all_targets,
    get_parent_target,
    get_target,
    id_partition,
    inspect_leaves,
    is_equivalent,
    scan_for_invalid_JAX_types,
    update,
)


class Branch(eqx.Module):
    value: Any
    label: str = eqx.field(static=True, default="Main Wing")

    @property
    def field_name(self):
        return self.label.replace(" ", "_").lower()


class Root(eqx.Module):
    subcomponents: tuple
    counter: int = 0

    def __getattr__(self, name):
        if name == "main_wing":
            return self.subcomponents[0]
        raise AttributeError(name)


@pytest.fixture
def root():
    return Root(subcomponents=(Branch(jnp.arange(6).reshape(2, 3)),))


@pytest.mark.parametrize(
    ("spec", "expected_path", "expected_value"),
    [("branch.value", ("branch", "value"), 7), (("branch.value", 3), ("branch", "value"), 3)],
)
def test_tree_path_casts_paths_and_update_pairs(spec, expected_path, expected_value):
    path = TreePath.cast(spec, default_val=7)

    assert path.path == expected_path
    assert path.value == expected_value
    assert path.name == "branch.value"
    assert len(path) == 2
    assert path.as_item() == ("branch.value", expected_value)


def test_tree_path_preserves_slice_and_existing_instance():
    path = TreePath.cast(("branch.value", 8, slice(1, 3)))

    assert path.path_slice == slice(1, 3)
    assert TreePath.cast(path, default_val=99) is path
    assert TreePath(path).path_slice == slice(1, 3)
    with pytest.raises(FrozenInstanceError):
        path.name = "changed"


def test_tree_path_integer_indices_have_readable_names():
    path = TreePath(("subcomponents", 0, "value"), name=None)

    assert path.name == "subcomponents[0].value"
    assert path.path == ("subcomponents", 0, "value")


@pytest.mark.parametrize("spec", [None, 42, [], ({"bad": "path"}, 1), ("value", 1, object())])
def test_tree_path_cast_rejects_invalid_specs(spec):
    with pytest.raises(TypeError, match="Cannot automatically cast"):
        TreePath.cast(spec)


@pytest.mark.parametrize("path", [42, ["value"], ("value", object())])
def test_tree_path_constructor_rejects_invalid_path_elements(path):
    with pytest.raises(ValueError):
        TreePath(path)


def test_get_targets_distinguishes_parent_from_sliced_value(root):
    path = TreePath(("subcomponents", 0, "value"), path_slice=(slice(None), 1))

    assert get_parent_target(root, path) is root.subcomponents[0].value
    np.testing.assert_allclose(get_target(root, path), [1, 4])
    parents = get_all_parents(root, [path, "counter"])
    targets = get_all_targets(root, [path, "counter"])
    assert parents[0] is root.subcomponents[0].value
    assert parents[1] == targets[1] == 0
    np.testing.assert_allclose(targets[0], [1, 4])
    assert get_all_targets(root, []) == ()


def test_get_target_traverses_dictionaries_and_sequence_indices():
    leaf = jnp.array([2.0, 4.0])
    tree = {"items": [{"value": leaf}]}
    path = TreePath(("items", 0, "value"))

    assert get_target(tree, path) is leaf


def test_virtual_path_resolves_to_structural_component_path(root):
    path = TreePath("main_wing.value", path_slice=slice(0, 1))
    actual = get_actual_path(root, path)

    assert actual.path == ("subcomponents", 0, "value")
    assert actual.path_slice == slice(0, 1)
    np.testing.assert_allclose(get_target(root, actual), [[0, 1, 2]])


def test_get_actual_path_preserves_real_dictionary_keys(root):
    actual = get_actual_path({"root": root}, "root.counter")

    assert actual.path == ("root", "counter")


def test_get_actual_path_rejects_unknown_attribute(root):
    with pytest.raises(AttributeError, match="no real or virtual attribute 'missing'"):
        get_actual_path(root, "missing.value")


def test_virtual_attribute_resolution_is_rejected_inside_jit():
    with pytest.raises(RuntimeError, match="Cannot resolve virtual attribute 'alias' on a JAX tracer"):
        jax.jit(lambda array: get_actual_path(array, "alias"))(jnp.ones(2))


@pytest.mark.parametrize("route", ["string", "pair", "path", "lambda", "list", "tuple"])
def test_update_supported_routes_are_immutable(root, route):
    if route == "string":
        changed = update(root, "counter", 5)
    elif route == "pair":
        changed = update(root, ("counter", 5))
    elif route == "path":
        changed = update(root, TreePath("counter", value=5))
    elif route == "lambda":
        changed = update(root, lambda tree: tree.counter, 5)
    elif route == "list":
        changed = update(root, [("counter", 5), ("main_wing.value", jnp.ones((2, 3)))])
    else:
        changed = update(root, (TreePath("counter", 5), TreePath("main_wing.value", jnp.ones((2, 3)))))

    assert changed.counter == 5
    assert root.counter == 0
    np.testing.assert_allclose(root.subcomponents[0].value, [[0, 1, 2], [3, 4, 5]])
    if route in ("list", "tuple"):
        np.testing.assert_allclose(changed.subcomponents[0].value, np.ones((2, 3)))
    else:
        assert changed.subcomponents[0].value is root.subcomponents[0].value


def test_update_slice_changes_only_selected_elements(root):
    changed = update(root, TreePath("main_wing.value", value=-1, path_slice=(slice(None), 1)))

    np.testing.assert_allclose(changed.main_wing.value, [[0, -1, 2], [3, -1, 5]])
    np.testing.assert_allclose(root.main_wing.value, [[0, 1, 2], [3, 4, 5]])


@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="A tuple of updates beginning with (path, integer) is misread as one path/value pair",
)
def test_update_tuple_of_scalar_pairs_is_not_a_single_indexed_path(root):
    changed = update(root, (("counter", 5), ("main_wing.value", jnp.ones((2, 3)))))

    assert changed.counter == 5
    np.testing.assert_allclose(changed.main_wing.value, np.ones((2, 3)))
    assert root.counter == 0


def test_update_can_replace_leaf_with_none(root):
    changed = update(root, ("counter", None))

    assert changed.counter is None
    assert root.counter == 0


def test_update_rejects_invalid_spec(root):
    with pytest.raises(TypeError, match=r"update\(\) requires"):
        update(root, object())


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ({}, {}, True),
        ({"x": 1}, {"x": 1.0}, True),
        ({"x": 1}, {"x": 2}, False),
        ({"x": 1}, {"y": 1}, False),
        ([1, 2], (1, 2), False),
        ({"x": [1]}, {"x": (1,)}, False),
        ({"x": "a"}, {"x": "b"}, False),
        ({"x": None}, {"x": None}, True),
        ({"x": jnp.array([[1.0]])}, {"x": jnp.array([1.0])}, True),
        ({"x": jnp.array([np.nan, 2.0])}, {"x": np.array([np.nan, 2.0])}, True),
        ({"x": jnp.array([np.nan, 2.0])}, {"x": jnp.array([1.0, 2.0])}, False),
        ({"x": jnp.ones((2, 3))}, {"x": jnp.ones((3, 2))}, False),
        ({"x": jnp.array([1.0])}, {"x": jnp.array([1.00001])}, False),
        ({"x": jnp.empty((0,))}, {"x": jnp.empty((0,))}, True),
    ],
)
def test_is_equivalent_checks_structure_and_exact_numeric_values(left, right, expected):
    assert is_equivalent(left, right) is expected


def test_is_equivalent_does_not_accept_string_arrays_as_numeric_leaves():
    with pytest.raises(TypeError):
        is_equivalent({"x": np.array(["label"])}, {"x": np.array(["label"])})


def test_is_equivalent_cannot_convert_traced_equality_to_python_bool():
    with pytest.raises(jax.errors.TracerBoolConversionError):
        jax.jit(lambda array: is_equivalent({"x": array}, {"x": jnp.ones(2)}))(jnp.ones(2))


def test_delta_round_trip_preserves_base_and_unchanged_leaves():
    shared = object()
    old = {"a": jnp.array([1, 2]), "b": (shared, jnp.zeros(2))}
    new = {"a": jnp.array([1, 2]), "b": (shared, jnp.ones(2))}

    indices, values = compute_tree_delta(old, new)
    restored = apply_tree_delta(old, indices, values)

    assert indices == [2]
    assert values[0] is new["b"][1]
    assert restored["a"] is old["a"]
    assert restored["b"][0] is shared
    np.testing.assert_allclose(restored["b"][1], [1, 1])
    np.testing.assert_allclose(old["b"][1], [0, 0])


@pytest.mark.parametrize("value", [np.array([1.0]), jnp.array([[1.0]])])
def test_delta_detects_numpy_identity_and_shape_changes(value):
    old = [jnp.array([1.0]) if isinstance(value, jax.Array) else np.array([1.0])]

    indices, leaves = compute_tree_delta(old, [value])

    assert indices == [0]
    assert leaves[0] is value


def test_delta_treats_same_nan_array_as_unchanged():
    value = jnp.array([np.nan])

    assert compute_tree_delta([value], [value]) == ([], [])


def test_distinct_nan_arrays_are_changed():
    replacement = jnp.array([np.nan])
    indices, values = compute_tree_delta([jnp.array([np.nan])], [replacement])

    assert indices == [0]
    assert values[0] is replacement


def test_apply_delta_supports_duplicate_and_negative_indices():
    base = [1, 2, 3]

    assert apply_tree_delta(base, [0, 0, -1], [4, 5, 6]) == [5, 2, 6]
    assert base == [1, 2, 3]


@pytest.mark.parametrize("indices,values", [([0, 1], [9]), ([0], [9, 8])])
def test_apply_delta_pairs_only_available_replacements(indices, values):
    assert apply_tree_delta([1, 2], indices, values) == [9, 2]


@pytest.mark.parametrize("index", [2, -3])
def test_apply_delta_rejects_out_of_range_indices(index):
    with pytest.raises(IndexError):
        apply_tree_delta([1, 2], [index], [9])


def test_partition_uses_identity_and_requires_array_like_leaves():
    active, equal_but_inactive = jnp.ones(2), jnp.ones(2)
    label = "metadata"
    tree = {"active": active, "inactive": equal_but_inactive, "label": label}

    dynamic, static, mask = id_partition(tree, {id(active), id(label)})

    assert mask == {"active": True, "inactive": False, "label": False}
    assert dynamic["active"] is active
    assert dynamic["inactive"] is dynamic["label"] is None
    assert static["active"] is None
    assert static["inactive"] is equal_but_inactive
    assert static["label"] is label
    assert is_equivalent(combine(dynamic, static), tree)


def test_inspect_leaves_groups_counts_and_types_at_root(capsys):
    settings = SimpleNamespace(logging=SimpleNamespace(stream_ouput=True, log_dir=None))

    assert inspect_leaves({"a": [1, 2], "b": "label"}, {"a": [True, False], "b": False}, settings, depth=0) is None

    lines = capsys.readouterr().out.strip().splitlines()
    assert "PyTree Path (Depth 0)" in lines[0]
    assert [part.strip() for part in lines[-1].split("|")] == ["Tree", "1", "2", "int"]


def test_inspect_leaves_creates_and_overwrites_log_without_stream_output(tmp_path, capsys):
    log_dir = tmp_path / "nested" / "logs"
    settings = SimpleNamespace(logging=SimpleNamespace(stream_ouput=False, log_dir=log_dir), verbose=True)

    inspect_leaves({"x": 1}, {"x": True}, settings, tree_name="sample")
    log_path = log_dir / "sample_structure.log"
    first_output = capsys.readouterr().out
    assert first_output == f"\nSample leaf structure log saved to {log_path}\n"
    assert "sample['x']" in log_path.read_text()

    settings.verbose = False
    inspect_leaves({"y": 2}, {"y": False}, settings, tree_name="sample")

    assert capsys.readouterr().out == ""
    assert "sample['y']" in log_path.read_text()
    assert "sample['x']" not in log_path.read_text()


def test_scan_reports_nested_invalid_leaves_without_mutation(capsys):
    tree = {"valid": jnp.ones(2), "invalid": ["label", object()]}
    original = tree["invalid"][1]

    assert scan_for_invalid_JAX_types(tree) is None

    output = capsys.readouterr().out
    assert output.count("Invalid JAX Type Found") == 2
    assert "PyTree['invalid'][0]" in output
    assert "Type:  <class 'str'>" in output
    assert "Value: label" in output
    assert "is a valid PyTree" not in output
    assert tree["invalid"][1] is original


def test_scan_accepts_numeric_leaves_and_ignores_static_metadata(root, capsys):
    scan_for_invalid_JAX_types(root)
    output = capsys.readouterr().out

    assert "is a valid PyTree" in output
    assert "Invalid JAX Type" not in output
