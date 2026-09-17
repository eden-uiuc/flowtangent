from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, overload

if TYPE_CHECKING:
    from ..core._settings import Settings

import os
from collections import deque
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from equinox import Partial, combine, is_array, is_array_like, partition

# -----------------------------------------------------------------------------
# UPSTREAM FACADE IMPORTS
# -----------------------------------------------------------------------------
from jax.tree_util import (
    tree_flatten,
    tree_flatten_with_path,
    tree_leaves,
    tree_map,
    tree_map_with_path,
    tree_unflatten,
)

# -----------------------------------------------------------------------------
# FLOWTANGENT WRAPPERS
# -----------------------------------------------------------------------------


@overload
def update(obj: Any, where_or_updates: Callable, val: Any, **kwargs) -> Any: ...


@overload
def update(obj: Any, where_or_updates: TreePath | tuple | Sequence[TreePath | tuple], **kwargs) -> Any: ...


@overload
def update(obj: Any, where_or_updates: str, val: Any, **kwargs) -> Any: ...


def update(obj, where_or_updates, val=None, **kwargs):
    """FlowTangent wrapper for eqx.tree_at."""
    if callable(where_or_updates):
        return eqx.tree_at(where_or_updates, obj, val, **kwargs)

    paths = []

    # Route 1: Explicit Single Update
    if val is not None:
        # If val is provided, where_or_updates is strictly the path.
        paths = [TreePath.cast(where_or_updates, default_val=val)]

    # Route 2: Implicit Updates (where_or_updates contains both paths and values)
    else:
        if isinstance(where_or_updates, TreePath):
            paths = [where_or_updates]

        elif isinstance(where_or_updates, str):
            paths = [TreePath.cast(where_or_updates)]

        elif isinstance(where_or_updates, (list, set)):
            paths = [TreePath.cast(u) for u in where_or_updates]

        elif isinstance(where_or_updates, tuple):
            # The Ultimate Ambiguity: Is this ONE update spec `("path", val)`,
            # or a tuple of multiple update specs `(("path1", val1), ("path2", val2))`?
            try:
                # Try treating it as a single update spec first
                paths = [TreePath.cast(where_or_updates)]
            except TypeError:
                # If that fails (e.g. the first element isn't a valid path),
                # it MUST be a tuple containing multiple update specs.
                paths = [TreePath.cast(u) for u in where_or_updates]

        else:
            raise TypeError("update() requires a lambda, a string path, a TreePath, a tuple, or a sequence.")

    # Canonicalize and apply
    actual_paths = [get_actual_path(obj, p) for p in paths]
    where_fn = partial(get_all_targets, input_map=actual_paths)
    vals = tuple(p.value for p in paths)

    return eqx.tree_at(where_fn, obj, vals, **kwargs)


# -----------------------------------------------------------------------------
# CORE PYTREE UTILITIES
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class TreePath:
    path: tuple
    value: Any
    path_slice: slice
    name: str

    @classmethod
    def cast(cls, item: Any, default_val: Any = None) -> "TreePath":
        """Strictly casts strings, path tuples, or update tuples into a TreePath."""
        if isinstance(item, cls):
            return item

        if isinstance(item, str):
            return cls(path=item, value=default_val)

        if isinstance(item, tuple):
            # Helper to check if something is strictly a path (str, or tuple of str/int)
            def is_path(p):
                return isinstance(p, str) or (isinstance(p, tuple) and all(isinstance(x, (str, int)) for x in p))

            # Case A: Two-element update tuple -> (path, value)
            if len(item) == 2 and is_path(item[0]):
                return cls(path=item[0], value=item[1])

            # Case B: Three-element update tuple -> (path, value, slice)
            if len(item) == 3 and is_path(item[0]) and isinstance(item[2], slice):
                return cls(path=item[0], value=item[1], path_slice=item[2])

            # Case C: The tuple IS the path (e.g. ("subcomponents", 0))
            if is_path(item):
                return cls(path=item, value=default_val)

        raise TypeError(f"Cannot automatically cast {item} into a TreePath.")

    def __init__(
        self,
        path: tuple | str | "TreePath" = ("state",),
        value: Optional[Any] = None,
        path_slice: Optional[slice] = None,
        name: Optional[str] = None,
    ):
        if isinstance(path, TreePath):
            object.__setattr__(self, "path", path.path)
            object.__setattr__(self, "value", path.value)
            object.__setattr__(self, "path_slice", path.path_slice)
            object.__setattr__(self, "name", path.name)
            return

        # 1. Flatten the path (handles mixed formats like ("system.energy", 0))
        parsed_path = []
        if isinstance(path, str):
            parsed_path = path.split(".")
        elif isinstance(path, tuple):
            for p in path:
                if isinstance(p, str | int):
                    parsed_path.append(p)
                else:
                    raise ValueError(f"Path elements must be strings or ints, got {type(p)}")
        else:
            raise ValueError("TreePath path must be a tuple or string.")

        object.__setattr__(self, "path", tuple(parsed_path))
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "path_slice", path_slice if path_slice is not None else slice(None))

        # 2. Build a readable name for aliases and debugging (e.g., "subcomponents[0].energy")
        if name is None:
            name_parts = []
            for p in parsed_path:
                if isinstance(p, int):
                    name_parts[-1] = f"{name_parts[-1]}[{p}]" if name_parts else f"[{p}]"
                else:
                    name_parts.append(p)
            object.__setattr__(self, "name", ".".join(name_parts))
        else:
            object.__setattr__(self, "name", name)

    def __len__(self):
        return len(self.path)

    def _snip_lead(self):
        return update(self, lambda p: p.path, self.path[1:])


def get_actual_path(obj: Any, path: str | tuple | TreePath) -> TreePath:
    "Walks object to convert virtual aliases to canonical paths."
    path_obj = TreePath.cast(path)
    actual_keys = []
    current = obj

    for i, key in enumerate(path_obj.path):
        if isinstance(current, dict):
            actual_keys.append(key)
            current = current[key]
            continue

        is_real = hasattr(current.__class__, key) or (
            hasattr(current, "__dataclass_fields__") and key in current.__dataclass_fields__
        )

        if is_real:
            actual_keys.append(key)
            current = getattr(current, key)
        else:
            if isinstance(current, jax.core.Tracer):
                raise RuntimeError(
                    f"Cannot resolve virtual attribute '{key}' on a JAX tracer. "
                    "Virtual paths (via bookkeeping or subcomponent aliases) are "
                    "illegal inside jit/vmap. Use canonical PyTree structure instead."
                )

            is_virtual = (hasattr(current, "_bookkeeping") and key in current._bookkeeping) or any(
                getattr(sc, "field_name", None) == key for sc in getattr(current, "subcomponents", [])
            )

            if not is_virtual:
                raise AttributeError(f"'{current.__class__.__name__}' has no real or virtual attribute '{key}'")

            target = current
            anchor_obj = current
            suffix = []

            for remaining_key in path_obj.path[i:]:
                if isinstance(remaining_key, int) or isinstance(target, dict):
                    target = target[remaining_key]  # type: ignore
                else:
                    target = getattr(target, remaining_key)
                if isinstance(target, eqx.Module):
                    anchor_obj = target
                    suffix = []
                else:
                    suffix.append(remaining_key)

            target_id = id(anchor_obj)

            # BFS to find where this target physically lives in the canonical tree
            def bfs(start_node):
                # Queue stores tuples of (current_object, path_to_object)
                queue = deque([(start_node, [])])

                while queue:
                    curr, current_path = queue.popleft()

                    if id(curr) == target_id:
                        return current_path

                    if isinstance(curr, type) or callable(curr):
                        continue

                    if hasattr(curr, "__dataclass_fields__"):
                        for f in curr.__dataclass_fields__:
                            try:
                                queue.append((getattr(curr, f), current_path + [f]))
                            except AttributeError:
                                pass
                    elif isinstance(curr, (tuple, list)):
                        for idx, val in enumerate(curr):
                            queue.append((val, current_path + [idx]))
                    elif isinstance(curr, dict):
                        for k, val in curr.items():
                            queue.append((val, current_path + [k]))
                return None

            found_path = bfs(current)
            if found_path is None:
                raise ValueError(f"Target object from virtual path '{key}' not found in canonical PyTree.")

            actual_keys.extend(found_path)
            actual_keys.extend(suffix)

            break

    return TreePath(path=tuple(actual_keys), path_slice=path_obj.path_slice)


def get_parent_target(obj: Any, path: str | tuple | TreePath) -> Any:
    """Gets the full PyTree leaf, ignoring the slice."""
    path_obj = TreePath.cast(path)
    for key in path_obj.path:
        if isinstance(obj, dict) or isinstance(key, int):
            obj = obj[key]
        else:
            obj = getattr(obj, key)
    return obj


def get_target(obj: Any, path: str | tuple | TreePath) -> Any:
    """Gets the target and applies the slice if one exists."""
    path_obj = TreePath.cast(path)
    parent = get_parent_target(obj, path_obj)
    if hasattr(parent, "__getitem__") and path_obj.path_slice != slice(None):
        return parent[path_obj.path_slice]
    return parent


def get_all_parents(s: Any, input_map: Sequence[str | tuple | TreePath]) -> tuple:
    return tuple(get_parent_target(s, path) for path in input_map)


def get_all_targets(s: Any, input_map: Sequence[str | tuple | TreePath]) -> tuple:
    return tuple(get_target(s, path) for path in input_map)


def is_equivalent(a, b):
    """Safely checks deep equality between any two PyTrees, arrays, or scalars."""
    if type(a) is not type(b):
        return False

    try:
        a_leaves, a_treedef = tree_flatten(a)
        b_leaves, b_treedef = tree_flatten(b)
    except Exception:
        return False

    if a_treedef != b_treedef:
        return False

    for la, lb in zip(a_leaves, b_leaves):
        is_num_a = isinstance(la, (jax.Array, np.ndarray, float, int, bool))
        is_num_b = isinstance(lb, (jax.Array, np.ndarray, float, int, bool))

        if is_num_a and is_num_b:
            arr_a = jnp.squeeze(jnp.asarray(la))
            arr_b = jnp.squeeze(jnp.asarray(lb))

            if arr_a.shape != arr_b.shape:
                return False

            if not jnp.array_equal(arr_a, arr_b, equal_nan=True):
                return False
        else:
            if la != lb:
                return False

    return True


def compute_tree_delta(old_tree, new_tree):
    """Find changes between two identically structured PyTrees."""
    old_leaves, _ = tree_flatten(old_tree)
    new_leaves, _ = tree_flatten(new_tree)

    changed_indices = []
    changed_leaves = []

    for i, (old, new) in enumerate(zip(old_leaves, new_leaves)):
        if old is new:
            continue
        if isinstance(old, jax.Array) and isinstance(new, jax.Array):
            if old.shape == new.shape and jnp.all(old == new):
                continue

        changed_indices.append(i)
        changed_leaves.append(new)

    return changed_indices, changed_leaves


def apply_tree_delta(base_tree, delta_indices, delta_leaves):
    """Reconstructs new tree from base tree and delta."""
    old_leaves, treedef = tree_flatten(base_tree)
    new_leaves = list(old_leaves)
    for idx, leaf in zip(delta_indices, delta_leaves):
        new_leaves[idx] = leaf

    return tree_unflatten(treedef, new_leaves)


def id_partition(tree, active_ids: set[int]):
    """
    Partitions a PyTree in dynamic and static halves based on an IO whitelist.
    Only JAX arrays whose paths are in the whitelist are kept dynamic.
    """

    def is_active(leaf):
        return is_array_like(leaf) and id(leaf) in active_ids

    mask = tree_map(is_active, tree)
    dyn, stat = partition(tree, mask)
    return dyn, stat, mask


# -----------------------------------------------------------------------------
# DEBUGGING UTILITIES
# -----------------------------------------------------------------------------
def inspect_leaves(tree, mask, settings: Settings, tree_name: str = "Tree", depth: int = 3):
    """Groups PyTree leaves by their hierarchical path and outputs the summary."""
    leaves_with_path, _ = tree_flatten_with_path(tree)
    mask_leaves, _ = tree_flatten(mask)

    summary = {}

    for (path, leaf), is_kept in zip(leaves_with_path, mask_leaves):
        path_strs = []
        for p in path:
            if hasattr(p, "name"):
                path_strs.append(f".{p.name}")
            elif hasattr(p, "key"):
                path_strs.append(f"['{p.key}']")
            elif hasattr(p, "idx"):
                path_strs.append(f"[{p.idx}]")
            else:
                path_strs.append(str(p))

        prefix = tree_name + "".join(path_strs[:depth])
        if not prefix:
            prefix = tree_name

        if prefix not in summary:
            summary[prefix] = {"kept": 0, "pruned": 0, "types": set()}

        if not is_kept:
            summary[prefix]["pruned"] += 1
        else:
            summary[prefix]["kept"] += 1
            summary[prefix]["types"].add(type(leaf).__name__)

    lines = []
    header = f"{'PyTree Path (Depth ' + str(depth) + ')':<{35 + 15 * depth}} | {'Kept':<6} | {'Pruned':<6} | {
        'Common Kept Types'
    }"
    lines.append(header)
    lines.append("-" * 100)

    for prefix, counts in sorted(summary.items()):
        if counts["kept"] > 0 or counts["pruned"] > 0:
            types_str = ", ".join(sorted(list(counts["types"]))[:3])
            lines.append(f"{prefix:<{35 + 15 * depth}} | {counts['kept']:<6} | {counts['pruned']:<6} | {types_str}")

    output_text = "\n".join(lines)

    if settings.logging.stream_ouput:
        print("\n" + output_text)

    if settings.logging.log_dir is not None:
        output_file = Path(settings.logging.log_dir) / f"{tree_name}_structure.log"
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, "w") as f:
            f.write(output_text)
        if getattr(settings, "verbose", False):
            print(f"\n - {tree_name.title()} leaf structure log saved to {output_file}")


def scan_for_invalid_JAX_types(pytree, name: Optional[str] = None) -> None:
    tree_name = getattr(pytree, "name", "PyTree")
    scan_name = tree_name if name is None else tree_name
    print(f"--- Scanning {scan_name} for invalid dynamic leaves ---")
    found_invalid = False

    def check_leaf(path, leaf):
        nonlocal found_invalid
        valid_jax_types = (jax.Array, np.ndarray, float, int, complex, bool)

        if not isinstance(leaf, valid_jax_types):
            found_invalid = True
            path_str = ""
            for p in path:
                if hasattr(p, "name"):
                    path_str += f".{p.name}"
                elif hasattr(p, "key"):
                    path_str += f"[{repr(p.key)}]"
                elif hasattr(p, "idx"):
                    path_str += f"[{p.idx}]"
                else:
                    path_str += f"<{p}>"

            print(f"Invalid JAX Type Found: {scan_name}{path_str}\n   Type:  {type(leaf)}\n   Value: {leaf}\n")
        return leaf

    tree_map_with_path(check_leaf, pytree)

    if not found_invalid:
        print(f"{scan_name} is a valid PyTree.\n")


# -----------------------------------------------------------------------------
# EXPLICIT FACADE EXPORTS
# -----------------------------------------------------------------------------
__all__ = [
    # JAX & Equinox Native
    "Partial",
    "tree_map",
    "tree_flatten",
    "tree_unflatten",
    "tree_leaves",
    "tree_map_with_path",
    "tree_flatten_with_path",
    "partition",
    "combine",
    "is_array",
    "is_array_like",
    # FlowTangent API Wrappers
    "update",
    # FlowTangent Custom Functions
    "TreePath",
    "get_parent_target",
    "get_target",
    "get_all_parents",
    "get_all_targets",
    "is_equivalent",
    "compute_tree_delta",
    "apply_tree_delta",
    "id_partition",
    "inspect_leaves",
    "scan_for_invalid_JAX_types",
]
