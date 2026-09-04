from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, overload

if TYPE_CHECKING:
    from ..core._settings import Settings

import os
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from equinox import combine, is_array, is_array_like, partition

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
def update(obj: Any, where_or_updates: Callable, val: Any) -> Any:
    ...

@overload
def update(obj: Any, where_or_updates: TreePath | tuple | Sequence[TreePath | tuple]) -> Any:
    ...

def update(obj, where_or_updates, val=None):
    """
    FlowTangent wrapper for eqx.tree_at.
    """
    # Route 1: The Canonical Equinox Lambda
    if callable(where_or_updates):
        return eqx.tree_at(where_or_updates, obj, val)

    # Route 2: Single Tuple or TreePath (e.g., ('aero.alpha', 3.0))
    # We check if it's a tuple where the first element is a string/tuple path
    if isinstance(where_or_updates, TreePath) or (
        isinstance(where_or_updates, tuple)
        and len(where_or_updates) in (2, 3)
        and isinstance(where_or_updates[0], (str, tuple))
    ):
        paths = [TreePath.cast(where_or_updates)]

    # Route 3: A List/Sequence of Updates
    elif isinstance(where_or_updates, (list, tuple, set)):
        paths = [TreePath.cast(u) for u in where_or_updates]

    else:
        raise TypeError(
            "update() requires a lambda function, a TreePath, a path tuple, "
            "or a sequence of updates."
        )

    where_fn = lambda s: get_all_targets(s, paths)
    vals = tuple(p.value for p in paths)
    return eqx.tree_at(where_fn, obj, vals)

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
    def cast(cls, item: Any) -> 'TreePath':
        """Convenience method to intelligently cast strings and tuples into TreePaths."""
        if isinstance(item, cls):
            return item
        if isinstance(item, str):
            return cls(path=item)
        if isinstance(item, tuple):
            if len(item) == 2:
                return cls(path=item[0], value=item[1])
            elif len(item) == 3:
                return cls(path=item[0], value=item[1], path_slice=item[2])
        raise TypeError(f"Cannot automatically cast {type(item)} into a TreePath.")

    def __init__(
            self,
            path: tuple | str | 'TreePath' = ('state',),
            value: Optional[Any] = None,
            path_slice: slice = slice(None),
            name: Optional[str] = None,
        ):

        if isinstance(path, TreePath):
            object.__setattr__(self, "path", path.path)
            object.__setattr__(self, "value", path.value)
            object.__setattr__(self, "path_slice", path.path_slice)
            object.__setattr__(self, "name", path.name)

        else:
            if isinstance(path, tuple):
                path_tuple = path
            elif isinstance(path, str):
                path_tuple = tuple(path.split('.'))
            else:
                raise ValueError("TreePath path must be a tuple or string.")

            object.__setattr__(self, "path", path_tuple)
            object.__setattr__(self, "value", value)
            object.__setattr__(self, "path_slice", path_slice)

            if name is None:
                path_name = '.'.join(self.path)
            else:
                path_name = name

            object.__setattr__(self, "name", path_name)

    def __len__(self):
        return len(self.path)

    def _snip_lead(self):
        return update(self, lambda p: p.path, self.path[1:])

def get_parent_target(obj: Any, path: str | tuple | TreePath) -> Any:
    """Gets the full PyTree leaf, ignoring the slice."""
    path_obj = TreePath.cast(path)
    for key in path_obj.path:
        if isinstance(obj, dict):
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
    if type(a) != type(b):
        return False

    try:
        a_leaves, a_treedef = tree_flatten(a)
        b_leaves, b_treedef = tree_flatten(b)
    except Exception:
        return False

    if a_treedef != b_treedef:
        return False

    for la, lb in zip(a_leaves, b_leaves):
        is_num_a = isinstance(la, (jnp.ndarray, np.ndarray, float, int, bool))
        is_num_b = isinstance(lb, (jnp.ndarray, np.ndarray, float, int, bool))

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
        if isinstance(old, jnp.ndarray) and isinstance(new, jnp.ndarray):
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

def io_partition(tree, active_ids: set[int]):
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
def inspect_leaves(tree, mask, settings: Settings, tree_name:str="Tree", depth:int=3):
    """Groups PyTree leaves by their hierarchical path and outputs the summary."""
    leaves_with_path, _ = tree_flatten_with_path(tree)
    mask_leaves, _ = tree_flatten(mask)

    summary = {}

    for (path, leaf), is_kept in zip(leaves_with_path, mask_leaves):
        path_strs = []
        for p in path:
            if hasattr(p, 'name'):
                path_strs.append(f".{p.name}")
            elif hasattr(p, 'key'):
                path_strs.append(f"['{p.key}']")
            elif hasattr(p, 'idx'):
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
    header = f"{'PyTree Path (Depth ' + str(depth) + ')':<{35 + 15 * depth}} | {'Kept':<6} | {'Pruned':<6} | {'Common Kept Types'}"
    lines.append(header)
    lines.append("-" * 100)

    for prefix, counts in sorted(summary.items()):
        if counts['kept'] > 0 or counts['pruned'] > 0:
            types_str = ", ".join(sorted(list(counts['types']))[:3])
            lines.append(f"{prefix:<{35 + 15 * depth}} | {counts['kept']:<6} | {counts['pruned']:<6} | {types_str}")

    output_text = "\n".join(lines)

    if settings.logging.stream_ouput:
        print("\n" + output_text)

    if settings.logging.log_dir is not None:
        output_file = Path(settings.logging.log_dir) / f"{tree_name}_structure.log"
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(output_text)
        if getattr(settings, 'verbose', False):
            print(f"\n - {tree_name.title()} leaf structure log saved to {output_file}")

def scan_for_invalid_JAX_types(pytree, name="PyTree") -> None:
    print(f"--- Scanning {name} for invalid dynamic leaves ---")
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

            print(f"Invalid JAX Type Found: {name}{path_str}\n   Type:  {type(leaf)}\n   Value: {leaf}\n")
        return leaf

    tree_map_with_path(check_leaf, pytree)

    if not found_invalid:
        print(f"{name} is a valid PyTree.\n")

# -----------------------------------------------------------------------------
# EXPLICIT FACADE EXPORTS
# -----------------------------------------------------------------------------
__all__ = [
    # JAX & Equinox Native
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
    "io_partition",
    "inspect_leaves",
    "scan_for_invalid_JAX_types",
]
