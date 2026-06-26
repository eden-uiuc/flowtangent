# RCAIDE/utils.py
# (c) Copyright 2026 Aerospace Research Community LLC
#
# Created: Apr 2026, J. Smart
# Modified: Apr 2026, J. Smart

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Self

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

# --- Framework Imports (Strictly for Type Hinting to avoid Circular Imports) ---
if TYPE_CHECKING:
    from RCAIDE.Framework.Settings import Settings
    from RCAIDE.Framework.State import State
    from RCAIDE.Framework.Systems import System

# ----------------------------------------------------------------------------------------------------------------------
#  Utility Functions
# ----------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------
# Syntax Helpers
#----------------------------------------------------------

def init_field(initializer: Any, as_value: bool = False, **kwargs):
    """
    Smart wrapper for eqx.field that automatically routes the initializer
    to `default` (for immutables) or `default_factory` (for classes/callables).
    """
    # Handle factories and classes (e.g., list, dict, MediumRange)
    if as_value:
        return eqx.field(default=initializer, **kwargs)
    if callable(initializer):
        return eqx.field(default_factory=initializer, **kwargs)

    # Guardrail: Catch accidentally instantiated mutable defaults
    if isinstance(initializer, (list, dict, set)):
        raise ValueError(
            f"Mutable instance {initializer} passed to init_field. "
            "Pass the uninstantiated class (e.g., list) or a lambda instead."
        )

    # Handle static/immutable defaults (e.g., 'Aircraft', 0.0, (1, 2))
    return eqx.field(default=initializer, **kwargs)

def empty_array(shape: tuple | int = 0, dtype: Any = float, **kwargs):
    """Syntactic sugar for an empty JAX array in an Equinox module."""
    return init_field(lambda: jnp.empty(shape, dtype=dtype), **kwargs)

# ---------------------------------------------------------
# Programmatic Helpers
#----------------------------------------------------------

def get_RCAIDE_root():
    return Path(os.path.dirname(os.path.abspath(__file__))).parents[0].resolve()

# ---------------------------------------------------------
# Input/Output Function Decorators
# ---------------------------------------------------------

def inputs(*dependencies: str):
    def decorator(func: Callable):
        func._inputs = set(dependencies)
        return func
    return decorator


def outputs(*outputs: str):
    def decorator(func: Callable):
        func._outputs = set(outputs)
        return func
    return decorator

MERMAID_STYLES = {
    "default": "",

    "formal": """%%{init: {'theme': 'base', 'themeVariables': {
        'primaryColor': '#ffffff',
        'primaryBorderColor': '#000000',
        'primaryTextColor': '#000000',
        'lineColor': '#000000',
        'fontFamily': 'Times New Roman, serif'
    }}}%%""",

    "modern": """%%{init: {'theme': 'base', 'themeVariables': {
        'primaryColor': '#f8fafc',
        'primaryBorderColor': '#3b82f6',
        'primaryTextColor': '#0f172a',
        'lineColor': '#94a3b8',
        'fontFamily': 'Inter, system-ui, sans-serif'
    }}}%%""",

    "dark": """%%{init: {'theme': 'dark', 'themeVariables': {
        'primaryColor': '#1e1e1e',
        'primaryBorderColor': '#10b981',
        'primaryTextColor': '#e5e7eb',
        'lineColor': '#10b981',
        'fontFamily': 'Fira Code, monospace'
    }}}%%"""
}

# ---------------------------------------------------------
# Find Targets from Path in PyTrees
# ---------------------------------------------------------

class Token(eqx.Module):

    state: eqx.Module
    system: eqx.Module
    settings: eqx.Module

@dataclass(frozen=True)
class DataPath:

    path: tuple
    slice_obj: slice
    tag: str = "Variable Path"

    def __init__(self, path: tuple | Self = (slice(None),), tag="Variable Path"):

        if isinstance(path, DataPath):
            object.__setattr__(self, 'path', path.path)
            object.__setattr__(self, 'slice_obj', path.slice_obj)
            object.__setattr__(self, 'tag', path.tag)

        else:
            if isinstance(path[-1], slice):
                object.__setattr__(self, 'path', path[:-1])
                object.__setattr__(self, 'slice_obj', path[-1])
            else:
                object.__setattr__(self, 'path', path)
                object.__setattr__(self, 'slice_obj', slice(None))

            object.__setattr__(self, 'tag', tag)

    def __len__(self):
        return len(self.path)

def get_parent_target(obj, path_tuple: DataPath):
    """Gets the full PyTree leaf, ignoring the slice."""
    for key in path_tuple.path:
        if isinstance(obj, dict):
            obj = obj[key]
        else:
            obj = getattr(obj, key)
    return obj

def get_target(obj, path_tuple: DataPath):
    """Gets the target and applies the slice if one exists."""
    parent = get_parent_target(obj, path_tuple)
    if hasattr(parent, "__getitem__") and path_tuple.slice_obj != slice(None):
        return parent[path_tuple.slice_obj]
    return parent

def get_all_parents(s, input_map):
    return tuple(get_parent_target(s, path) for path in input_map)

def get_all_targets(s, input_map):
    return tuple(get_target(s, path) for path in input_map)

# ---------------------------------------------------------
# PyTree Deltas
# ---------------------------------------------------------

def compute_tree_delta(old_tree, new_tree):
    """Find changes between two identically structured PyTrees."""
    old_leaves, _ = jax.tree_util.tree_flatten(old_tree)
    new_leaves, _ = jax.tree_util.tree_flatten(new_tree)

    changed_indices = []
    changed_leaves = []

    for i, (old, new) in enumerate(zip(old_leaves, new_leaves)):
        # Handle unchanged leaves
        if old is new: continue
        if isinstance(old, jnp.ndarray) and isinstance(new, jnp.ndarray):
            if old.shape == new.shape and jnp.all(old == new): continue

        changed_indices.append(i)
        changed_leaves.append(new)

    return changed_indices, changed_leaves

def apply_tree_delta(base_tree, delta_indices, delta_leaves):

    """Reconstructs new tree from base tree and delta."""
    old_leaves, treedef = jax.tree_util.tree_flatten(base_tree)
    new_leaves = list(old_leaves)
    for idx, leaf in zip(delta_indices, delta_leaves):
        new_leaves[idx] = leaf

    return jax.tree_util.tree_unflatten(treedef, new_leaves)

# ---------------------------------------------------------
# Debugging Tools
# ---------------------------------------------------------

def scan_for_invalid_JAX_types(pytree, name="PyTree"):
    print(f"--- Scanning {name} for invalid dynamic leaves ---")
    found_invalid = False

    def check_leaf(path, leaf):
        nonlocal found_invalid

        # These are the only types JAX should ever see in the dynamic leaves
        valid_jax_types = (jax.Array, np.ndarray, float, int, complex, bool)

        if not isinstance(leaf, valid_jax_types):
            found_invalid = True

            # Format the exact path (handles Equinox attributes, dict keys, and tuple indices)
            path_str = ""
            for p in path:
                if hasattr(p, 'name'):
                    path_str += f".{p.name}"
                elif hasattr(p, 'key'):
                    path_str += f"[{repr(p.key)}]"
                elif hasattr(p, 'idx'):
                    path_str += f"[{p.idx}]"
                else:
                    path_str += f"<{p}>"

            print(f"Invalid JAX Type Found: {name}{path_str}")
            print(f"   Type:  {type(leaf)}")
            print(f"   Value: {leaf}\n")

        return leaf

    # Walk the tree and check every single dynamic leaf
    jax.tree_util.tree_map_with_path(check_leaf, pytree)

    if not found_invalid:
        print(f"{name} is a valid PyTree.\n")

# ---------------------------------------------------------
# General Mathematical Utilities
# ---------------------------------------------------------

@jax.jit
def cubic_spline_blender(x, start, end):

    eta = (x - start) / (end - start)
    eta_clamped = jnp.clip(eta, 0.1, 1.0)
    y = -2.0 * eta_clamped ** 3 + 3.0 * eta_clamped ** 2
    return y


if __name__ == '__main__':
    print(get_RCAIDE_root())
