# src/eden_trace/utils/__init__.py

# 1. Base / Syntax
from .base import field, static_field, method_field, empty_array, Module, StateData

# 2. PyTree Facade
from .tree import (
    TreePath, update, get_target, get_parent_target, get_all_targets, get_all_parents,
    is_equivalent, compute_tree_delta, apply_tree_delta, io_partition, inspect_leaves,
    scan_for_invalid_JAX_types,
    
    # Upstream JAX/Equinox
    tree_map, tree_flatten, tree_unflatten, tree_leaves, tree_map_with_path, tree_flatten_with_path,
    partition, combine, is_array, is_array_like
)

# 3. I/O and Serialization
from .io import inputs, outputs, parse_io, jax_path_string, register, save_data, load_data

# 4. Math and Display
from .display import format_array, MERMAID_STYLES
from .math import cubic_spline_blender