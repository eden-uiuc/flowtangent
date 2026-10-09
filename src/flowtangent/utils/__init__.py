# src/flowtangent/utils/__init__.py

# 1. Base / Syntax
from .base import (
    null_step,
    field,
    static_field,
    method_field,
    empty_array,
    Module,
)

# 2. PyTree Facade
from .tree import (
    TreePath,
    Partial,
    update,
    get_actual_path,
    get_target,
    get_parent_target,
    get_all_targets,
    get_all_parents,
    is_equivalent,
    compute_tree_delta,
    apply_tree_delta,
    id_partition,
    inspect_leaves,
    scan_for_invalid_JAX_types,
    ArraySlice,
    # Upstream JAX/Equinox
    tree_map,
    tree_flatten,
    tree_unflatten,
    tree_leaves,
    tree_map_with_path,
    tree_flatten_with_path,
    partition,
    combine,
    is_array,
    is_array_like,
)

from .backend import configure_environment

# 3. I/O and Serialization
from .io import (
    inputs,
    outputs,
    parse_io,
    jax_path_string,
    save_data,
    load_data,
    LoggingSettings,
)

# 4. Math and Display
from .display import format_array, MERMAID_STYLES
from .math import cubic_spline_blender

# 5. JAX Typing
from .typing import (
    ScalarFloat,
    ScalarInt,
    ScalarBool,
    TimeScalar,
    TimeVector3,
    NameType,
    TreePathLike,
)

from .data import Dataset, DataLoader

__all__ = [
    "null_step",
    "field",
    "static_field",
    "method_field",
    "empty_array",
    "Module",
    "TreePath",
    "ArraySlice",
    "Partial",
    "update",
    "get_actual_path",
    "get_target",
    "get_parent_target",
    "get_all_targets",
    "get_all_parents",
    "is_equivalent",
    "compute_tree_delta",
    "apply_tree_delta",
    "id_partition",
    "inspect_leaves",
    "scan_for_invalid_JAX_types",
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
    "configure_environment",
    "inputs",
    "outputs",
    "parse_io",
    "jax_path_string",
    "save_data",
    "load_data",
    "LoggingSettings",
    "format_array",
    "cubic_spline_blender",
    "MERMAID_STYLES",
    "ScalarFloat",
    "ScalarInt",
    "ScalarBool",
    "TimeScalar",
    "TimeVector3",
    "NameType",
    "TreePathLike",
    "Dataset",
    "DataLoader"
]
