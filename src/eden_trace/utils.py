# Trace/utils.py
# (c) Copyright 2026 Aerospace Research Community LLC
#
# Created: Apr 2026, J. Smart
# Modified: Apr 2026, J. Smart

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, Self, Sequence, Optional
if TYPE_CHECKING:
    from .framework.settings import Settings

import os
import gzip
import itertools
import json
import string
import time
import warnings

from dataclasses import dataclass
from functools import reduce
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
#  Utility Functions
# ----------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------
# Syntax Helpers
# ----------------------------------------------------------


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
# ----------------------------------------------------------


def get_trace_root():
    return Path(os.path.dirname(os.path.abspath(__file__))).resolve()

def null_step(*args):
    return args

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

def parse_io(io_string: str, var_map: dict | eqx.Module) -> set:
    # 1. Strip type hints
    io_parts = io_string.split(":")
    io_string = io_parts[0].strip()
    if len(io_parts) > 1:
        meta_string = ": " + io_parts[1].strip()
    else:
        meta_string = ""

    # 2. Extract raw keys (e.g., 'flow_inputs.network_ID')
    required_keys = [
        tup[1] for tup in string.Formatter().parse(io_string) 
        if tup[1] is not None
    ]

    if not required_keys:
        return {io_string}

    normalized_map = {}
    for full_key in required_keys:
        # Prevent .format() from natively trying to evaluate the dot on our unpacked strings
        safe_key = full_key.replace('.', '___')
        io_string = io_string.replace(f"{{{full_key}}}", f"{{{safe_key}}}")

        # Split into base key and attribute chain (e.g., ['flow_inputs', 'network_ID'])
        parts = full_key.split('.')
        base_key = parts[0]
        attrs = parts[1:]

        # Fetch the base value and wrap it in a list if necessary
        if isinstance(var_map, dict):
            base_val = var_map.get(base_key, [])
        else:
            base_val = getattr(var_map, base_key, [])
        if isinstance(base_val, (str, int, float, bool)):
            base_list = [base_val]
        elif isinstance(base_val, (list, tuple, set)):
            base_list = list(base_val)
        else:
            base_list = [base_val]

        # 3. Iterate through the list and apply the attribute chain using your reduce logic!
        resolved_list = []
        for item in base_list:
            try:
                resolved_item = item
                for attr in attrs:
                    if attr.endswith('()'):
                        # It is a method call (e.g., "lower()")
                        method_name = attr[:-2]
                        method = getattr(resolved_item, method_name)
                        resolved_item = method()
                    else:
                        # It is a standard attribute
                        resolved_item = getattr(resolved_item, attr)
                        
                resolved_list.append(resolved_item)
            except AttributeError:
                raise AttributeError(f"Could not resolve '{'.'.join(attrs)}' on {item}")

        normalized_map[safe_key] = resolved_list

    # 4. Generate all permutations (Cartesian product)
    keys = list(normalized_map.keys())
    value_lists = list(normalized_map.values())
    
    resolved_paths = set()
    for combination in itertools.product(*value_lists):
        combo_dict = dict(zip(keys, combination))
        resolved_string = io_string.format(**combo_dict)
        resolved_paths.add(resolved_string)
            
    return resolved_paths

def jax_path_string(jax_path: tuple) -> str:
    """Converts internal JAX path tuple into standard Python syntax."""
    path_str = ""
    for p in jax_path:
        if hasattr(p, 'name'):
            path_str += f".{p.name}"
        elif hasattr(p, 'key'):
            path_str += f"['{p.key}']"
        elif hasattr(p, 'idx'):
            path_str += f"[{p.idx}]"
    return path_str.lstrip(".")

# ---------------------------------------------------------
# Formatting
# ---------------------------------------------------------

def format_array(v, precision=3, width=10):
    v_np = np.asarray(v)
    if v_np.size == 1:
        return f"{v_np.item():>{width}.{precision}e}"
    # For 1D/2D arrays, use numpy's built-in pretty printer
    return np.array2string(v_np, precision=precision, separator=', ')


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
    }}}%%""",
}

# ---------------------------------------------------------
# Find Targets from Path in PyTrees
# ---------------------------------------------------------

@dataclass(frozen=True)
class DataPath:
    path: tuple
    path_slice: slice
    value: Any
    tag: str

    def __init__(
            self,
            path: tuple | str | Self = ('state',),
            path_slice: slice = slice(None),
            value: Optional[Any] = None,
            tag: Optional[str] = None,
        ):

        if isinstance(path, DataPath):
            object.__setattr__(self, "path", path.path)
            object.__setattr__(self, "path_slice", path.path_slice)
            object.__setattr__(self, "value", path.value)
            object.__setattr__(self, "tag", path.tag)

        else:
            if isinstance(path, tuple):
                path_tuple = path
            elif isinstance(path, str):
                path_tuple = tuple(path.split('.'))
            else:
                raise ValueError(f"DataPath path must be a tuple or string.")

            object.__setattr__(self, "path", path_tuple)
            object.__setattr__(self, "path_slice", path_slice)
            object.__setattr__(self, "value", value)

            if tag is None:
                path_tag = '.'.join(self.path)
            else:
                path_tag = tag

            object.__setattr__(self, "tag", path_tag)

    def __len__(self):
        return len(self.path)

    def _snip_lead(self):
        return eqx.tree_at(lambda p: p.path, self, self.path[1:])


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
    if hasattr(parent, "__getitem__") and path_tuple.path_slice != slice(None):
        return parent[path_tuple.path_slice]
    return parent

def get_all_parents(s, input_map: Sequence[DataPath]):
    return tuple(get_parent_target(s, path) for path in input_map)

def get_all_targets(s, input_map: Sequence[DataPath]):
    return tuple(get_target(s, path) for path in input_map)


# ---------------------------------------------------------
# PyTree Deltas
# ---------------------------------------------------------

def is_equivalent(a, b):
    """
    Safely checks deep equality between any two PyTrees, arrays, or scalars.
    Accounts for dimension broadcasting (e.g., float == array([[float]])).
    """
    # 1. Check if the top-level classes are the same type
    if type(a) != type(b):
        return False
    
    try:
        a_leaves, a_treedef = jax.tree_util.tree_flatten(a)
        b_leaves, b_treedef = jax.tree_util.tree_flatten(b)
    except Exception:
        return False
        
    if a_treedef != b_treedef:
        return False
        
    # 2. Compare the flattened leaves safely
    for la, lb in zip(a_leaves, b_leaves):
        
        # Check if both leaves are some form of numeric/boolean data
        is_num_a = isinstance(la, (jnp.ndarray, np.ndarray, float, int, bool))
        is_num_b = isinstance(lb, (jnp.ndarray, np.ndarray, float, int, bool))
        
        if is_num_a and is_num_b:
            # Normalize to arrays and strip empty dimensions 
            # (e.g., 5.0 and [[5.0]] both become dimensionless scalars)
            arr_a = jnp.squeeze(jnp.asarray(la))
            arr_b = jnp.squeeze(jnp.asarray(lb))
            
            if arr_a.shape != arr_b.shape:
                return False
                
            # array_equal safely handles identical contents and NaNs
            if not jnp.array_equal(arr_a, arr_b, equal_nan=True):
                return False
                
        else:
            # Fallback for strings, None, or custom object leaves
            if la != lb:
                return False
                
    return True

def compute_tree_delta(old_tree, new_tree):
    """Find changes between two identically structured PyTrees."""
    old_leaves, _ = jax.tree_util.tree_flatten(old_tree)
    new_leaves, _ = jax.tree_util.tree_flatten(new_tree)

    changed_indices = []
    changed_leaves = []

    for i, (old, new) in enumerate(zip(old_leaves, new_leaves)):
        # Handle unchanged leaves
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
    old_leaves, treedef = jax.tree_util.tree_flatten(base_tree)
    new_leaves = list(old_leaves)
    for idx, leaf in zip(delta_indices, delta_leaves):
        new_leaves[idx] = leaf

    return jax.tree_util.tree_unflatten(treedef, new_leaves)

def io_partition(tree, active_ids: set[int]):
    """
    Partitions a PyTree in dynamic and static halves based on an IO whitelist.
    Only JAX arrays whose paths are in the whitelist are kept dynamic.

    Returns (dyn_tree, stat_tree)
    """

    def is_active(leaf):
        return eqx.is_array_like(leaf) and id(leaf) in active_ids

    mask = jax.tree_util.tree_map(is_active, tree)
    dyn, stat = eqx.partition(tree, mask)
    return dyn, stat, mask

def inspect_leaves(tree, mask, settings, tree_name:str="Tree", depth:int=3):
    """
    Groups PyTree leaves by their hierarchical path and writes the summary 
    to the terminal, a file, or both based on a boolean mask.
    """
    # Flatten both the tree and the boolean mask.
    # Because mask was generated from tree, their structures match perfectly.
    leaves_with_path, _ = jax.tree_util.tree_flatten_with_path(tree)
    mask_leaves, _ = jax.tree_util.tree_flatten(mask)
    
    summary = {}
    
    for (path, leaf), is_kept in zip(leaves_with_path, mask_leaves):
        # Convert JAX path keys into a readable string
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
                
        # Truncate to depth
        prefix = tree_name + "".join(path_strs[:depth])
        if not prefix:
            prefix = tree_name
            
        if prefix not in summary:
            summary[prefix] = {"kept": 0, "pruned": 0, "types": set()}
            
        # Check pruning logic directly from the mask!
        if not is_kept:
            summary[prefix]["pruned"] += 1
        else:
            summary[prefix]["kept"] += 1
            # Track the type of the kept leaves to spot rogue scalars
            leaf_type = type(leaf).__name__
            summary[prefix]["types"].add(leaf_type)
            
    # Format the output table
    lines = []
    header = f"{'PyTree Path (Depth ' + str(depth) + ')':<{35 + 15 * depth}} | {'Kept':<6} | {'Pruned':<6} | {'Common Kept Types'}"
    lines.append(header)
    lines.append("-" * 100)
    
    for prefix, counts in sorted(summary.items()):
        if counts['kept'] > 0 or counts['pruned'] > 0:
            types_str = ", ".join(sorted(list(counts['types']))[:3])
            lines.append(f"{prefix:<{35 + 15 * depth}} | {counts['kept']:<6} | {counts['pruned']:<6} | {types_str}")
            
    output_text = "\n".join(lines)
    
    # Route the output (Note: check if stream_ouput should be stream_output in your settings schema!)
    if settings.logging.stream_ouput:
        print("\n" + output_text)
        
    if settings.logging.log_dir is not None:
        # Ensure the directory exists
        output_file = Path(settings.logging.log_dir) / f"{tree_name}_structure.log"
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(output_text)
        if getattr(settings, 'verbose', False):
            print(f"\n - {tree_name.title()} leaf structure log saved to {output_file}")

#----------------------------------------------------------
# Saving and Loading
#----------------------------------------------------------

Trace_REGISTRY = {}

def register(cls):
    """Decorator to safely register any Trace class for standalone serialization."""
    if cls.__name__ in Trace_REGISTRY:
        raise ValueError(f"Trace class '{cls.__name__}' is already registered.")
    Trace_REGISTRY[cls.__name__] = cls
    return cls

def serialize_Trace_node(obj):
    """Recursively walks data, skipping attributes that match class defaults."""
    
    # 1. JAX or Numpy Array
    if isinstance(obj, (jnp.ndarray, np.ndarray)):
        if obj.size == 1:
            return obj.item()
        else:
            return {"__type__": "ndarray", "data": obj.tolist()}
        
    # 2. Registered Trace Class
    elif type(obj).__name__ in Trace_REGISTRY:
        cls = type(obj)
        state = {}
        
        # Attempt to conjure a default instance to compare against
        try:
            default_obj = cls()
            has_default = True
        except TypeError:
            # Class requires mandatory init arguments; we must save everything
            default_obj = None
            has_default = False
            
        for k, v in obj.__dict__.items():
            if k.startswith("__"): 
                continue
                
            # If we have a default instance, check if the current value matches it
            if has_default:
                default_v = getattr(default_obj, k, None)
                if is_equivalent(v, default_v):
                    continue  # SKIP SAVING! Massive file size reduction.
                    
            state[k] = serialize_Trace_node(v)
            
        return {"__class__": cls.__name__, "state": state}
        
    # 3. Standard Python Containers
    elif isinstance(obj, list):
        return {"__type__": "list", "data": [serialize_Trace_node(i) for i in obj]}
    elif isinstance(obj, tuple):
        return {"__type__": "tuple", "data": [serialize_Trace_node(i) for i in obj]}
    elif isinstance(obj, dict):
        return {"__type__": "dict", "data": {k: serialize_Trace_node(v) for k, v in obj.items()}}
        
    # 4. Standard Scalars
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
        
    else:
        if hasattr(obj, "tag") and obj.tag:
            warnings.warn(
                f"Attempted to save '{obj.tag}' with unregisterd class {type(obj).__name__}. "
                "Trace will be unable to load this data until the class is registered.", UserWarning)
        else:
            warnings.warn(
                f"Attempted to save '{obj}' with unregisterd class {type(obj).__name__}. "
                "Trace will be unable to load this data until the class is registered.", UserWarning)
        return {"__type__": "unknown", "data": str(obj)}

def deserialize_Trace_node(data):
    """Unpacks JSON, relying on default initializers to fill in missing attributes."""
    if not isinstance(data, dict):
        return data
        
    # 1. Reconstruct Trace Classes
    if "__class__" in data:
        cls_name = data["__class__"]
        
        if cls_name not in Trace_REGISTRY:
            raise ValueError(f"Class '{cls_name}' is not a registered Trace class and cannot be loaded.")
            
        cls = Trace_REGISTRY[cls_name]
        
        # Conjure the instance
        try:
            # Try normal instantiation to get all default attributes (like 'Air')
            instance = cls()
        except TypeError:
            # If it required args, we know it didn't have a default state to skip,
            # so the JSON contains 100% of the attributes. Use __new__.
            instance = object.__new__(cls)
        
        # Overwrite defaults with any saved differences
        for k, v in data["state"].items():
            object.__setattr__(instance, k, deserialize_Trace_node(v))
            
        return instance
        
    # 2. Reconstruct Arrays
    elif data.get("__type__") == "ndarray":
        return jnp.array(data["data"])
        
    # 3. Reconstruct Containers
    elif data.get("__type__") == "list":
        return [deserialize_Trace_node(i) for i in data["data"]]
    elif data.get("__type__") == "tuple":
        return tuple(deserialize_Trace_node(i) for i in data["data"])
    elif data.get("__type__") == "dict":
        return {k: deserialize_Trace_node(v) for k, v in data["data"].items()}
        
    return data

def save_data(obj, filename:str | Path):
    """
    Serializes any registered Trace data structure and compresses it to a file.
    """

    file_path = Path(filename).resolve()
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        payload = serialize_Trace_node(obj)
    
    with gzip.open(filename, 'wt', encoding='utf-8') as f:
        json.dump(payload, f)
        
    if hasattr(obj, "tag") and obj.tag:
        print(f"Successfully saved {type(obj).__name__} '{obj.tag}' to {file_path}")
    else:
        print(f"Successfully saved {type(obj).__name__} to {file_path}")


def load_data(filename: str | Path) -> Any:
    """
    Loads any Trace data structure from a file.
    No setup scripts or templates are required.
    """
    with gzip.open(filename, 'rt', encoding='utf-8') as f:
        payload = json.load(f)
        
    obj = deserialize_Trace_node(payload)
    
    if hasattr(obj, "tag") and obj.tag:
        print(f"Successfully loaded {type(obj).__name__} '{obj.tag}' from {filename}")
    else:
        print(f"Successfully loaded {type(obj).__name__} from {filename}")
    
    return obj

# ---------------------------------------------------------
# Debugging Tools
# ---------------------------------------------------------

def configure_environment(settings: Settings):
    """
    Configures global JAX and XLA compiler flags. 
    MUST be called at the very top of your script before any arrays are created.
    """

    dev_mode = settings._DEV_MODE
    debug_mode = settings.DEBUG_MODE

    if dev_mode:
        # os.environ["XLA_FLAGS"] = "--xla_backend_optimization_level=0"
        # os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["JAX_LOGGING_LEVEL"] = "DEBUG"
        os.environ["JAX_DEBUG_LOG_MODULES"] = "jax._src.compiler, jax._src.lru_cache"
        os.environ["JAX_EXPLAIN_CACHE_MISSES"] = "1"
        
    if debug_mode:
        jax.config.update("jax_disable_jit", True)
        jax.config.update("jax_debug_nans", True)
        print("TRACE WARNING: Debug mode is active. JIT disabled and NaN debugging enabled.")
    else:
        jax.config.update("jax_disable_jit", False)
        jax.config.update("jax_debug_nans", False)

def scan_for_invalid_JAX_types(pytree, name="PyTree") -> None:
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
                if hasattr(p, "name"):
                    path_str += f".{p.name}"
                elif hasattr(p, "key"):
                    path_str += f"[{repr(p.key)}]"
                elif hasattr(p, "idx"):
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
# JAX Caching
# ---------------------------------------------------------

def initialize_jax_cache(
    cache_dir="~/.eden_trace/jax_cache", 
    max_size_gb=2.0, 
    max_age_days=30
):
    """
    Initializes the JAX persistent compilation cache and prunes old entries.
    Safe to call every time Trace is imported.
    """
    # 1. Resolve the absolute path and ensure it exists
    cache_path = os.path.expanduser(cache_dir)
    os.makedirs(cache_path, exist_ok=True)
    
    # 2. Tell JAX to route all compiled XLA binaries here
    jax.config.update("jax_compilation_cache_dir", cache_path)
    
    # 3. Silently prune the cache so we don't blow up the user's hard drive
    try:
        _prune_cache(cache_path, max_size_gb, max_age_days)
    except Exception as e:
        # Never let a cache cleanup error crash the main physics library
        print(f"Trace Warning: Failed to prune JAX compilation cache - {e}")

def _prune_cache(cache_path, max_size_gb, max_age_days):
    """
    Implements an LRU (Least Recently Used) eviction policy.
    """
    max_size_bytes = max_size_gb * (1024 ** 3)
    max_age_seconds = max_age_days * 24 * 3600
    now = time.time()
    
    files = []
    total_size = 0
    
    # Scan the directory
    for filename in os.listdir(cache_path):
        filepath = os.path.join(cache_path, filename)
        if os.path.isfile(filepath):
            stat = os.stat(filepath)
            # Use st_atime (Last Accessed Time) for LRU, fallback to modified time
            last_accessed = stat.st_atime
            age = now - last_accessed
            size = stat.st_size
            
            files.append((filepath, age, size))
            total_size += size
            
    # Phase 1: Age-based Eviction (Delete anything older than max_age_days)
    files_to_keep = []
    for filepath, age, size in files:
        if age > max_age_seconds:
            os.remove(filepath)
            total_size -= size
        else:
            files_to_keep.append((filepath, age, size))
            
    # Phase 2: Size-based LRU Eviction (If still too big, delete oldest accessed first)
    if total_size > max_size_bytes:
        # Sort descending by age (oldest accessed at the front of the list)
        files_to_keep.sort(key=lambda x: x[1], reverse=True)
        
        for filepath, age, size in files_to_keep:
            if total_size <= max_size_bytes:
                break # We are back under the size limit
            os.remove(filepath)
            total_size -= size

# ---------------------------------------------------------
# General Mathematical Utilities
# ---------------------------------------------------------


@jax.jit
def cubic_spline_blender(x, start, end):

    eta = (x - start) / (end - start)
    eta_clamped = jnp.clip(eta, 0.1, 1.0)
    y = -2.0 * eta_clamped**3 + 3.0 * eta_clamped**2
    return y


if __name__ == "__main__":
    print(get_trace_root())
