import gzip
import itertools
import json
import os
import string
import warnings
from pathlib import Path
from typing import Any, Callable

import jax.numpy as jnp
import numpy as np

# Import our tree utility for checking defaults during serialization
from .tree import is_equivalent

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

def parse_io(io_string: str, var_map: dict | Any) -> set:
    io_parts = io_string.split(":")
    io_string = io_parts[0].strip()

    required_keys = [
        tup[1] for tup in string.Formatter().parse(io_string)
        if tup[1] is not None
    ]

    if not required_keys:
        return {io_string}

    normalized_map = {}
    for full_key in required_keys:
        safe_key = full_key.replace('.', '___')
        io_string = io_string.replace(f"{{{full_key}}}", f"{{{safe_key}}}")

        parts = full_key.split('.')
        base_key = parts[0]
        attrs = parts[1:]

        base_val = var_map.get(base_key, []) if isinstance(var_map, dict) else getattr(var_map, base_key, [])
        base_list = [base_val] if isinstance(base_val, (str, int, float, bool)) else list(base_val)

        resolved_list = []
        for item in base_list:
            try:
                resolved_item = item
                for attr in attrs:
                    if attr.endswith('()'):
                        resolved_item = getattr(resolved_item, attr[:-2])()
                    else:
                        resolved_item = getattr(resolved_item, attr)
                resolved_list.append(resolved_item)
            except AttributeError:
                raise AttributeError(f"Could not resolve '{'.'.join(attrs)}' on {item}")

        normalized_map[safe_key] = resolved_list

    keys = list(normalized_map.keys())
    value_lists = list(normalized_map.values())

    resolved_paths = set()
    for combination in itertools.product(*value_lists):
        combo_dict = dict(zip(keys, combination))
        resolved_paths.add(io_string.format(**combo_dict))

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

# ----------------------------------------------------------
# Saving and Loading
# ----------------------------------------------------------

def _ft_root() -> Path:
    """Returns the absolute path to the src/flowtangent directory."""
    # .parent steps up from src/flowtangent/utils to src/flowtangent
    return Path(os.path.dirname(os.path.abspath(__file__))).resolve().parent

FlowTangent_REGISTRY = {}

def register(cls):
    """Decorator to safely register any class for standalone serialization."""
    if cls.__name__ in FlowTangent_REGISTRY:
        raise ValueError(f"Class '{cls.__name__}' is already registered.")
    FlowTangent_REGISTRY[cls.__name__] = cls
    return cls

def serialize_node(obj):
    if isinstance(obj, (jnp.ndarray, np.ndarray)):
        if obj.size == 1:
            return obj.item()
        return {"__type__": "ndarray", "data": obj.tolist()}

    elif type(obj).__name__ in FlowTangent_REGISTRY:
        cls = type(obj)
        state = {}

        try:
            default_obj = cls()
            has_default = True
        except TypeError:
            default_obj = None
            has_default = False

        for k, v in obj.__dict__.items():
            if k.startswith("__"): continue

            if has_default:
                default_v = getattr(default_obj, k, None)
                if is_equivalent(v, default_v):
                    continue

            state[k] = serialize_node(v)

        return {"__class__": cls.__name__, "state": state}

    elif isinstance(obj, list):
        return {"__type__": "list", "data": [serialize_node(i) for i in obj]}
    elif isinstance(obj, tuple):
        return {"__type__": "tuple", "data": [serialize_node(i) for i in obj]}
    elif isinstance(obj, dict):
        return {"__type__": "dict", "data": {k: serialize_node(v) for k, v in obj.items()}}

    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj

    else:
        name = getattr(obj, "name", str(obj))
        warnings.warn(f"Attempted to save '{name}' with unregistered class {type(obj).__name__}.", UserWarning)
        return {"__type__": "unknown", "data": str(obj)}

def deserialize_node(data):
    if not isinstance(data, dict):
        return data

    if "__class__" in data:
        cls_name = data["__class__"]
        if cls_name not in FlowTangent_REGISTRY:
            raise ValueError(f"Class '{cls_name}' is not registered and cannot be loaded.")

        cls = FlowTangent_REGISTRY[cls_name]
        try:
            instance = cls()
        except TypeError:
            instance = object.__new__(cls)

        for k, v in data["state"].items():
            object.__setattr__(instance, k, deserialize_node(v))

        return instance

    elif data.get("__type__") == "ndarray":
        return jnp.array(data["data"])
    elif data.get("__type__") == "list":
        return [deserialize_node(i) for i in data["data"]]
    elif data.get("__type__") == "tuple":
        return tuple(deserialize_node(i) for i in data["data"])
    elif data.get("__type__") == "dict":
        return {k: deserialize_node(v) for k, v in data["data"].items()}

    return data

def save_data(obj, filename: str | Path):
    file_path = Path(filename).resolve()
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        payload = serialize_node(obj)

    with gzip.open(file_path, 'wt', encoding='utf-8') as f:
        json.dump(payload, f)

    name = getattr(obj, "name", "")
    print(f"Successfully saved {type(obj).__name__} '{name}' to {file_path}" if name else f"Successfully saved {type(obj).__name__} to {file_path}")

def load_data(filename: str | Path) -> Any:
    with gzip.open(filename, 'rt', encoding='utf-8') as f:
        payload = json.load(f)

    obj = deserialize_node(payload)
    name = getattr(obj, "name", "")
    print(f"Successfully loaded {type(obj).__name__} '{name}' from {filename}" if name else f"Successfully loaded {type(obj).__name__} from {filename}")
    return obj
