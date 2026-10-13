import gzip
import itertools
import json
import logging
import os
import string
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from beartype.typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from .base import FLOWTANGENT_REGISTRY, Module, field

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

    required_keys = [tup[1] for tup in string.Formatter().parse(io_string) if tup[1] is not None]

    if not required_keys:
        return {io_string}

    normalized_map = {}
    for full_key in required_keys:
        safe_key = full_key.replace(".", "___")
        io_string = io_string.replace(f"{{{full_key}}}", f"{{{safe_key}}}")

        parts = full_key.split(".")
        base_key = parts[0]
        attrs = parts[1:]

        base_val = var_map.get(base_key, []) if isinstance(var_map, dict) else getattr(var_map, base_key, [])
        base_list = [base_val] if isinstance(base_val, (str, int, float, bool)) else list(base_val)

        resolved_list = []
        for item in base_list:
            try:
                resolved_item = item
                for attr in attrs:
                    if attr.endswith("()"):
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
        if hasattr(p, "name"):
            path_str += f".{p.name}"
        elif hasattr(p, "key"):
            path_str += f"['{p.key}']"
        elif hasattr(p, "idx"):
            path_str += f"[{p.idx}]"
    return path_str.lstrip(".")


# ----------------------------------------------------------
# Saving and Loading
# ----------------------------------------------------------


def _ft_root() -> Path:
    """Returns the absolute path to the src/flowtangent directory."""
    # .parent steps up from src/flowtangent/utils to src/flowtangent
    return Path(os.path.dirname(os.path.abspath(__file__))).resolve().parent


def serialize_node(obj):
    if isinstance(obj, (jax.Array, np.ndarray)):
        if obj.size == 1:
            return obj.item()
        return {"__type__": "ndarray", "data": obj.tolist()}

    elif type(obj).__name__ in FLOWTANGENT_REGISTRY:
        cls = type(obj)
        state = {}

        try:
            default_obj = cls()
            has_default = True
        except TypeError:
            default_obj = None
            has_default = False

        for k, v in obj.__dict__.items():
            if k.startswith("__"):
                continue

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
        if cls_name not in FLOWTANGENT_REGISTRY:
            raise ValueError(f"Class '{cls_name}' is not registered and cannot be loaded.")

        cls = FLOWTANGENT_REGISTRY[cls_name]
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

    with gzip.open(file_path, "wt", encoding="utf-8") as f:
        json.dump(payload, f)

    name = getattr(obj, "name", "")
    print(
        f"Successfully saved {type(obj).__name__} '{name}' to {file_path}"
        if name
        else f"Successfully saved {type(obj).__name__} to {file_path}"
    )


def load_data(filename: str | Path) -> Any:
    with gzip.open(filename, "rt", encoding="utf-8") as f:
        payload = json.load(f)

    obj = deserialize_node(payload)
    name = getattr(obj, "name", "")
    print(
        f"Successfully loaded {type(obj).__name__} '{name}' from {filename}"
        if name
        else f"Successfully loaded {type(obj).__name__} from {filename}"
    )
    return obj


# ----------------------------------------------------------
# Logging
# ----------------------------------------------------------


class JAXCompileFilter(logging.Filter):
    def __init__(self, name: str = "", whitelist: Optional[tuple[str]] = None) -> None:
        super().__init__(name)
        self.whitelist = whitelist

    def filter(self, record):
        msg = record.getMessage()

        # 1. Identify if this is a compilation/tracing log
        is_compile_log = any(
            keyword in msg
            for keyword in ["Compiling", "tracing + transforming", "Finished jaxpr to MLIR", "Finished XLA compilation"]
        )

        # If it is a compile log, apply whitelist & formatting
        if is_compile_log and self.whitelist is not None:
            # Block it if it's not the main solve
            if not any([f"jit({w})" in msg for w in self.whitelist]):
                return False

            # If it is the main solve, truncate the massive PyTree dump
            if "with global shapes and types" in msg:
                parts = msg.split("with global shapes and types")
                prefix = parts[0] + "with global shapes and types"
                suffix = parts[1][:30] if len(parts) > 1 else ""

                record.msg = f"{prefix} {suffix} ... [PyTree Truncated]"
                record.args = ()

            return True

        # If it's NOT a compile log (e.g., GPU memory warning), let it through untouched
        return True


class LoggingSettings(Module):
    handle: Optional[str] = field(None, static=True)
    log_dir: Optional[str | Path] = field(None, static=True)

    format_string: str = field("[%(asctime)s] - %(levelname)s - %(message)s", static=True)
    date_format: str = field("%Y-%m-%d %H:%M:%S", static=True)
    stream_ouput: bool = field(False, static=True)
    jax_logging: bool = field(False, static=True)
    jax_compile_whitelist: Optional[tuple[str]] = field(None, static=True)

    def setup_logger(self, handle: Optional[str] = None) -> None:
        if self.log_dir is None and not self.stream_ouput:
            return
        else:
            if handle is None and self.handle is None:
                log_handle = "flowtangent"
            else:
                log_handle = handle if handle is not None else self.handle
            logger = logging.getLogger(log_handle)
            formatter = logging.Formatter(self.format_string)
            handlers = []

            if self.stream_ouput:
                sh = logging.StreamHandler()
                sh.setLevel(logging.INFO)
                sh.setFormatter(formatter)
                handlers.append(sh)

            if self.log_dir is not None:
                log_dir = Path(self.log_dir)
                log_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime(self.date_format).replace(" ", "_").replace(":", "-")
                logfile = log_dir / f"main_{timestamp}.log"
                fh = logging.FileHandler(logfile)
                fh.setLevel(logging.INFO)
                fh.setFormatter(formatter)
                handlers.append(fh)

            for h in handlers:
                logger.addHandler(h)

            if self.jax_logging:
                jl = logging.getLogger("jax")
                jl.propagate = False
                jl.handlers.clear()
                j_filter = JAXCompileFilter("jax_compile_filter", self.jax_compile_whitelist)

                for h in handlers:
                    h.addFilter(j_filter)
                    jl.addHandler(h)

                if getattr(jax.config, "jax_log_compiles", False):
                    jl.setLevel(logging.INFO)
                else:
                    jl.setLevel(logging.WARNING)

            return
