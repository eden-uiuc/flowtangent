# src/eden_trace/utils/base.py
from typing import Any, dataclass_transform
from functools import partial
import equinox as eqx
import jax.numpy as jnp

def field(initializer: Any, as_value: bool = False, **kwargs):
    """Smart wrapper for eqx.field that auto-routes default vs default_factory."""
    if as_value:
        return eqx.field(default=initializer, **kwargs)
    if callable(initializer):
        return eqx.field(default_factory=initializer, **kwargs)
    if isinstance(initializer, (list, dict, set)):
        raise ValueError(
            f"Mutable instance {initializer} passed to init_field. "
            "Pass the uninstantiated class (e.g., list) or a lambda instead."
        )
    return eqx.field(default=initializer, **kwargs)

static_field = partial(field, static=True)
method_field = partial(field, as_value=True, static=True)

def empty_array(shape: tuple | int = 0, dtype: Any = float, **kwargs):
    """Syntactic sugar for an empty JAX array in an Equinox module."""
    return field(lambda: jnp.empty(shape, dtype=dtype), **kwargs)

# Instruct IDEs to treat our custom base class as a dataclass generator
@dataclass_transform(field_specifiers=(eqx.field, field, static_field, method_field))
class Module(eqx.Module):
    """Base class for all FlowTangent modules to preserve IDE autocompletion."""
    pass

# Metaclass logic from earlier
class StateDataMeta(type(eqx.Module)):
    def __new__(mcs, name, bases, namespace):
        annotations = namespace.get('__annotations__', {})
        for key, hint in annotations.items():
            if key.startswith("__"): continue
            
            hint_str = str(hint)
            if ("ndarray" in hint_str or "Array" in hint_str) and key not in namespace:
                namespace[key] = empty_array()
                
        return super().__new__(mcs, name, bases, namespace)

class StateData(Module, metaclass=StateDataMeta):
    pass