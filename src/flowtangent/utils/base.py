# src/flowtangent/utils/base.py
import inspect
import typing
from typing import Any, dataclass_transform, get_args

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import jaxtyped

from .typing import NameType, _Placeholder


def null_step(*args):
    """A generic no-op function that returns its inputs unchanged."""
    return args


if typing.TYPE_CHECKING:
    T = typing.TypeVar("T")

    # Overload 1: If passed a class/callable, Pylance binds it to 'default_factory'
    @typing.overload
    def field(default_factory: typing.Callable[[], T], as_func: bool = False, **kwargs) -> T: ...

    # Overload 2: If passed a standard value, Pylance binds it to 'default'
    @typing.overload
    def field(default: T, as_func: bool = False, **kwargs) -> T: ...


def field(initializer: typing.Any = None, as_func: bool = False, **kwargs):
    """Smart wrapper for eqx.field that auto-routes default vs default_factory."""
    if as_func:
        return eqx.field(default_factory=lambda: initializer, **kwargs)
    if callable(initializer):
        return eqx.field(default_factory=initializer, **kwargs)
    if isinstance(initializer, (list, dict, set)):
        raise ValueError(
            f"Mutable instance {initializer} passed to init_field. "
            "Pass the uninstantiated class (e.g., list) or a lambda instead."
        )
    if isinstance(initializer, jnp.ndarray):
        return eqx.field(default_factory=lambda: initializer, **kwargs)
    return eqx.field(default=initializer, **kwargs)


def static_field(*args, **kwargs):
    return field(*args, static=True, **kwargs)


def method_field(*args, **kwargs):
    return field(*args, as_func=True, static=True, **kwargs)


def empty_array(shape: tuple | int = 0, dtype: Any = float, **kwargs):
    """Syntactic sugar for an empty JAX array in an Equinox module."""
    return field(lambda: jnp.empty(shape, dtype=dtype), **kwargs)


FLOWTANGENT_REGISTRY = {}


@dataclass_transform(field_specifiers=(eqx.field, static_field, method_field))
class Module(eqx.Module):
    """Base class for all FlowTangent modules."""

    name: NameType = static_field(None)

    def __check_init__(self):
        if self.name is None:
            object.__setattr__(self, "name", self.__class__.__name__)

    def __init_subclass__(cls, **kwargs) -> None:
        # Prevent kw_only being passed to object.__init_subclass__
        # It's used in eqx._ModuleMeta.__new__ only
        kwargs.pop("kw_only", None)

        if cls.__name__ in FLOWTANGENT_REGISTRY:
            existing_cls = FLOWTANGENT_REGISTRY[cls.__name__]
            if existing_cls is not cls:
                raise ValueError(
                    f"Class '{cls.__name__}' is already registered.\n"
                    f"  First registered by: {existing_cls.__module__}\n"
                    f"  Now registered by:   {cls.__module__}"
                )
        FLOWTANGENT_REGISTRY[cls.__name__] = cls

        dataclass_fields = getattr(cls, "__dataclass_fields__", {})

        # Auto-apply jaxtyped to all standard methods that have type annotations
        for attr_name, attr_value in cls.__dict__.items():
            if attr_name in dataclass_fields:
                continue  # Skip dataclass defaults
            # Skip dunder methods (__init__, __call__, etc.) to avoid breaking Equinox
            if inspect.isfunction(attr_value) and not attr_name.startswith("__"):
                annotations = getattr(attr_value, "__annotations__", {})
                # If the method has any annotations (return or args), wrap it
                if annotations:
                    wrapped_method = jaxtyped(typechecker=beartype)(attr_value)
                    setattr(cls, attr_name, wrapped_method)

        super().__init_subclass__(**kwargs)

    def __repr__(self) -> str:
        return f"{self.name}"

    @property
    def field_name(self):
        actual_name = self.name
        if actual_name and not isinstance(actual_name, str):
            if hasattr(actual_name, "value"):
                actual_name = actual_name.value
            else:
                raise AttributeError(f"Unable to resolve field name for {self}.")

        return str(actual_name).replace(" ", "_").lower()


class StateDataMeta(type(Module)):
    def __new__(mcs, name, bases, namespace):
        ### ADDED
        annotations = namespace.get("__annotations__", {})
        # Python 3.14 defers class annotations instead of putting them directly
        # in the namespace. Resolve them before replacing placeholder defaults.
        if not annotations and "__annotate_func__" in namespace:
            import annotationlib

            annotate = annotationlib.get_annotate_from_class_namespace(namespace)
            if annotate is not None:
                annotations = annotationlib.call_annotate_function(annotate, annotationlib.Format.FORWARDREF)
        ####
        for key, hint in annotations.items():
            if key.startswith("__"):
                continue

            args = get_args(hint)
            hint_str = str(hint) + "".join(str(a) for a in args)

            # Check if it has NO default OR if the user used the Ellipsis placeholder
            val = namespace.get(key)
            if key not in namespace or isinstance(val, _Placeholder):
                if "ndarray" in hint_str or "Array" in hint_str:
                    # Deduce the correct placeholder shape directly from the type hint
                    shape = (0,)
                    if "time 1" in hint_str:
                        shape = (0, 1)
                    elif "time 3" in hint_str:
                        shape = (0, 3)

                    namespace[key] = empty_array(shape)

        return super().__new__(mcs, name, bases, namespace)
