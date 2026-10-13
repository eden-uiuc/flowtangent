"""Behavioral tests for the documented module and field helpers."""

from enum import Enum
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import TypeCheckError

from flowtangent.utils import base
from flowtangent.utils.base import Module, StateDataMeta, empty_array, field, method_field, null_step, static_field
from flowtangent.utils.typing import AnyArray, TimeScalar, TimeVector3, _


@pytest.fixture
def isolated_registry(monkeypatch):
    """Keep test subclasses from changing the application's class registry."""
    monkeypatch.setattr(base, "FLOWTANGENT_REGISTRY", dict(base.FLOWTANGENT_REGISTRY))
    return base.FLOWTANGENT_REGISTRY


@pytest.mark.parametrize("args", [(), (object(),), (object(), [], {"key": 1})])
def test_null_step_preserves_argument_identity(args):
    result = null_step(*args)

    assert isinstance(result, tuple)
    assert len(result) == len(args)
    assert all(actual is expected for actual, expected in zip(result, args))


@pytest.mark.parametrize("default", [None, False, 0, 2.5, "value", (1, 2)])
def test_field_accepts_immutable_defaults(default):
    class Example(eqx.Module):
        value: Any = field(default)

    assert Example().value is default


@pytest.mark.parametrize("mutable", [[], {}, set()])
def test_field_rejects_mutable_instances(mutable):
    with pytest.raises(ValueError, match="Mutable instance.*Pass the uninstantiated class"):
        field(mutable)


@pytest.mark.parametrize("factory", [list, dict, set])
def test_field_factories_produce_independent_defaults(factory):
    class Example(eqx.Module):
        value: Any = field(factory)

    first, second = Example(), Example()

    assert isinstance(first.value, factory)
    assert first.value == second.value
    assert first.value is not second.value


def test_field_calls_factory_only_on_instantiation():
    calls = []

    def factory():
        calls.append(True)
        return {"instance": len(calls)}

    class Example(eqx.Module):
        value: dict = field(factory)

    assert calls == []
    assert Example().value == {"instance": 1}
    assert Example().value == {"instance": 2}


def test_field_preserves_jax_array_default():
    initial = jnp.array([1.0, 2.0])

    class Example(eqx.Module):
        value: jax.Array = field(initial)

    assert Example().value is initial
    np.testing.assert_allclose(Example().value, [1.0, 2.0])


def test_as_func_preserves_initializer_instead_of_calling_it():
    def stored_callback():
        pytest.fail("A stored callable must not run during initialization")

    class Example(eqx.Module):
        callback: Any = field(stored_callback, as_func=True)

    assert Example().callback is stored_callback


def test_static_and_method_fields_stay_out_of_dynamic_leaves():
    def stored_callback(value):
        return value + 1

    class Example(eqx.Module):
        value: int = field(3)
        label: str = static_field("metadata")
        callback: Any = method_field(stored_callback)

    instance = Example()

    assert jax.tree_util.tree_leaves(instance) == [3]
    assert instance.label == "metadata"
    assert instance.callback is stored_callback
    assert instance.callback(2) == 3


def test_field_forwards_keyword_arguments():
    class Example(eqx.Module):
        value: int = field(7, kw_only=True)

    assert Example(value=9).value == 9
    with pytest.raises(TypeError):
        Example(9)


@pytest.mark.parametrize("shape", [0, 3, (0, 3), (2, 0, 4), ()])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.int32, jnp.bool_])
def test_empty_array_preserves_shape_and_dtype(shape, dtype):
    class Example(eqx.Module):
        value: jax.Array = empty_array(shape, dtype)

    value = Example().value

    assert value.shape == ((shape,) if isinstance(shape, int) else shape)
    assert value.dtype == jnp.dtype(dtype)


def test_module_registers_subclass_and_defaults_name(isolated_registry):
    class NamedExample(Module):
        pass

    instance = NamedExample()

    assert isolated_registry["NamedExample"] is NamedExample
    assert instance.name == "NamedExample"
    assert repr(instance) == "NamedExample"
    assert instance.field_name == "namedexample"


@pytest.mark.parametrize(
    ("name", "expected"),
    [("Main Wing", "main_wing"), ("  2 Wing-A!  ", "__2_wing-a!__"), ("", ""), ("Wing\tTip", "wing\ttip")],
)
def test_module_field_name_normalizes_only_spaces_and_case(name, expected):
    module = Module(name=name)

    assert module.field_name == expected
    assert module.name == name
    assert repr(module) == name


def test_module_field_name_resolves_enum_values():
    class Names(Enum):
        WING = "Main Wing"

    assert Module(name=Names.WING).field_name == "main_wing"


def test_module_field_name_rejects_unresolvable_name():
    with pytest.raises(AttributeError, match="Unable to resolve field name"):
        _ = Module(name=object()).field_name


def test_module_rejects_duplicate_class_names(isolated_registry):
    first = type("RegistryExample", (Module,), {"__module__": "first_test_module"})

    with pytest.raises(ValueError, match="Class 'RegistryExample' is already registered"):
        type("RegistryExample", (Module,), {"__module__": "second_test_module"})

    assert isolated_registry["RegistryExample"] is first


def test_module_typechecks_annotated_methods(isolated_registry):
    class CheckedExample(Module):
        def increment(self, value: int) -> int:
            return value + 1

        def unchecked(self, value):
            return value

        def wrong_return(self) -> int:
            return "incorrect"

    instance = CheckedExample()

    assert instance.increment(2) == 3
    assert instance.unchecked("allowed") == "allowed"
    with pytest.raises(TypeCheckError):
        instance.increment("incorrect")
    with pytest.raises(TypeCheckError):
        instance.wrong_return()


def test_state_metaclass_shapes_missing_and_placeholder_arrays(isolated_registry):
    class StateExample(Module, metaclass=StateDataMeta):
        scalar: TimeScalar
        vector: TimeVector3 = _
        arbitrary: AnyArray = _
        count: int = 4

    state = StateExample()

    assert state.scalar.shape == (0, 1)
    assert state.vector.shape == (0, 3)
    assert state.arbitrary.shape == (0,)
    assert state.count == 4


def test_state_metaclass_preserves_explicit_array_defaults(isolated_registry):
    initial = jnp.ones((2, 3))

    class ExplicitStateExample(Module, metaclass=StateDataMeta):
        vector: TimeVector3 = field(lambda: initial)

    assert ExplicitStateExample().vector is initial
