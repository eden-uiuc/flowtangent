"""Shape and dtype boundaries of the documented JAX type aliases."""

import jax.numpy as jnp
import pytest
from beartype.door import is_bearable

from flowtangent.utils.typing import (
    AnyArray,
    AnyFloatArray,
    ScalarBool,
    ScalarFloat,
    ScalarInt,
    TimeScalar,
    TimeVector3,
)


@pytest.mark.parametrize(
    ("annotation", "value", "expected"),
    [
        (ScalarFloat, 1.5, True),
        (ScalarFloat, jnp.array(1.5), True),
        (ScalarFloat, jnp.array([1.5]), True),
        (ScalarFloat, jnp.array([1.5, 2.5]), False),
        (ScalarFloat, jnp.array([[1.5]]), False),
        (ScalarFloat, jnp.array(1, dtype=jnp.int32), False),
        (ScalarInt, 1, True),
        (ScalarInt, jnp.array(1, dtype=jnp.int32), True),
        (ScalarInt, jnp.array([1.5]), True),
        (ScalarInt, jnp.array(1.5), False),
        (ScalarInt, jnp.array([1, 2], dtype=jnp.int32), False),
        (ScalarBool, True, True),
        (ScalarBool, jnp.array(False), True),
        (ScalarBool, jnp.array([1.0]), True),
        (ScalarBool, jnp.array([True]), False),
        (ScalarBool, jnp.array(1, dtype=jnp.int32), False),
    ],
)
def test_scalar_alias_shape_and_dtype_boundaries(annotation, value, expected):
    assert is_bearable(value, annotation) is expected


@pytest.mark.parametrize("annotation,columns", [(TimeScalar, 1), (TimeVector3, 3)])
@pytest.mark.parametrize("rows", [0, 1, 4])
def test_time_series_accepts_empty_and_nonempty_float_arrays(annotation, columns, rows):
    assert is_bearable(jnp.zeros((rows, columns), dtype=jnp.float32), annotation)


@pytest.mark.parametrize("annotation", [TimeScalar, TimeVector3])
@pytest.mark.parametrize("shape,dtype", [((3,), jnp.float32), ((2, 2), jnp.float32), ((2, 1, 3), jnp.float32)])
def test_time_series_rejects_incorrect_shapes(annotation, shape, dtype):
    assert not is_bearable(jnp.zeros(shape, dtype=dtype), annotation)


@pytest.mark.parametrize("annotation,columns", [(TimeScalar, 1), (TimeVector3, 3)])
def test_time_series_rejects_integer_dtype(annotation, columns):
    assert not is_bearable(jnp.zeros((2, columns), dtype=jnp.int32), annotation)


@pytest.mark.parametrize("shape", [(), (0,), (2, 3, 4)])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.int32, jnp.bool_])
def test_generic_aliases_distinguish_float_arrays_from_any_dtype(shape, dtype):
    array = jnp.zeros(shape, dtype=dtype)

    assert is_bearable(array, AnyArray)
    assert is_bearable(array, AnyFloatArray) is (dtype is jnp.float32)
