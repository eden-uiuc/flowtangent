"""Numeric display contracts for scalars, arrays, and unusual values."""

import jax.numpy as jnp
import numpy as np
import pytest

from flowtangent.utils.display import format_array


@pytest.mark.parametrize("value", [1.0, [1.0], np.array([[1.0]]), jnp.array([[[1.0]]])])
def test_single_element_inputs_use_scientific_notation(value):
    assert format_array(value) == " 1.000e+00"


@pytest.mark.parametrize(
    ("value", "precision", "width", "expected"),
    [
        (12.345, 2, 12, "    1.23e+01"),
        (-0.025, 1, 1, "-2.5e-02"),
        (0, 0, 1, "0e+00"),
        (True, 2, 1, "1.00e+00"),
        (float("nan"), 3, 5, "  nan"),
        (float("inf"), 3, 5, "  inf"),
        (float("-inf"), 3, 5, " -inf"),
    ],
)
def test_scalar_precision_width_and_nonfinite_values(value, precision, width, expected):
    assert format_array(value, precision=precision, width=width) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [([], "[]"), ([1, 2, 3], "[1, 2, 3]"), ([[1.25, 2.5], [3.0, 4.125]], "[[1.25, 2.5 ],\n [3.  , 4.12]]")],
)
def test_multiple_elements_keep_array_layout_and_ignore_width(value, expected):
    assert format_array(value, precision=2, width=100) == expected


def test_large_arrays_respect_numpy_truncation_settings():
    with np.printoptions(threshold=5, edgeitems=2):
        assert format_array(np.arange(10)) == "[0, 1, ..., 8, 9]"


def test_non_numeric_single_element_is_rejected():
    with pytest.raises(ValueError):
        format_array("not numeric")


def test_invalid_scalar_precision_is_rejected():
    with pytest.raises(ValueError):
        format_array(1.0, precision=-1)
