# src/flowtangent/utils/typing.py
"""FlowTangent JAX-aware type aliases and dimension glossary.

This module provides strict shape and dtype annotations utilizing `jaxtyping`.
It defines the standardized tensor geometries expected by FlowTangent components
and solvers.

Dimension Glossary:
    time: The primary simulation time steps (`N_time`).
    _: An arbitrary, unconstrained dimension length.
    ...: Any number of dimensions (variadic wildcard).
"""

# ruff: noqa: F722
from typing import Any, Optional, Union, cast

from jaxtyping import Array, Bool, Float, Int, Shaped

from .tree import TreePath


class _Placeholder:
    """Sentinel class used by `StateDataMeta` to detect uninitialized attributes."""

    pass


_ = cast(Any, _Placeholder())
"""Any: A singleton instance of `_Placeholder` used as a default assignment `_ = _` to trigger array generation."""

# FT Metatypes

NameType = Union[str, None]
"""Union[str, None]: Dummy type to allow names to be defaulted to \\
    None for modules while still being parsed by registry."""

TreePathLike = Union[TreePath, str, tuple[tuple | str | TreePath, Any, Optional[slice]]]
"""Union[TreePath, str, tuple]: Represents a valid selector for traversing and updating Equinox module trees."""

# --- Scalars ---
# Accepts JAX 0D arrays, 1D arrays of size 1, or standard Python primitives.
# Used for Variable bounds, constants, and initial conditions.

ScalarFloat = Union[Float[Array, ""], Float[Array, "1"], float]
"""Union[Array, float]: A floating-point scalar, accepting standard floats or 0D/1D JAX arrays."""

ScalarInt = Union[Int[Array, ""], Float[Array, "1"], int]
"""Union[Array, int]: An integer scalar, accepting standard ints or 0D/1D JAX arrays. \\
Note: Accepts Float[Array, "1"] due to JAX's default casting behaviors in some optimizers."""

ScalarBool = Union[Bool[Array, ""], Float[Array, "1"], bool]
"""Union[Array, bool]: A boolean scalar, accepting standard bools or 0D/1D JAX arrays."""


# --- Time Series (N_time) ---
# Standard arrays for component states and ports

TimeScalar = Float[Array, "time 1"]
"""Float[Array, "time 1"]: A 2D column vector representing a single scalar metric tracked over time (N, 1)."""

TimeVector3 = Float[Array, "time 3"]
"""Float[Array, "time 3"]: A 2D array representing a 3D spatial vector tracked over time (N, 3)."""


# --- Generic / Escape Hatches ---
# For utilities like `ft.update` that must accept anything

AnyFloatArray = Float[Array, "..."]
"""Float[Array, "..."]: Any JAX array containing floating-point values of arbitrary shape."""

AnyArray = Shaped[Array, "..."]
"""Shaped[Array, "..."]: Any JAX array of any shape and dtype."""

__all__ = [
    "_",
    "NameType",
    "TreePathLike",
    "ScalarFloat",
    "ScalarInt",
    "ScalarBool",
    "TimeScalar",
    "TimeVector3",
    "AnyFloatArray",
    "AnyArray",
]
