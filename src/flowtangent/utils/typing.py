# ruff: noqa: F722
from typing import Any, Optional, Union, cast

from jaxtyping import Array, Bool, Float, Int, Shaped

from .tree import TreePath

# ==========================================
# FLOWTANGENT DIMENSION GLOSSARY
# ------------------------------------------
# time: The primary simulation time steps (N_time)
# _:    An arbitrary, unconstrained dimension length
# ...:  Any number of dimensions (variadic wildcard)
# ==========================================


# Used by metaclasses in utils.base to parse type hints into default arrays
class _Placeholder:
    pass


_ = cast(Any, _Placeholder())

# FT Metatypes
NameType = Union[str, None]
TreePathLike = Union[TreePath, str, tuple[tuple | str | TreePath, Any, Optional[slice]]]

# --- Scalars (Accepts JAX 0D arrays or Python primitives) ---
# Perfect for Variable bounds, constants, and initial conditions
ScalarFloat = Union[Float[Array, ""], Float[Array, "1"], float]
ScalarInt = Union[Int[Array, ""], Float[Array, "1"], int]
ScalarBool = Union[Bool[Array, ""], Float[Array, "1"], bool]

# --- Time Series (N_time) ---
# Standard arrays for component states and ports
TimeScalar = Float[Array, "time 1"]  # 2D column vector: (N, 1)
TimeVector3 = Float[Array, "time 3"]  # 3D spatial vector: (N, 3)

# --- Generic / Escape Hatches ---
# For utilities like `ft.update` that must accept anything
AnyFloatArray = Float[Array, "..."]
AnyArray = Shaped[Array, "..."]

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
