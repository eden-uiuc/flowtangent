import abc

from .. import Module
from ..utils.typing import _

import jax
import jax.numpy as jnp

class _Spline(Module):

    control_points: jax.Array = _
    dim: int = 3

    def __check_init__(self):
        assert self.control_points.ndim == 2
        object.__setattr__(self, "dim", self.control_points.shape[-1])
        return super().__check_init__()

    @abc.abstractmethod
    def evaluate(self, s: float):
        pass # TODO: implement evaluation method

class BSpline(_Spline):

    def evaluate(self, s: float):
        return super().evaluate(s)

class BezierSpline(_Spline):

    def evaluate(self, s: float):
        return super().evaluate(s)

__all__ = [
    "BSpline",
    "BezierSpline"
]