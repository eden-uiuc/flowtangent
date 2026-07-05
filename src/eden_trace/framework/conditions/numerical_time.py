# Trace/Framework/Missions/Conditions/Numerics.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Jul 2024, Trace Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from typing import Callable

# package imports
import jax.numpy as jnp
import equinox as eqx

# Trace imports
from src.eden_trace.utils import empty_array, init_field, register

from src.eden_trace.framework.conditions import Condition

# ----------------------------------------------------------------------------------------------------------------------
#  Time Conditions
# ----------------------------------------------------------------------------------------------------------------------


@register
class NumericalTime(Condition):
    control_points: jnp.ndarray = empty_array(0)
    differentiate: jnp.ndarray = empty_array(0)
    integrate: jnp.ndarray | None = None

    def __repr__(self):
        return ""


@register
class Time(Condition):
    tag: str = init_field("Time", static=True)

    dimensionless: NumericalTime = init_field(lambda: NumericalTime(tag="Dimensionless Time"))
    dimensional: NumericalTime = init_field(lambda: NumericalTime(tag="Dimensional Time"))

    def update_chebyshev_matrices(
        self,
        n_cp: int = 16,
    ):

        assert n_cp > 0, "Attempted to calculate Chebyshev matrices with non-positive number of control points."

        # Find Chebyshev-Gauss-Lobatto Control Point Nodes
        x = 0.5 * (1 - jnp.cos(jnp.pi * jnp.arange(n_cp) / (n_cp - 1)))

        # Assume a Lagrange polynomial of degree N-1.
        # Exact derivative can be found as: F' = D * F
        # D_ij = c_i/c_j * (-1)^(i+j)/(x_i - x_j) for i !=j
        # c_i = 2 if i=0 or N-1; c_i = 1 otherwise
        # See Trefethen, L.N. "Spectral Methods in MATLAB"

        # c_neg = c_i*(-1)^i
        c_neg = jnp.array([2.0] + [1.0] * (n_cp - 2) + [2.0])
        c_neg *= (-1.0) ** jnp.arange(n_cp)
        c_neg_inv = 1.0 / c_neg

        # Calculate x_i - x_j as matrix operation
        A = jnp.tile(x, (n_cp, 1)).T
        dA = A - A.T + jnp.eye(n_cp)

        # c = c_neg * c_neg_inv = c_i/c_j*(-1)^(i-j) = c_i/c_j * (-1)^(i+j)
        c = jnp.multiply(jnp.atleast_2d(c_neg), jnp.atleast_2d(c_neg_inv).T)

        # D = c_i/c_j * (-1)^(i+j) / (x_i - x_j) -> c/dA
        # Diagnoal term must exactly cancel the rest of the row
        # Recompute diagonal to avoid floating point errors
        D = jnp.divide(c.T, dA)
        D -= jnp.diag(jnp.sum(D.T, axis=0))

        # Invert D, trimming first row and column
        I = jnp.linalg.inv(D[1:, 1:])

        # Repack missing columns with zeros
        I = jnp.append(jnp.zeros((1, n_cp - 1)), I, axis=0)
        I = jnp.append(jnp.zeros((n_cp, 1)), I, axis=1)

        updated_dimensionless = NumericalTime(
            tag="Dimensionless Time",
            control_points=jnp.atleast_2d(x).T,
            differentiate=D,
            integrate=I
        )

        updated_time = eqx.tree_at(
            lambda s: s.dimensionless,
            self,
            updated_dimensionless
        )

        return updated_time

    def __post_init__(self):
        # Guard against abstract tracers during JIT
        if self.number_of_control_points <= 1:
            return
