# RCAIDE/Framework/Missions/Conditions/Numerics.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Jul 2024, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from typing import Callable

# package imports
import jax.numpy as jnp

# RCAIDE imports
from RCAIDE.utils import empty_array, init_field

from RCAIDE.Framework.Conditions import Conditions

# ----------------------------------------------------------------------------------------------------------------------
#  Numerics
# ----------------------------------------------------------------------------------------------------------------------


def chebyshev_matrices(
    n: int = 16,
    calculate_integration: bool = True,
):

    assert n > 0, "Attempted to calculate Chebyshev matrices with non-positive number of control points."

    # Find Chebyshev-Gauss-Lobatto Control Point Nodes
    x = 0.5 * (1 - jnp.cos(jnp.pi * jnp.arange(n) / (n - 1)))

    # Assume a Lagrange polynomial of degree N-1.
    # Exact derivative can be found as: F' = D * F
    # D_ij = c_i/c_j * (-1)^(i+j)/(x_i - x_j) for i !=j
    # c_i = 2 if i=0 or N-1; c_i = 1 otherwise
    # See Trefethen, L.N. "Spectral Methods in MATLAB"

    # c_neg = c_i*(-1)^i
    c_neg = jnp.array([2.0] + [1.0] * (n - 2) + [2.0])
    c_neg *= (-1.0) ** jnp.arange(n)
    c_neg_inv = 1.0 / c_neg

    # Calculate x_i - x_j as matrix operation
    A = jnp.tile(x, (n, 1)).T
    dA = A - A.T + jnp.eye(n)

    # c = c_neg * c_neg_inv = c_i/c_j*(-1)^(i-j) = c_i/c_j * (-1)^(i+j)
    c = jnp.multiply(jnp.atleast_2d(c_neg), jnp.atleast_2d(c_neg_inv).T)

    # D = c_i/c_j * (-1)^(i+j) / (x_i - x_j) -> c/dA
    # Diagnoal term must exactly cancel the rest of the row
    # Recompute diagonal to avoid floating point errors
    D = jnp.divide(c.T, dA)
    D -= jnp.diag(jnp.sum(D.T, axis=0))

    if calculate_integration:
        # Invert D, trimming first row and column
        I = jnp.linalg.inv(D[1:, 1:])

        # Repack missing columns with zeros
        I = jnp.append(jnp.zeros((1, n - 1)), I, axis=0)
        I = jnp.append(jnp.zeros((n, 1)), I, axis=1)
    else:
        I = None

    return jnp.atleast_2d(x).T, D, I


class NumericalTime(Conditions):
    # Attribute     Type                Default Value
    control_points: jnp.ndarray = empty_array(0)
    differentiate: jnp.ndarray = empty_array(0)
    integrate: jnp.ndarray | None = None

    def __repr__(self):
        return ""


class Numerics(Conditions):
    # Attribute                 Type                Default Value
    tag: str = init_field("Numerics", static=True)

    number_of_control_points: int = init_field(16, static=True)
    control_point_spacing: str = init_field("cosine", static=True)
    calculate_integration: bool = init_field(True, static=True)
    discretization_method: Callable | None = init_field(None, static=True)

    solver_jacobian: str | None = init_field(None, static=True)
    solution_tolerance: float = init_field(1e-8, static=True)
    max_evaluations: int = init_field(500, static=True)
    step_size: float | None = init_field(None, static=True)

    converged: bool = False

    dimensionless: NumericalTime = init_field(lambda: NumericalTime(tag="Dimensionless Time"))
    time: NumericalTime = init_field(lambda: NumericalTime(tag="Time"))

    def __post_init__(self):
        # Guard against abstract tracers during JIT
        if self.number_of_control_points <= 1:
            return

        # Calculate the matrices (finite-difference psudospectral operators)
        if self.discretization_method:
            cps, diff, intg = self.discretization_method()
        else:
            cps, diff, intg = chebyshev_matrices(
                n=self.number_of_control_points, calculate_integration=self.calculate_integration
            )

        new_dimensionless = NumericalTime(
            tag="Dimensionless Time", control_points=cps, differentiate=diff, integrate=intg
        )

        object.__setattr__(self, "dimensionless", new_dimensionless)
