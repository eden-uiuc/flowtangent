# -------------------------------------------------------------------------------
#  Imports
# -------------------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flowtangent.framework import Settings, State, System

# package imports
import equinox as eqx
import jax.numpy as jnp

# -------------------------------------------------------------------------------
#  Stateful/Framework Version
# -------------------------------------------------------------------------------

def update_mass_and_weight(
    state: State,
    system: System,
    settings: Settings,
):
    """
    Updates the current mass of the system
    """

    mdot = state.mass.rate_of_change
    I = state.time.dimensional.integrate
    g = state.freestream.gravity

    # Integrate mdot
    integrated_mass = state.initials.mass.total + jnp.dot(I, mdot)
    integrated_weight = integrated_mass * g

    # Update State
    updated_state = eqx.tree_at(
        lambda s: (
            s.mass.total,
            s.frames.inertial.gravity_force_vector
        ),
            state,
        (
            integrated_mass,
            state.frames.inertial.gravity_force_vector.at[:, 2].set(integrated_weight[:, 0])
        ),
    )

    return updated_state, system, settings
