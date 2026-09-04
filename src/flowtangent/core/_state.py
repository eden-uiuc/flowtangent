# flowtangent/Framework/State.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Jul 2024, Flowtangent Team
# Modified: Mar 2026, J.Smart

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from dataclasses import replace

import equinox as eqx

# package imports
import jax.numpy as jnp

from flowtangent.core._state_data import (
    Aerodynamics,
    FrameData,
    Freestream,
    Mass,
    NetworkState,
    StabilityData,
    StateData,
    Time,
)
from flowtangent.utils import empty_array, field, register

# ----------------------------------------------------------------------------------------------------------------------
#  State
# ----------------------------------------------------------------------------------------------------------------------

@register
class State[EnergyType: NetworkState](StateData):

    tag: str = field("State", static=True)

    initials: eqx.Module | None = None
    time: Time = field(Time)

    frames: FrameData = field(FrameData)
    freestream: Freestream = field(Freestream)

    mass: Mass = field(Mass)
    energy: EnergyType = field(NetworkState)
    aerodynamics: Aerodynamics = field(Aerodynamics)
    stability: StabilityData = field(StabilityData)

    process_jacobian: jnp.ndarray = empty_array()

    def freeze_initials(self):
        frozen_initials = eqx.tree_at(lambda s: s.initials, self, None, is_leaf=lambda x: x is None)
        return replace(self, initials=frozen_initials)

