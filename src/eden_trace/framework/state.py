# Trace/Framework/State.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Jul 2024, Trace Team
# Modified: Mar 2026, J.Smart

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from functools import reduce
from dataclasses import replace

import equinox as eqx

# package imports
import jax.numpy as jnp

from eden_trace.framework.state_data import StateData, Aerodynamics, NetworkState, Freestream, Mass, StabilityData, Time
from eden_trace.utils import field, get_target, empty_array, register

from eden_trace.framework.state_data import (
    FrameData,
)

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

