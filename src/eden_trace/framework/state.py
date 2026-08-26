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

from eden_trace.framework.state_data import StateData, Aerodynamics, NetworkData, Freestream, Mass, StabilityData, Time
from eden_trace.utils import init_field, get_target, empty_array, register

from eden_trace.framework.state_data import (
    FrameData,
)

# ----------------------------------------------------------------------------------------------------------------------
#  State
# ----------------------------------------------------------------------------------------------------------------------

@register
class State[EnergyType: NetworkData](StateData):

    tag: str = init_field("State", static=True)

    initials: eqx.Module | None = None
    time: Time = init_field(Time)

    frames: FrameData = init_field(FrameData)
    freestream: Freestream = init_field(Freestream)

    mass: Mass = init_field(Mass)
    energy: EnergyType = init_field(NetworkData)
    aerodynamics: Aerodynamics = init_field(Aerodynamics)
    stability: StabilityData = init_field(StabilityData)

    process_jacobian: jnp.ndarray = empty_array()

    def freeze_initials(self):
        frozen_initials = eqx.tree_at(lambda s: s.initials, self, None, is_leaf=lambda x: x is None)
        return replace(self, initials=frozen_initials)

