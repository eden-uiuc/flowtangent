# Trace/Framework/State.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Jul 2024, Trace Team
# Modified: Mar 2026, J.Smart

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from functools import reduce

import equinox as eqx

# package imports
import jax.numpy as jnp

from eden_trace.framework.conditions import AerodynamicsConditions, Condition, ControlsConditions, DynamicsConditions, EnergyNetworkConditions, FreestreamConditions, MassConditions, StabilityConditions, Time
from eden_trace.utils import init_field, get_target, empty_array, register

from eden_trace.framework.conditions import (
    Frames,
)

# ----------------------------------------------------------------------------------------------------------------------
#  State
# ----------------------------------------------------------------------------------------------------------------------

@register
class State[EnergyType: EnergyNetworkConditions](Condition):

    tag: str = init_field("State", static=True)

    initials: eqx.Module | None = None
    time: Time = init_field(Time)

    frames: Frames = init_field(Frames)
    freestream: FreestreamConditions = init_field(FreestreamConditions)

    mass: MassConditions = init_field(MassConditions)
    energy: EnergyType = init_field(EnergyNetworkConditions)
    aerodynamics: AerodynamicsConditions = init_field(AerodynamicsConditions)
    stability: StabilityConditions = init_field(StabilityConditions)

    # controls: ControlsConditions = init_field(ControlsConditions)
    # dynamics: DynamicsConditions = init_field(DynamicsConditions)

    process_jacobian: jnp.ndarray = empty_array()

    def __post_init__(self):
        frozen_initials = eqx.tree_at(lambda s: s.initials, self, None, is_leaf=lambda x: x is None)
        object.__setattr__(self, "initials", frozen_initials)
