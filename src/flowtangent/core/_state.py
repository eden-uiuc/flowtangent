# flowtangent/Framework/State.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Jul 2024, Flowtangent Team
# Modified: Mar 2026, J.Smart

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from dataclasses import replace
from typing import Optional

import jax

from ..utils import Module, empty_array, field, update

# package imports
from ._state_data import (
    Aerodynamics,
    FrameData,
    Freestream,
    Mass,
    NetworkState,
    StabilityData,
    StateData,
    Time,
)

# ----------------------------------------------------------------------------------------------------------------------
#  State
# ----------------------------------------------------------------------------------------------------------------------


class State[EnergyType: NetworkState](StateData):

    time: Time = field(Time)

    frames: FrameData = field(FrameData)
    freestream: Freestream = field(Freestream)

    mass: Mass = field(Mass)
    energy: EnergyType = field(NetworkState)  # type: ignore
    aerodynamics: Aerodynamics = field(Aerodynamics)
    stability: StabilityData = field(StabilityData)

    process_jacobian: jax.Array = empty_array()

    initials: Optional[Module] = None

    def freeze_initials(self):
        frozen_initials = update(self, "initials", None, is_leaf=lambda x: x is None)
        return update(self, "initials", frozen_initials)

    def expand_time(self, N: int = 0):

        if N == 0:
            exp_N = self.time.N
            exp_state = self
        else:
            exp_N = N
            exp_state = update(self, "time", replace(self.time, N=N))

        return super(State, exp_state).expand_time(exp_N)
