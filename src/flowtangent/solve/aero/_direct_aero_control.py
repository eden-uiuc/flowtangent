# flowtangent/Library/Methods/Aerodynamics/Test_Aero/test_aero.py
# (c) Copyright 2026 Aerospace Research Community LLC#
# Created:  Jan 2026, J. Smart
# Modified:
# -------------------------------------------------------------------------------
#  Imports
# -------------------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flowtangent.core._settings import Settings
    from flowtangent.core._state import State
    from flowtangent.core._systems import System


# package imports

# FlowTangent Imports

from ...utils import update

# -------------------------------------------------------------------------------
#  Direct CL/CD Control Analysis
# -------------------------------------------------------------------------------


def direct_aero(
    state: State,
    system: System,
    settings: Settings,
):

    C_L = state.aerodynamics.coefficients.lift.total
    C_D = state.aerodynamics.coefficients.drag.total

    rho = 0.4
    flight_speed = state.freestream.speed
    S = system.areas.reference
    qS = 0.5 * rho * flight_speed**2 * S

    F_Z = qS * C_L
    F_X = qS * (C_D - 0.06)

    state = update(
        state,
        ("frames.wind.total_force_vector", state.frames.wind.total_force_vector.at[:, 2].set(F_Z.flatten())),
    )
    state = update(
        state,
        ("frames.wind.total_force_vector", state.frames.wind.total_force_vector.at[:, 0].set(F_X.flatten())),
    )

    return state, system, settings
