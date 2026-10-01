# flowtangent/Framework/Missions/Initialization/mass.py
# (c) Copyright 2024 Aerospace Research Community LLC
# Created: Aug 2024, FlowTangent Team

# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------

# FlowTangent Imports
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ... import Settings, State, System
from ...utils import update

# ----------------------------------------------------------------------------------------------------------------------
# Initialize Mass
# ----------------------------------------------------------------------------------------------------------------------


def initialize_mass(
    state: State,
    system: System,
    settings: Settings,
):

    m_initial = state.initials.mass.total[-1, 0]

    # This needs to be set automatically if the initial value is 0
    if m_initial == 0.0:
        m_initial = system.mass_properties.total

    m_current = state.mass.total[0, 0]

    state = update(state, "mass.total", state.mass.total + (m_initial - m_current))

    return state, system, settings
