# RCAIDE/Framework/Missions/Initialize/state.py
# (c) Copyright 2026 Aerospace Research Community LLC
#
# Created: Mar 2026, J. Smart
# Modified: Mar 2026, J. Smart

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from typing import TYPE_CHECKING

# --- Framework Imports (Strictly for Type Hinting to avoid Circular Imports) ---
if TYPE_CHECKING:
    from RCAIDE.Framework.settings import Settings
    from RCAIDE.Framework.state import State
    from RCAIDE.Framework.systems import System

# ----------------------------------------------------------------------------------------------------------------------
#  Initialize/Expand State
# ----------------------------------------------------------------------------------------------------------------------


def expand_state(state: "State", system: "System", settings: "Settings"):

    updated_state = state.expand_rows(state.numerics.number_of_control_points)

    return updated_state, system, settings
