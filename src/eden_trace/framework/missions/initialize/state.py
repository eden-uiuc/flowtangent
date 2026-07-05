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
    from src.eden_trace.framework.settings import Settings
    from src.eden_trace.framework.state import State
    from src.eden_trace.framework.systems import System

# ----------------------------------------------------------------------------------------------------------------------
#  Initialize/Expand State
# ----------------------------------------------------------------------------------------------------------------------


def expand_state(state: "State", system: "System", settings: "Settings"):

    updated_state = state.expand_rows(state.time.number_of_control_points)

    return updated_state, system, settings
