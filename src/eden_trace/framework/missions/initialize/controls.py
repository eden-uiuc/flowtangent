# RCAIDE/Framework/Missions/Initialize/controls.py
# (c) Copyright 2026 Aerospace Research Community LLC
#
# Created: Jun 2026, J.
# Modified: Jun 2026, J.

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING

# --- Framework Imports (Strictly for Type Hinting to avoid Circular Imports) ---
if TYPE_CHECKING:
    from src.eden_trace.library import Component

    from src.eden_trace.framework.settings import Settings
    from src.eden_trace.framework.state import State
    from src.eden_trace.framework.systems import System

import equinox as eqx

from src.eden_trace.library.components import Wing

from src.eden_trace.framework.conditions.controls import Control, SurfaceControl

# ----------------------------------------------------------------------------------------------------------------------
# Initialize
# ----------------------------------------------------------------------------------------------------------------------


def build_controls_from_system(state: State, system: System | Component, settings: Settings):

    new_controls = state.controls
    surface_controls = []
    direct_controls = []
    unbound_controls = []

    if settings.DEBUG_MODE:
        print(f"Building controls from {system.tag}...")

    for component in system.subcomponents:
        state, component, settings = build_controls_from_system(state, component, settings)

        if component.is_control_component:
            if isinstance(component, Wing):
                new_controls = state.controls.add_control_variable(
                    SurfaceControl(
                        tag=component.tag + "_deflection",
                        surfaces=(system,),
                    )
                )
                surface_controls.append(component.tag + "_deflection")

            else:
                unbound = False
                if hasattr(component, "control_path"):
                    path = component.control_path
                else:
                    path = ()
                    unbound = True

                if hasattr(component, "control_path_indices"):
                    indices = component.control_path_indices
                else:
                    indices = (slice(None), 0)
                    unbound = True

                if unbound:
                    unbound_controls.append(component.tag)

                new_controls = state.controls.add_control_variable(
                    Control(
                        tag=component.tag,
                        state_path=path,
                        path_indices=indices,
                    )
                )
                if not unbound:
                    direct_controls.append(component.tag)

            if settings.DEBUG_MODE:
                if surface_controls:
                    print("Added the following aerodynamic surface controls:" + "\n\t- ".join(surface_controls))
                if direct_controls:
                    print("Added the following direct controls:" + "\n\t- ".join(direct_controls))
                if unbound_controls:
                    print(
                        "The following control components were found, but without path information."
                        "They may not function as intended:"
                        "\n\t- ".join(unbound_controls)
                    )

    if settings.DEBUG_MODE:
        print(
            f"Completed building controls from {system.tag}.\n"
            f"Controls may be activated for mission segments by setting "
            f"segment.active_controls = ('control_variable', ...)"
        )

    return eqx.tree_at(lambda s: s.controls, state, new_controls), system, settings
