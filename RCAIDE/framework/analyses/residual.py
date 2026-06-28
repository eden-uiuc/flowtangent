# RCAIDE/Framework/Analyses/residual.py
# (c) Copyright 2026 Aerospace Research Community LLC
#
# Created: Jun 2026, J. Smart
# Modified: Jun 2026, J. Smart

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from typing import TYPE_CHECKING, Optional, Callable

import equinox as eqx
from jaxopt import GaussNewton

# --- Framework Imports (Strictly for Type Hinting to avoid Circular Imports) ---
if TYPE_CHECKING:
    from RCAIDE.framework import State, System, Settings
    from RCAIDE.framework.conditions import ControlsConditions

from RCAIDE.utils import init_field
from RCAIDE.framework import Process, Settings, State, System
from RCAIDE.framework.conditions.Controls import Control, Residual, ControlsConditions
# ----------------------------------------------------------------------------------------------------------------------
#  Residual Minimization Analysis
# ----------------------------------------------------------------------------------------------------------------------

class ResidualAnalysis(Process):

    tag: str = init_field("Residual Analysis")

    analyze: Process = init_field(Process)
    solver: Callable = init_field(GaussNewton, as_value=True, static=True)
    solver_kwargs: Optional[dict] = None

    controls: tuple[Control, ...] = init_field(tuple)
    residuals: tuple[Residual, ...] = init_field(tuple)

    @staticmethod
    def _check_controls_balance(state: State, settings: Settings) -> bool:
        """
        Checks that the number of active controls is equal to the number of active dynamics residuals.
        """

        valid_controls = len(state.controls.active_controls) == len(state.dynamics.active_residuals)

        if settings.DEBUG_MODE:
            if valid_controls:
                print("Number of active controls matches number of active dynamics residuals.")
            else:
                print("Number of active controls does not match number of active dynamics residuals.")

            print("\nCurrent active controls:")
            for control in state.controls.active_controls:
                print(f"- {control.tag}")

            print("\nCurrent active dynamics residuals:")
            for residual in state.dynamics.active_residuals:
                print(f"- {residual.tag}")

        return valid_controls

    def _activate_controls_and_dynamics(self, state: State, settings:Settings):
        
        analysis_controls = ControlsConditions()
        analysis_dynamics = ControlsConditions()
        
        for ctrl in self.controls:
            active_ctrl = eqx.tree_at(lambda c: c._active, ctrl, True)
            analysis_controls.add_subcondition(active_ctrl)
        
        for res in self.residuals:
            active_res = eqx.tree_at(lambda c: c._active, res, True)
            analysis_dynamics.add_subcondition(active_res)

        
        analysis_state = eqx.tree_at(lambda s: (
                s.controls,
                s.dynamics
            ),
            state,
            (
                analysis_controls,
                analysis_dynamics
            )
        )

        assert self._check_controls_balance(analysis_state, settings)
        
        analysis_state = analysis_state.initialize_controls()

        return analysis_state

    def _get_residuals(self, control_values, state: "State", system: "System", settings: "Settings"):

        analysis_state = state.update_controls(control_values)
        analysis_state, _, _ = self.analyze(analysis_state, system, settings)
        residual_array = analysis_state.get_residual_array()

        return residual_array
    
    @eqx.filter_jit
    def _run_solver(
        self,
        control_values,
        state: State,
        system: System,
        settings: Settings,
    ):
        if self.solver_kwargs is None:  # Assume GaussNetwon Solver
            solver_kwargs = {
                "residual_fun": self.get_residuals,
                "tol": state.numerics.solution_tolerance,
                "maxiter": state.numerics.max_evaluations
            }
        else:
            solver_kwargs = self.solver_kwargs
        
        root = self.solver(**solver_kwargs)

        return root.run(control_values, state, system, settings)
    
    def __call__(self, state: State, system: System, settings: Settings) -> tuple[State, System, Settings]:
        
        # Set controls for current analysis
        analysis_state = self._activate_controls_and_dynamics(state, settings)

        # Get analysis control values 
        initial_control_values = analysis_state.get_control_array()
        final_control_values, opt_state = self._run_solver(initial_control_values, state, system, settings)

        analysis_state = analysis_state.update_controls(final_control_values)
        
        # Return control back to higher process
        final_state = eqx.tree_at(lambda s: s.controls, analysis_state, state.controls)

        return final_state, system, settings
