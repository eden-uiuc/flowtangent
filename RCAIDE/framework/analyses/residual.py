# RCAIDE/Framework/Analyses/residual.py
# (c) Copyright 2026 Aerospace Research Community LLC
#
# Created: Jun 2026, J. Smart
# Modified: Jun 2026, J. Smart

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from typing import TYPE_CHECKING, Optional, Callable
if TYPE_CHECKING:
    from RCAIDE.framework import State, System, Settings
    from RCAIDE.framework.conditions import ControlsConditions

import sys
import time
import threading
from dataclasses import replace

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxopt import GaussNewton

jax.config.update("jax_enable_x64", True)

# --- Framework Imports (Strictly for Type Hinting to avoid Circular Imports) ---


from RCAIDE.utils import init_field, get_target, scan_for_invalid_JAX_types
from RCAIDE.framework import Process, Settings, State, System
from RCAIDE.framework.conditions.controls import Control, Residual, ControlsConditions, DynamicsConditions
# ----------------------------------------------------------------------------------------------------------------------
#  Residual Minimization Analysis
# ----------------------------------------------------------------------------------------------------------------------

class Spinner:
    def __init__(self, message="JIT compiling and solving...", enabled=True):
        self.spinner_chars = "|/-\\"
        self.message = message
        self.enabled = enabled
        self.running = False
        self.thread = None

    def spin(self):
        i = 0
        while self.running:
            sys.stdout.write(f"\r{self.message} {self.spinner_chars[i % 4]}")
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1

    def update_status(self, message):
        """Allows external updates to the message while running."""
        self.message = message
        # \033[K clears to the end of the line so shorter strings don't leave artifacts
        sys.stdout.write(f"\r{self.message} \033[K")
        sys.stdout.flush()
    

    def __enter__(self):
        if self.enabled:
            self.running = True
            self.thread = threading.Thread(target=self.spin)
            self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.enabled:
            self.running = False
            if self.thread is not None:
                self.thread.join()
            sys.stdout.write(f"\nAnalysis complete.\n")
            sys.stdout.flush()

class ResidualAnalysis(Process):

    tag: str = init_field("Residual Analysis")

    analyze: Process = init_field(Process)
    solver: Callable = init_field(GaussNewton, as_value=True, static=True)
    solver_kwargs: Optional[dict] = None
    solution_tolerance: Optional[float] = None
    max_evaluations: Optional[int] = None

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
        
        analysis_controls = ControlsConditions(tag="Analysis Controls")
        analysis_dynamics = DynamicsConditions(tag="Analysis Dynamics")
        
        for ctrl in self.controls:
            active_ctrl = replace(ctrl, _active=True)
            analysis_controls = analysis_controls.add_subcondition(active_ctrl)
        
        for res in self.residuals:
            active_res = replace(res, _active=True)
            analysis_dynamics = analysis_dynamics.add_subcondition(active_res)

        
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
    
    @eqx.filter_jit
    def _run_solver(
        self,
        control_values,
        state: State,
        system: System,
        settings: Settings,
    ):
        # Residual wrapper ---------------------------------------------------------------------------------------------

        def _get_residuals(control_values, state: State, system: System, settings: Settings):
            
            analysis_state = state.update_controls(control_values)
            updated_state, _, _ = self.analyze(analysis_state, system, settings)
            residual_array = updated_state.get_residual_array()

            return residual_array
        
        # Set up solver ------------------------------------------------------------------------------------------------

        if self.solver_kwargs is None:
            if self.solution_tolerance is not None:
                tol = self.solution_tolerance
            else:
                tol = state.numerics.solution_tolerance
            if self.max_evaluations is not None:
                maxiter=self.max_evaluations
            else:
                maxiter=state.numerics.max_evaluations
            
            if self.solver is GaussNewton:
                fun_kwarg = "residual_fun"
            else:
                fun_kwarg = 'fun'
            
            solver_kwargs = {
                fun_kwarg: _get_residuals,
                "tol": tol,
                "maxiter": maxiter,
                # "unroll": False,
                "implicit_diff": False,
            }
        else:
            solver_kwargs = self.solver_kwargs

        # Run solver ---------------------------------------------------------------------------------------------------

        solver = self.solver(**solver_kwargs)

        # print("Tracing Forward Pass...")
        # t0 = time.time()
        # # Use your objective wrapper from earlier
        # forward_func = jax.jit(lambda x: _get_residuals(x, state, system, settings))
        # _ = forward_func(control_values) # Force compile
        # print(f"Forward Pass Compile Time: {time.time() - t0:.2f} seconds")

        # # 2. Profile the Jacobian
        # print("Tracing Jacobian...")
        # t0 = time.time()
        # jac_func = jax.jit(jax.jacfwd(lambda x: _get_residuals(x, state, system, settings)))
        # _ = jac_func(control_values) # Force compile
        # print(f"Jacobian Compile Time: {time.time() - t0:.2f} seconds")

        print(f"\n{'='*60}")
        print("Starting JAX AOT Compilation Profiler...")
        print(f"{'-'*60}")

        # 1. Profile the JAX 'Lowering' Phase
        t0 = time.time()
        # We use a lambda to cleanly pass all arguments to the solver's run method
        run_fn = lambda c, s, sys, set: solver.run(c, s, sys, set)
        lowered = jax.jit(run_fn).lower(control_values, state, system, settings)
        t_lower = time.time() - t0
        print(f"Lowering Time (JAX Tracing & Autodiff) : {t_lower:.2f} seconds")

        # 2. Measure the Graph Size
        hlo_text = lowered.as_text()
        print(f"XLA HLO Graph Size (Lines of Code)     : {len(hlo_text.splitlines())}")

        # 3. Profile the XLA 'Compiling' Phase
        t0 = time.time()
        compiled = lowered.compile()
        t_compile = time.time() - t0
        print(f"Compilation Time (XLA Backend)         : {t_compile:.2f} seconds")
        print(f"{'='*60}\n")

        # Run the actually compiled function to get your result
        results = compiled(control_values, state, system, settings)
        # results = solver.run(control_values, state, system, settings)
        
        return results
    
    def __call__(self, state: State, system: System, settings: Settings) -> tuple[State, System, Settings]:
        
        # Set controls for current analysis
        analysis_state = self._activate_controls_and_dynamics(state, settings)

        if settings.DEBUG_MODE:
            scan_for_invalid_JAX_types(analysis_state,  "Analysis State")
            scan_for_invalid_JAX_types(system,  "Analysis System")

        # Get analysis control values 
        initial_control_values = analysis_state.get_control_array()
        with Spinner(enabled=not settings.DEBUG_MODE, message=f"Solving {self.tag}...") as spin_obj:
            final_control_values, opt_state = self._run_solver(
                initial_control_values,
                analysis_state,
                system,
                settings,
            )

        if settings.verbose:
            import numpy as np
            
            print(f"\n{'='*60}")
            print(f"Final {self.tag} Solver State")
            print(f"{'-'*60}")
            
            # Safely extract scalar values for iterations and objective
            iter_num = np.asarray(opt_state.iter_num).item()
            obj_val = np.asarray(opt_state.value).item()
            print(f"  Num. Iterations : {iter_num}")
            print(f"  Final Objective : {obj_val:.6e}")
            
            # Determine the maximum tag length
            active_controls = analysis_state.controls.active_controls
            active_residuals = analysis_state.dynamics.active_residuals
            
            all_tags = [c.tag for c in active_controls] + [r.tag for r in active_residuals]
            # Default to 20 if empty, otherwise add 2 spaces of buffer to the longest tag
            pad = max((len(t) for t in all_tags), default=20) + 2

            # Strip JAX wrappers and format cleanly
            def format_array(v):
                v_np = np.asarray(v)
                if v_np.size == 1:
                    return f"{v_np.item():>12.6e}"
                # For 1D/2D arrays, use numpy's built-in pretty printer
                return np.array2string(v_np, precision=6, separator=', ')

            print(f"\n  Final Control Values:")
            for ctrl in active_controls:
                val = get_target(analysis_state, ctrl.state_path)
                print(f"    {ctrl.tag:<{pad}}: {format_array(val)}")
                
            print(f"\n  Final Residual Values:")
            final_residuals = np.asarray(opt_state.residual).flatten() 
            for i, res in enumerate(analysis_state.dynamics.active_residuals):
                print(f"    {res.tag:<{pad}}: {final_residuals[i]:>12.6e}")
                
            print(f"\n  Final Gradient / Jacobian:")
            grad_np = np.asarray(opt_state.gradient)
            grad_str = np.array2string(grad_np, precision=4, separator=', ', prefix="    ")
            print(f"    {grad_str}")
            
            print(f"{'='*60}\n")

        analysis_state = analysis_state.update_controls(final_control_values)
        
        # Return control back to higher process
        final_state = eqx.tree_at(lambda s: s.controls, analysis_state, state.controls)

        return final_state, system, settings
