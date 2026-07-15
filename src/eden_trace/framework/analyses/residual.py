# Trace/Framework/Analyses/residual.py
# (c) Copyright 2026 Aerospace Research Community LLC
#
# Created: Jun 2026, J. Smart
# Modified: Jun 2026, J. Smart

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from typing import TYPE_CHECKING, Optional, Callable, Literal
if TYPE_CHECKING:
    from eden_trace.framework import State, System, Settings
    from eden_trace.framework.conditions import ControlsConditions

import sys
import time
import warnings
import threading
from dataclasses import replace

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxopt import GaussNewton, LevenbergMarquardt, Broyden

jax.config.update("jax_enable_x64", True)

# --- Framework Imports (Strictly for Type Hinting to avoid Circular Imports) ---


from eden_trace.utils import init_field, get_target, scan_for_invalid_JAX_types, format_array
from eden_trace.framework import Process, Settings, State, System
from eden_trace.framework.conditions.controls import Control, Residual, ControlsConditions, DynamicsConditions
# ----------------------------------------------------------------------------------------------------------------------
#  Residual Minimization Analysis
# ----------------------------------------------------------------------------------------------------------------------

class Spinner:
    def __init__(self, message="Compiling ...", enabled=True):
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
    solver_type: Optional[Literal["min", "root"]] = init_field(None, static=True)
    solver_kwargs: Optional[dict] = init_field(None, static=True)

    solution_tolerance: Optional[float] = None
    max_evaluations: Optional[int] = None

    controls: tuple[Control, ...] = init_field(tuple)
    residuals: tuple[Residual, ...] = init_field(tuple)

    def __post_init__(self):
        if self.solver_type is None:
            if self.solver in [GaussNewton, LevenbergMarquardt]:
                object.__setattr__(self, "solver_type", "min")
            elif self.solver in [Broyden]:
                object.__setattr__(self, "solver_type", "root")
            else:
                warnings.warn(f"{self.tag} intialized with unrecognized solver '{type(self.solver).__name__}'. "
                              "Assuming solver is a residual minimizer. "
                              "If it is a direct root finder, please restart with solver_type='root'.")
                object.__setattr__(self, "solver_type", "min")
        
        if self.solver in [GaussNewton, LevenbergMarquardt]:
            assert self.solver_type == "min"
        elif self.solver in [Broyden]:
            assert self.solver_type == "root"

    def _check_controls_balance(self, state: State, settings: Settings) -> bool:
        """
        Checks that the number of active controls is equal to the number of active dynamics residuals.
        """

        valid_controls = len(state.controls.active_controls) == len(state.dynamics.active_residuals)

        if settings.verbose:
            print("\n")
            print("="*70)
            print(f" {self.tag} Controls Setup")
            print("-"*70)

            active_controls = state.controls.active_controls
            active_residuals = state.dynamics.active_residuals
            
            all_tags = [c.tag for c in active_controls] + [r.tag for r in active_residuals]
            # Default to 20 if empty, otherwise add 2 spaces of buffer to the longest tag
            pad = max((len(t) for t in all_tags), default=20) + 2

            print(f"\n{'Active Controls':<{pad+2}}| {'Init. Values':<14}| Bounds")
            print("-"*65)
            for control in state.controls.active_controls:
                print(f"- {control.tag:<{pad}}| {format_array(control.initial_value, width=12)} | {format_array(jnp.asarray(control.bounds))}")

            print("\nActive Residuals")
            print("-"*65)
            for residual in state.dynamics.active_residuals:
                if residual.get_value.__name__ != "<lambda>":
                    print(f"- {residual.tag}; func: {residual.get_value.__name__}")
                else:
                    print(f"- {residual.tag}")
            print("="*70)
            print("\n")

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

        assert self._check_controls_balance(analysis_state, settings), "Number of active controls does not match number of active dynamics residuals."
        
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
                tol = settings.numerical.solution_tolerance
            if self.max_evaluations is not None:
                maxiter=self.max_evaluations
            else:
                maxiter=settings.numerical.max_evaluations
            
            if self.solver_type == "min":
                fun_kwarg = "residual_fun"
            else:
                fun_kwarg = 'fun'
            
            solver_kwargs = {
                fun_kwarg: _get_residuals,
                "tol": tol,
                "maxiter": maxiter,
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

        # print(f"\n{'='*60}")
        # print("Starting JAX AOT Compilation Profiler...")
        # print(f"{'-'*60}")

        # # 1. Profile the JAX 'Lowering' Phase
        # t0 = time.time()
        # # We use a lambda to cleanly pass all arguments to the solver's run method
        # run_fn = lambda c, s, sys, set: solver.run(c, s, sys, set)
        # lowered = jax.jit(run_fn).lower(control_values, state, system, settings)
        # t_lower = time.time() - t0
        # print(f"Lowering Time (JAX Tracing & Autodiff) : {t_lower:.2f} seconds")

        # # 2. Measure the Graph Size
        # hlo_text = lowered.as_text()
        # print(f"XLA HLO Graph Size (Lines of Code)     : {len(hlo_text.splitlines())}")

        # # 3. Profile the XLA 'Compiling' Phase
        # t0 = time.time()
        # compiled = lowered.compile()
        # t_compile = time.time() - t0
        # print(f"Compilation Time (XLA Backend)         : {t_compile:.2f} seconds")
        # print(f"{'='*60}\n")

        # # Run the actually compiled function to get your result
        # results = compiled(control_values, state, system, settings)
        results = solver.run(control_values, state, system, settings)
        
        return results
    
    def __call__(self, state: State, system: System, settings: Settings) -> tuple[State, System, Settings]:

        # Set controls for current analysis
        analysis_state = self._activate_controls_and_dynamics(state, settings)

        if settings.DEBUG_MODE:
            scan_for_invalid_JAX_types(analysis_state,  "Analysis State")
            scan_for_invalid_JAX_types(system,  "Analysis System")
            print("\n")

        # Get analysis control values 
        initial_control_values = analysis_state.get_control_array()
        with Spinner(enabled=not settings.DEBUG_MODE, message=f"Compiling {self.tag}...") as spin_obj:
            final_control_values, opt_state = self._run_solver(
                initial_control_values,
                analysis_state,
                system,
                settings,
            )

        analysis_state = analysis_state.update_controls(final_control_values)
        f_st, f_sys, f_set = self.analyze(analysis_state, system, settings)

        if settings.verbose:
            import numpy as np
            
            print(f"\n{'='*70}")
            print(f"Final {self.tag} Solver State")
            print(f"{'-'*70}")
            
            # Safely extract scalar values for iterations and objective
            iter_num = np.asarray(opt_state.iter_num).item()
            obj_val = np.mean(np.asarray(opt_state.value)).item()
            print(f"  Solver          : {self.solver.__name__}")
            print(f"  Num. Iterations : {iter_num}")
            print(f"  Final Objective : {obj_val:.6e}")
                
            # Determine the maximum tag length
            active_controls = f_st.controls.active_controls
            active_residuals = f_st.dynamics.active_residuals
            
            all_tags = [c.tag for c in active_controls] + [r.tag for r in active_residuals]
            # Default to 20 if empty, otherwise add 2 spaces of buffer to the longest tag
            pad = max((len(t) for t in all_tags), default=20) + 2
            
            # Run the forward pass one last time
            final_residuals = np.asarray(f_st.get_residual_array().flatten())

            print(f"\n  Final Control Values:")
            for ctrl in active_controls:
                val = get_target(f_st, ctrl.state_path)
                print(f"    {ctrl.tag:<{pad}}: {format_array(val)}")

            print(f"\n  Final Residual Values:")
            for i, res in enumerate(f_st.dynamics.active_residuals):
                print(f"    {res.tag:<{pad}}: {format_array(final_residuals[i])}")

            if self.solver is GaussNewton or self.solver is LevenbergMarquardt:

                print(f"\n Final Jacobian:")
                grad_np = np.asarray(opt_state.gradient)
                for i, ctrl in enumerate(active_controls):
                    print(f"    {ctrl.tag:<{pad}}: {format_array(grad_np[i])}")
            
            print(f"{'='*70}\n")
        
        if settings.DEBUG_MODE:
            print(f"\n{'='*70}")
            print(f"DEBUG: Full {self.tag} Solver State")
            print(f"{'-'*70}")
            from pprint import pprint
            pprint(opt_state._asdict())
            print(f"\n{'='*70}")
        
        # Return control back to higher process
        final_state = eqx.tree_at(lambda s: s.controls, f_st, state.controls)

        return final_state, f_sys, settings
