# Trace/Framework/Analyses/residual.py
# (c) Copyright 2026 Aerospace Research Community LLC
#
# Created: Jun 2026, J. Smart
# Modified: Jun 2026, J. Smart

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from typing import TYPE_CHECKING, Optional, Any
if TYPE_CHECKING:
    from eden_trace.framework import State, System, Settings
    from eden_trace.framework.conditions import ControlsConditions

import sys
import time
import warnings
import threading

from dataclasses import replace
from collections import Counter

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import optimistix as optx

from jax.core import Tracer

jax.config.update("jax_enable_x64", True)

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

# ----------------------------------------------------------------------------------------------------------------------
#  Helper/Diagnostic Functions
# ----------------------------------------------------------------------------------------------------------------------

_last_static = None
_last_shapes = None
_trace_count = 0

def diff_args(args):
    global _last_static, _last_shapes, _trace_count
    _trace_count += 1
    
    dynamic, static = eqx.partition(args, eqx.is_array)

    dynamic_leaves = jax.tree_util.tree_leaves(dynamic)
    if len(dynamic_leaves) > 0:
        if isinstance(dynamic_leaves[0], Tracer):
            print("  [EXECUTION MODE] TRACING (JAX Compiler / AD is active)")
        else:
            print("  [EXECUTION MODE] EAGER PYTHON (JIT is disabled, actual numbers flowing)")

    shapes = jax.tree_util.tree_map(
        lambda x: (x.shape, x.dtype) if hasattr(x, "shape") else type(x), 
        dynamic
    )
    
    print(f"\n--- TRACE PASS {_trace_count} ---")
    
    if _last_static is not None:
        differences_found = False
        
        # 1. Check Dynamic Shapes
        old_dyn, _ = jax.tree_util.tree_flatten_with_path(_last_shapes)
        new_dyn, _ = jax.tree_util.tree_flatten_with_path(shapes)
        for (path, old_val), (_, new_val) in zip(old_dyn, new_dyn):
            if old_val != new_val:
                print(f"  [SHAPE CHANGED] {jax.tree_util.keystr(path)}: {old_val} -> {new_val}")
                differences_found = True
                
        # 2. Check Static Values & Structure
        old_stat, old_treedef = jax.tree_util.tree_flatten_with_path(_last_static)
        new_stat, new_treedef = jax.tree_util.tree_flatten_with_path(static)
        
        if old_treedef != new_treedef:
            print(f"  [TREEDEF CHANGED] The fundamental static PyTree structure mutated!")
            differences_found = True
        else:
            for (path, old_val), (_, new_val) in zip(old_stat, new_stat):
                # Check both value and exact type
                if type(old_val) != type(new_val) or old_val != new_val:
                    print(f"  [STATIC VALUE CHANGED] {jax.tree_util.keystr(path)}")
                    print(f"    Old: {type(old_val)} {old_val} | New: {type(new_val)} {new_val}")
                    differences_found = True
                    
        if not differences_found:
            print("  [IDENTICAL INPUTS] JAX retraced despite identical inputs!")
            
    _last_static = static
    _last_shapes = shapes

def analyze_compute_graph(func, *args):
    print("Tracing AD graph to count operations...")
    
    # Trace the Jacobian
    jaxpr_obj = jax.make_jaxpr(jax.jacfwd(func))(*args)
    
    source_counts = Counter()
    
    for eqn in jaxpr_obj.jaxpr.eqns:
        if eqn.source_info.traceback:
            user_location = "Unknown Source"
            
            # Walk backward from the innermost frame (JAX internals) up to the user code
            for frame in reversed(eqn.source_info.traceback.frames):
                # Handle varying JAX attribute naming conventions
                file_name = getattr(frame, "file_name", None) or getattr(frame, "file", "unknown_file")
                
                # Skip internal libraries to find YOUR code
                is_internal = any(lib in file_name for lib in ["jax/", "jax\\", "equinox", "jaxtyping"])
                
                if file_name != "unknown_file" and not is_internal:
                    func_name = getattr(frame, "code_name", None) or getattr(frame, "name", "unknown_func")
                    line_num = getattr(frame, "line_num", None) or getattr(frame, "lineno", "?")
                    
                    short_file = file_name.split('/')[-1].split('\\')[-1]
                    user_location = f"{func_name} ({short_file}:{line_num})"
                    break  # Found the user code, stop walking up the stack!
            
            source_counts[user_location] += 1
        else:
            source_counts["Unknown Source"] += 1
            
    print("\n" + "="*60)
    print("Top 20 Functions by Node Count")
    print("="*60)
    
    total_nodes = sum(source_counts.values())
    
    for loc, count in source_counts.most_common(20):
        percentage = (count / total_nodes) * 100
        print(f"{count:8d} nodes ({percentage:4.1f}%) | {loc}")
        
    print("="*60)
    print(f"Total Nodes Analyzed: {total_nodes}")

# ----------------------------------------------------------------------------------------------------------------------
#  Residual Analysis Class
# ----------------------------------------------------------------------------------------------------------------------

class ResidualAnalysis(Process):

    tag: str = init_field("Residual Analysis")

    analyze: Process = init_field(Process)
    solver: Any = init_field(optx.LevenbergMarquardt, as_value=True, static=True)
    solver_kwargs: Optional[dict] = init_field(None, static=True)

    solution_tolerance: Optional[float] = None
    max_evaluations: Optional[int] = None

    controls: tuple[Control, ...] = init_field(tuple)
    residuals: tuple[Residual, ...] = init_field(tuple)

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

            print(f"\n{'Active Controls':<{pad+2}}| {'Init. Values':<13}| Bounds")
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
    
    def _report_results(self, f_st, settings: Settings, f_jac=None, opt_state=None, f_res=None):

        print(f"\n{'='*70}")
        print(f"Final {self.tag} Solver State")
        print(f"{'-'*70}")
        
        if opt_state is not None:
            # Safely extract scalar values for iterations and objective
            iter_num = opt_state.num_steps.item()
            obj_val = np.mean(np.asarray(opt_state.f_info.residual)).item()
            print(f"  Solver          : {self.solver.__name__}")
            print(f"  Num. Iterations : {iter_num}")
            print(f"  Final Residual  : {obj_val:.4e}")
            
        # Determine the maximum tag length
        active_controls = f_st.controls.active_controls
        active_residuals = f_st.dynamics.active_residuals
        
        all_tags = [c.tag for c in active_controls] + [r.tag for r in active_residuals]
        # Default to 20 if empty, otherwise add 2 spaces of buffer to the longest tag
        pad = max((len(t) for t in all_tags), default=20) + 2
        
        # Run the forward pass one last time
        if f_res is None:
            final_residuals = np.asarray(f_st.get_residual_array().flatten())
        else:
            final_residuals = f_res

        print(f"\n  Final Control Values:")
        for ctrl in active_controls:
            val = get_target(f_st, ctrl.state_path)
            print(f"    {ctrl.tag:<{pad}}: {format_array(val)}")

        print(f"\n  Final Residual Values:")
        for i, res in enumerate(f_st.dynamics.active_residuals):
            print(f"    {res.tag:<{pad}}: {format_array(final_residuals[i])}")

        if f_jac is not None:
            print(f"\n Final Jacobian:")
            grad_np = np.asarray(f_jac)
            for i, ctrl in enumerate(active_controls):
                print(f"    {ctrl.tag:<{pad}}: {format_array(grad_np[i])}")
        
        print(f"{'='*70}\n")

    @eqx.filter_jit
    def _run_solver(
        self,
        control_values,
        state: State,
        system: System,
        settings: Settings,
    ):
        # Residual wrapper ---------------------------------------------------------------------------------------------

        def get_residuals(control_values, args):
            state, system, settings = args
            if settings.DEBUG_MODE:
                diff_args((control_values, state, system, settings))

            analysis_state = state.update_controls(control_values)
            updated_state, _, _ = self.analyze(analysis_state, system, settings)
            residual_array = updated_state.get_residual_array()
            return residual_array
        
        # Run solver w/ dev mode profiling -----------------------------------------------------------------------------

        if settings._DEV_MODE:
            print("Tracing Forward Pass...")
            t0 = time.time()
            # Use your objective wrapper from earlier
            forward_func = jax.jit(lambda x: get_residuals(x, (state, system, settings)))
            _ = forward_func(control_values) # Force compile
            print(f"Forward Pass Compile Time: {time.time() - t0:.2f} seconds")

            # 2. Profile the Jacobian
            print("\nTracing Jacobian...")
            t0 = time.time()
            jac_func = jax.jit(jax.jacfwd(lambda x: get_residuals(x, (state, system, settings))))
            _ = jac_func(control_values) # Force compile
            print(f"Jacobian Compile Time: {time.time() - t0:.2f} seconds")

            print(f"\n{'='*60}")
            print("Starting JAX AOT Compilation Profiler...")
            print(f"{'-'*60}")

            # 1. Profile the JAX 'Lowering' Phase
            t0 = time.time()
            # We use a lambda to cleanly pass all arguments to the solver's run method
            run_fn = lambda c, s, sys, set: optx.least_squares(
                fn=get_residuals,
                solver=self.solver(rtol=settings.numerical.relative_tolerance, atol=settings.numerical.absolute_tolerance),
                y0=c,
                args=(s, sys, set),
                max_steps=settings.numerical.max_evaluations,
            )
            lowered = jax.jit(run_fn).lower(control_values, state, system, settings)
            t_lower = time.time() - t0
            print(f"Lowering Time (JAX Tracing & Autodiff) : {t_lower:.2f} seconds")

            # 2. Measure the Graph Size
            hlo_text = lowered.as_text()
            graph_length = len(hlo_text.splitlines())
            print(f"XLA HLO Graph Size (Lines of Code)     : {graph_length:,}")

            if graph_length < settings.numerical.maximum_graph_complexity:
                # 3. Profile the XLA 'Compiling' Phase
                t0 = time.time()
                compiled = lowered.compile()
                t_compile = time.time() - t0
                print(f"Compilation Time (XLA Backend)         : {t_compile:.2f} seconds")
                print(f"{'='*60}\n")
            else:
                sys.exit(f"Graph complexity ({graph_length:,}) higher than \
                         settings.numerical.maximum_graph_complexity ({settings.numerical.maximum_graph_complexity: ,}). \
                        Terminating.")
        
        solver = self.solver(rtol=settings.numerical.relative_tolerance, atol=settings.numerical.absolute_tolerance)
        args = (state, system, settings)

        if settings.DEBUG_MODE:
            print(f"DEBUG MODE: Executing single forward pass...")
            f_res = get_residuals(control_values, (state, system, settings))
            self._report_results(state, settings, None, None, f_res)
            sys.exit("DEBUG MODE: Forward pass complete. Terminating.")
        
        else:
            results = optx.least_squares(
                fn=get_residuals,
                solver=solver,
                y0=control_values,
                args=args,
                max_steps=settings.numerical.max_evaluations,
            )

        final_control_values = results.value
        opt_state = results.state

        final_jacobian = jax.jacfwd(get_residuals)(final_control_values, (state, system, settings))
        
        return final_control_values, final_jacobian, opt_state
    
    def __call__(self, state: State, system: System, settings: Settings) -> tuple[State, System, Settings]:

        # Set controls for current analysis
        analysis_state = self._activate_controls_and_dynamics(state, settings)

        if settings.DEBUG_MODE or settings._DEV_MODE:
            scan_for_invalid_JAX_types(analysis_state,  "Analysis State")
            scan_for_invalid_JAX_types(system,  "Analysis System")

        # Get analysis control values 
        initial_control_values = analysis_state.get_control_array()
        with Spinner(enabled=not settings.DEBUG_MODE,
                     message=f"Compiling {self.tag}...") as spin_obj:
            
            final_control_values, final_jacobian, opt_state = self._run_solver(
                initial_control_values,
                analysis_state,
                system,
                settings,
            )

        analysis_state = analysis_state.update_controls(final_control_values)
        f_st, f_sys, f_set = self.analyze(analysis_state, system, settings)

        if settings.verbose:
            self._report_results(f_st, f_set, final_jacobian, opt_state)
        
        if settings._DEV_MODE:
            print(f"\n{'='*70}")
            print(f"Full {self.tag} Solver State")
            print(f"{'-'*70}")
            from pprint import pprint
            pprint(opt_state)
            print(f"\n{'='*70}")
        
        # Return control back to higher process
        final_state = eqx.tree_at(lambda s: s.controls, f_st, state.controls)

        return final_state, f_sys, settings
