# Trace/Framework/Analyses/residual.py
# (c) Copyright 2026 Aerospace Research Community LLC
#
# Created: Jun 2026, J. Smart
# Modified: Jun 2026, J. Smart

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from typing import TYPE_CHECKING, Optional, Any, Callable, Literal, overload

if TYPE_CHECKING:
    from eden_trace.framework import State, System, Settings

import sys
import time
import threading

from collections import Counter

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import optimistix as optx

from jax.core import Tracer
from scipy.optimize import root

jax.config.update("jax_enable_x64", True)

from eden_trace.utils import init_field, get_target, scan_for_invalid_JAX_types, format_array
from eden_trace.framework import Process, ProcessStep, Settings, State, System
from eden_trace.framework.conditions.controls import Control, Residual
# ----------------------------------------------------------------------------------------------------------------------
#  Helper/Diagnostic Functions
# ----------------------------------------------------------------------------------------------------------------------

class Spinner:
    def __init__(self, message="Tracing ...", enabled=True):
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

_last_static = None
_last_shapes = None
_trace_count = [0]
_analysis_stack = []

def diff_args(args):
    global _last_static, _last_shapes, _trace_count, _analysis_stack
    
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
            print(f"  [TREEDEF CHANGED] Input PyTree structure mutated.")
            differences_found = True
        else:
            for (path, old_val), (_, new_val) in zip(old_stat, new_stat):
                # Check both value and exact type
                if type(old_val) != type(new_val) or old_val != new_val:
                    print(f"  [STATIC VALUE CHANGED] {jax.tree_util.keystr(path)}")
                    print(f"    Old: {type(old_val)} {old_val} | New: {type(new_val)} {new_val}")
                    differences_found = True
                    
        if not differences_found:
            print("  [IDENTICAL INPUTS] PyTree structure is identical.")
            
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
    solver: Any | str = init_field(optx.LevenbergMarquardt, as_value=True, static=True)
    solver_options: Optional[dict] = init_field(None, static=True)

    controls: tuple[Control, ...] = init_field(tuple)
    residuals: tuple[Residual, ...] = init_field(tuple)

    def __init__(
            self,
            analyze: Process = Process(tag="Residual Analysis Forward Pass"),
            solver: Any | str = optx.LevenbergMarquardt,
            solver_options: Optional[dict] = None,
            controls: tuple[Control, ...] = (),
            residuals: tuple[Residual, ...] = (),
            **kwds
        ) -> None:
        super().__init__(**kwds)
        self.analyze = analyze
        self.solver = solver
        self.solver_options = solver_options
        self.controls = controls
        self.residuals = residuals

    def _check_controls_balance(self, settings: Settings) -> bool:
        """
        Checks that the number of active controls is equal to the number of active dynamics residuals.
        """

        valid_controls = len(self.controls) == len(self.residuals)

        if settings.verbose:
            print("\n")
            print("="*70)
            print(f" {self.tag} Controls Setup")
            print("-"*70)

            active_controls = self.controls
            active_residuals = self.residuals
            
            all_tags = [c.tag for c in active_controls] + [r.tag for r in active_residuals]
            # Default to 20 if empty, otherwise add 2 spaces of buffer to the longest tag
            pad = max((len(t) for t in all_tags), default=20) + 2

            print(f"\n{'Active Controls':<{pad+2}}| {'Init. Values':<13}| Bounds")
            print("-"*65)
            for control in active_controls:
                print(f"- {control.tag:<{pad}}| {format_array(control.initial_value, width=12):>12} | {format_array(jnp.asarray(control.bounds))}")

            print("\nActive Residuals")
            print("-"*65)
            for residual in active_residuals:
                if residual.get_value.__name__ != "<lambda>":
                    print(f"- {residual.tag}; func: {residual.get_value.__name__}")
                else:
                    print(f"- {residual.tag}")
            print("="*70)
            print("\n")

        return valid_controls

    def _report_results(self, f_ctrls: jnp.ndarray | np.ndarray, f_st: State, opt_state=None):

        print(f"\n{'='*70}")
        print(f"Final {self.tag} Solver State")
        print(f"{'-'*70}")
        
        if opt_state is not None:
            if isinstance(self.solver, str):
                solver_name = f"Scipy Root; Method: {self.solver}"
                iter_num = opt_state.nit
                obj_val = np.mean(opt_state.fun)
                f_res = opt_state.fun
                f_jac = opt_state.jac
            else:
                solver_name = f"Optimistix Least Squares; Method: {self.solver.__name__}"
                iter_num = opt_state.num_steps.item()

                f_res = opt_state.f_info.residual
                f_jac = None
                obj_val = np.mean(np.asarray(f_res)).item()
            print(f"  Solver          : {solver_name}")
            print(f"  Num. Iterations : {iter_num}")
            print(f"  Final Residual  : {obj_val:.4e}")
        else:
            f_res = None
            f_jac = None
            
        # Determine the maximum tag length
        active_controls = self.controls
        active_residuals = self.residuals
        
        all_tags = [c.tag for c in active_controls] + [r.tag for r in active_residuals]
        # Default to 20 if empty, otherwise add 2 spaces of buffer to the longest tag
        pad = max((len(t) for t in all_tags), default=20) + 2
        
        # Run the forward pass one last time
        print(f"\n  Final Control Values:")
        for idx, ctrl in enumerate(active_controls):
            print(f"    {ctrl.tag:<{pad}}: {format_array(f_ctrls[idx])}")

        if f_res is not None:
            print(f"\n  Final Residual Values:")
            for i, res in enumerate(f_st.dynamics.active_residuals):
                print(f"    {res.tag:<{pad}}: {format_array(f_res[i])}")

        if f_jac is not None:
            print(f"\n Final Jacobian:")
            grad_np = np.asarray(f_jac)
            for i, ctrl in enumerate(active_controls):
                print(f"    {ctrl.tag:<{pad}}: {format_array(grad_np[i])}")
        
        print(f"{'='*70}\n")

    def _update_controls(self, state: State, control_values:jnp.ndarray, settings:Settings) -> State:
    
            control_state = state
            if settings.numerical.sum_residuals:
                N = 1
            else:
                N = int(state.time.N)
            ctrl_idx = 0
    
            for ctrl in self.controls:
                solver_logit = control_values[ctrl_idx : ctrl_idx + N]
                new_val = ctrl.scale(solver_logit[:N])
                control_state = eqx.tree_at(
                    lambda s: get_target(s, ctrl.state_path),
                    control_state,
                    jnp.atleast_2d(new_val).reshape((-1, 1)),
                )
                ctrl_idx += N
    
            return control_state

    def _initialize_controls(self, state: State, system: System, settings: Settings) -> tuple[State, System, Settings]:
        control_values = []
        
        for ctrl in self.controls:
            n_cp = int(state.time.N)

            # All control values are normalized by their initial value, so set initial control value to 1.0
            # Values are rescaled in update_controls when actually added to state
            if ctrl.initial_value is not None:
                    control_values.append(jnp.full(
                        (n_cp, 1),
                        ctrl.normalize(ctrl.initial_value))
                    )
            else:
                raise ValueError(f"Control {ctrl.tag} has no initial value: {ctrl.initial_value}."
                                "Initial value must be a float or an array of size matching the number of analysis control points.")

        ctrl_state = self._update_controls(state, jnp.concatenate(control_values, axis=0), settings)

        return ctrl_state, system, settings

    def _get_control_array(self, state: State, settings:Settings) -> jnp.ndarray:
        ctrl_vals = []
        for ctrl in self.controls:
            current_val = get_target(state, ctrl.state_path)
            logit_val = ctrl.normalize(current_val)
            if settings.numerical.sum_residuals:
                logit_val = jnp.atleast_2d(logit_val[0]) # Only take a batch instance as the control value
            ctrl_vals.append(logit_val)

        return jnp.concatenate(ctrl_vals, axis=0).flatten()

    def _get_residual_array(self, state: State, settings:Settings) -> jnp.ndarray:

        residual_values = [r.get_value(state) for r in self.residuals]
        if settings.numerical.sum_residuals:
            residual_values = [jnp.sum(r, axis=0) for r in residual_values]
        return jnp.concatenate(residual_values, axis=0).flatten()

    def _run_scipy_solver(
            self,
            get_residuals: Callable,
            control_values: jnp.ndarray,
            state: State,
            system: System,
            settings: Settings,
            solver_options: dict,
    ):

        args = (state, system, settings)
        jac_fn = jax.jacfwd(get_residuals, argnums=0, has_aux=True)

        def scipy_residual(x_np):
            x_jax = jnp.array(x_np)
            res_jax, _aux = get_residuals(x_jax, args)
            return np.array(res_jax)

        def scipy_jac(x_np):
            x_jax = jnp.array(x_np)
            jac_jax, _aux = jac_fn(x_jax, args)
            return np.array(jac_jax)

        results = root(
            fun=scipy_residual,
            x0=np.array(control_values),
            jac=scipy_jac,
            method=self.solver,
            options=solver_options
        )

        if settings.DEBUG_MODE:
            print(f"\n--- {self.tag.upper()} CLOSEOUT PASS ---")
        _, (f_st, f_sys, f_set) = get_residuals(jnp.array(results.x), args)

        return results.x, results, f_st, f_sys, f_set

    @eqx.filter_jit
    def _run_optx_solver(
            self,
            get_residuals: Callable,
            control_values: jnp.ndarray,
            state: State,
            system: System,
            settings: Settings,
            solver_options: dict,
    ):
        assert(isinstance(self.solver, Callable))
        results = optx.least_squares(
            fn=get_residuals,
            solver=self.solver(**solver_options),
            y0=control_values,
            args=(state, system, settings),
            max_steps=settings.numerical.max_evaluations,
            has_aux=True
        )

        final_state, final_system, final_settings = results.aux

        return results.value, results.state, final_state, final_system, final_settings

    def _run_solver(
            self,
            control_values,
            state: State,
            system: System,
            settings: Settings,
    ):
        # Residual wrapper definied in _run_solver scope to avoid tracing self argument if it were a bound method ------
        @eqx.filter_jit
        def get_residuals(control_values, args):

            state, system, settings = args

            if settings.DEBUG_MODE:
                global _analysis_stack, _trace_count
                if len(_analysis_stack) > len(_trace_count):
                        _trace_count.append(0)
                if _trace_count[_analysis_stack.index(self.tag)] > 1:
                    diff_args((control_values, state, system, settings))
                _trace_count[_analysis_stack.index(self.tag)] += 1
                print(f"\n--- {self.tag.upper()} PASS {_trace_count[_analysis_stack.index(self.tag)]} ---")

            control_state = self._update_controls(state, control_values, settings)
            analysis_state, analysis_system, analysis_settings = self.analyze(control_state, system, settings)
            
            return self._get_residual_array(analysis_state, analysis_settings), (analysis_state, analysis_system, analysis_settings)
        
        # Run solver w/ dev mode profiling -----------------------------------------------------------------------------
        if self.solver_options is None:
            if isinstance(self.solver, str):
                if self.solver == 'hybr':
                    max_str = "maxfev"
                else:
                    max_str = "maxiter"
                solver_options = {
                    max_str: settings.numerical.max_evaluations,
                    "xtol": settings.numerical.relative_tolerance,
                }
            else:
                solver_options = {
                    "rtol": settings.numerical.relative_tolerance,
                    "atol": settings.numerical.absolute_tolerance
                }
        else:
            solver_options = self.solver_options

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

            # Profile the JAX lowering phase if using Optimistix
            
            # We use a lambda to cleanly pass all arguments to the solver's run method
            if isinstance(self.solver, str):
                pass
            else:
                t0 = time.time()
                run_fn = lambda c, s, sys, set: optx.least_squares(
                    fn=get_residuals,
                    solver=self.solver(**solver_options),
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
        
        if settings.DEBUG_MODE:    

            print(f"DEBUG MODE: Executing single forward pass...")
            f_ctrls, (f_st, f_sys, f_set) = get_residuals(control_values, (state, system, settings))
            opt_state = None
        
        else:
            if isinstance(self.solver, str):
                run_fn = self._run_scipy_solver
            else:
                run_fn = self._run_optx_solver

            f_ctrls, opt_state, f_st, f_sys, f_set = run_fn(
                get_residuals,
                control_values,
                state,
                system,
                settings,
                solver_options
            )
        
        return f_ctrls, opt_state, f_st, f_sys, f_set
    
    def __call__(self, state: State, system: System, settings: Settings):

        global _analysis_stack, _last_static, _last_shapes, _trace_count
        _analysis_stack.append(self.tag)
        _last_static = None
        _last_shapes = None

        if settings.DEBUG_MODE or settings._DEV_MODE:
            scan_for_invalid_JAX_types(state,  "Analysis State")
            scan_for_invalid_JAX_types(system,  "Analysis System")

        # Get analysis control values 
        initial_control_values = self._get_control_array(state, settings)

        # Run Solver
        with Spinner(enabled=not settings.DEBUG_MODE and len(_analysis_stack) == 1,
                     message=f"Tracing {self.tag}..."):
            
            f_ctrls, opt_state, f_st, f_sys, f_set = self._run_solver(
                initial_control_values,
                state,
                system,
                settings,
            )

        # Post-Processing
        if settings.verbose and len(_analysis_stack) == 1:
            self._report_results(f_ctrls, f_st, opt_state)
        
        if settings._DEV_MODE:
            print(f"\n{'='*70}")
            print(f"Full {self.tag} Solver State")
            print(f"{'-'*70}")
            from pprint import pprint
            pprint(opt_state)
            print(f"\n{'='*70}")

        del _analysis_stack[-1]
        del _trace_count[-1]

        return f_st, f_sys, f_set

    @overload
    def run(
        self, state: State, system: System, settings: Settings, *,
        initialize: bool = ..., track_history: Literal[True]
    ) -> tuple[State, System, Settings, Process]: ...

    @overload
    def run(
        self, state: State, system: System, settings: Settings, *,
        initialize: bool = ..., track_history: Literal[False] = ...
    ) -> tuple[State, System, Settings]: ...

    def run(self, state: State, system: System, settings:Settings, *, initialize=False, track_history: bool = False):

        if initialize:
            state, system, settings = self.initialize(state, system, settings)
            state, system, settings = self._initialize_controls(state, system, settings)
        if not track_history:
            return self(state, system, settings)

        if settings.verbose:
            print(f"Residual analysis '{self.tag}' called with track_history enabled. "
                  "History returned will be single forward pass with final input values.")
            
        r_st, r_sys, r_setts = self(state, system, settings)
        f_st, f_sys, f_setts, history = self.analyze.run(r_st, r_sys, r_setts, track_history=True)
        return f_st, f_sys, f_setts, history
        

        