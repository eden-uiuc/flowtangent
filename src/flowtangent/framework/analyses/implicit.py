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

import contextlib
import io
import logging
import os
import sys
import time
import threading
import warnings

from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import optimistix as optx

from jax.core import Tracer
from scipy.optimize import root

jax.config.update("jax_enable_x64", True)

from ...utils import field, get_target, scan_for_invalid_JAX_types, format_array, io_partition, inspect_leaves
from .. import Settings, State, System
from ..state_data.controls import Control, Residual
from ..processes import Process, array_barrier
# ----------------------------------------------------------------------------------------------------------------------
#  Helper/Diagnostic Functions
# ----------------------------------------------------------------------------------------------------------------------

class TraceReadout:
    def __init__(self, message="Tracing ...", enabled=True):
        self.spinner_chars = "|/-\\"
        self.message = message
        self.enabled = enabled
        self.running = False
        self.thread = None

        self.start_time = None
        self.null_fd = None
        self.saved_stderr_fd = None

    def spin(self):
        i = 0
        while self.running:
            if self.start_time:
                elapsed = int(time.time() - self.start_time)
                mins, secs = divmod(elapsed, 60)
                sys.stdout.write(f"\r{self.message} [{mins:02d}:{secs:02d}] \033[K")
            else:
                sys.stdout.write(f"\r{self.message} {self.spinner_chars[i % 4]} \033[K")

            sys.stdout.flush()
            time.sleep(0.1)
            i += 1

    def update_status(self, message):
        self.message = message
        if self.start_time:
            elapsed = int(time.time() - self.start_time)
            mins, secs = divmod(elapsed, 60)
            sys.stdout.write(f"\r{self.message} [{mins:02d}:{secs:02d}] \033[K")
        else:
            sys.stdout.write(f"\r{self.message} \033[K")
        sys.stdout.flush()
    

    def __enter__(self):
        if self.enabled:
            # 1. Hijack the OS-level stderr (File Descriptor 2) to silence C++ XLA
            self.null_fd = os.open(os.devnull, os.O_WRONLY)
            self.saved_stderr_fd = os.dup(2) # Save the real stderr
            os.dup2(self.null_fd, 2)         # Point stderr to black hole
            
            self.start_time = time.time()
            self.running = True
            self.thread = threading.Thread(target=self.spin)
            self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.enabled:
            self.running = False
            if self.thread is not None:
                self.thread.join()
                
            # 2. Restore the OS-level stderr immediately so Python errors can print
            if self.saved_stderr_fd is not None:
                os.dup2(self.saved_stderr_fd, 2)
                os.close(self.saved_stderr_fd)
                os.close(self.null_fd)
                
            elapsed = int(time.time() - self.start_time)
            mins, secs = divmod(elapsed, 60)
            sys.stdout.write(f"\r{self.message} [{mins:02d}:{secs:02d}] Complete. \033[K\n")
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

class ImplicitAnalysis(Process):

    tag: str = field("Implicit Analysis")

    analyze: Process = field(Process)
    solver: Any | str = field(optx.LevenbergMarquardt, as_value=True, static=True)
    solver_options: Optional[dict] = field(None, static=True)

    controls: tuple[Control, ...] = field(tuple)
    residuals: tuple[Residual, ...] = field(tuple)

    def __init__(
            self,
            analyze: Process = Process(tag="Implicit Analysis Forward Pass"),
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

    def _report_results(self, f_ctrls: jax.Array, f_res: jax.Array, opt_stats=None):
    
        print(f"\n{'='*70}")
        print(f"Final {self.tag} Solver State")
        print(f"{'-'*70}")
        
        if opt_stats:
            try:
                if isinstance(self.solver, str):
                    solver_name = f"Scipy Root; Method: {self.solver}"
                    iter_num = opt_stats.nit
                    avg_res = np.mean(np.asarray(f_res)).item()
                else:
                    solver_name = f"Optimistix Least Squares; Method: {self.solver.__name__}"
                    iter_num = opt_stats['num_steps'].item()
                    avg_res = np.mean(np.asarray(f_res)).item()

                print(f"  Solver          : {solver_name}")
                print(f"  Num. Iterations : {iter_num}")
                print(f"  Avg. Residual   : {avg_res:.4e}")
            except Exception as e:
                print(f"  ERROR: Optimizer state parsing error: {e} Printing raw results...")
                print(opt_stats)
            
        # Determine the maximum tag length
        active_controls = self.controls
        active_residuals = self.residuals
        
        all_tags = [c.tag for c in active_controls] + [r.tag for r in active_residuals]
        # Default to 20 if empty, otherwise add 2 spaces of buffer to the longest tag
        pad = max((len(t) for t in all_tags), default=20) + 2
        
        # Run the forward pass one last time
        print(f"\n  Final Control Values:")
        for idx, ctrl in enumerate(active_controls):
            print(f"    {ctrl.tag:<{pad}}: {format_array(ctrl.scale(f_ctrls[idx]))}")

        print(f"\n  Final Residual Values:")
        for i, res in enumerate(self.residuals):
            print(f"    {res.tag:<{pad}}: {format_array(f_res[i])}")
        
        print(f"{'='*70}\n")

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

    def _update_controls(self, state: State, control_values:jnp.ndarray, settings:Settings) -> State:
    
            control_state = state
            if settings.numerical.sum_residuals:
                N = 1
            else:
                N = state.time.N
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

    def initialize_controls(self, state: State, system: System, settings: Settings) -> tuple[State, System, Settings]:
        control_values = []
        
        for ctrl in self.controls:
            n_cp = state.time.N

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
        _, (f_st, f_sys) = get_residuals(jnp.array(results.x), args)

        return results.x, results, f_st, f_sys

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
            args=(state, system),
            max_steps=settings.numerical.max_evaluations,
            has_aux=True
        )

        final_state, final_system = results.aux

        return results.value, results.stats, final_state, final_system

    def _run_solver(
            self,
            control_values,
            state: State,
            system: System,
            settings: Settings,
    ):

        # Partition inputs to avoid tracing the entire state and system trees
        active_paths = [p.split(':')[0].strip() for p in self.analyze.full_io]
        active_ids = set()

        ctx = {'state': state, 'system': system}
        for io_str in active_paths:
            try:
                target_obj = eval(io_str, {}, ctx)
                leaves = jax.tree_util.tree_leaves(target_obj)
                for leaf in leaves:
                    if eqx.is_array_like(leaf):
                        active_ids.add(id(leaf))
            except Exception as e:
                warnings.warn(f"Failed to evaluate IO dependency '{io_str}': {e}")

        dyn_state, stat_state, state_mask = io_partition(state, active_ids)
        dyn_system, stat_system, system_mask = io_partition(system, active_ids)

        if settings._DEV_MODE:
            inspect_leaves(state, state_mask, settings, tree_name="state", depth=3)
            inspect_leaves(system, system_mask, settings, tree_name="system", depth=3)
        
        # Residual closure defined in _run_solver scope to avoid tracing self argument if it were a bound method
        @eqx.filter_jit
        def get_residuals(control_values, args):

            r_state, r_system = args

            full_state = eqx.combine(r_state, stat_state)
            full_system = eqx.combine(r_system, stat_system)

            if settings.DEBUG_MODE:
                global _analysis_stack, _trace_count
                if len(_analysis_stack) > len(_trace_count):
                        _trace_count.append(0)
                if _trace_count[_analysis_stack.index(self.tag)] > 1:
                    diff_args((control_values, full_state, full_system, settings))
                _trace_count[_analysis_stack.index(self.tag)] += 1
                print(f"\n--- {self.tag.upper()} PASS {_trace_count[_analysis_stack.index(self.tag)]} ---")

            control_state = self._update_controls(full_state, control_values, settings)
            analysis_state, analysis_system, analysis_settings = self.analyze(control_state, full_system, settings)

            res = self._get_residual_array(analysis_state, analysis_settings)
            updated_r_state, _ = eqx.partition(analysis_state, state_mask)
            updated_r_system, _ = eqx.partition(analysis_system, system_mask)
            
            return res, (updated_r_state, updated_r_system)
        
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

        # Special Run Modes

        if settings._DEV_MODE:

            @contextlib.contextmanager
            def track_jax_cache():
                "Check JAX logs to read the cache status."
                stream = io.StringIO()
                handler = logging.StreamHandler(stream)
                logger = logging.getLogger("jax._src.compiler")

                old_level = logger.level
                logger.setLevel(logging.DEBUG)
                logger.addHandler(handler)

                try:
                    yield stream
                finally:
                    logger.removeHandler(handler)
                    logger.setLevel(old_level)

            def get_cache_status(log_text: str) -> str:
                log_text = log_text.lower()
                if "cache hit" in log_text:
                    return "CACHE HIT"
                elif "cache miss" in log_text:
                    return "CACHE MISS"
                elif "writing" in log_text:
                    return "WRITING TO CACHE"
                else:
                    return "UNKNOWN CACHE STATUS (Check cache dir)"

            print(f"\n{'='*60}")
            print("Starting JAX AOT Compilation Profiler...")
            print(f"{'-'*60}")

            leaves = jax.tree_util.tree_leaves((control_values, (dyn_state, dyn_system)))
            print(f"Total Input Leaves: {len(leaves)}\n")

            print("1. Tracing Forward Pass and Lowering to HLO...")
            t0 = time.time()
            
            fwd_fn = lambda x: get_residuals(x, (dyn_state, dyn_system))
            fwd_lowered = eqx.filter_jit(fwd_fn).lower(control_values)
            print(f" - Forward Lowering Time: {time.time() - t0:.2f} seconds")

            fwd_hlo_text = fwd_lowered.as_text()
            fwd_graph_length = len(fwd_hlo_text.splitlines())
            print(f" - Forward XLA HLO Graph Size: {fwd_graph_length:,} Lines of Code")

            print(" - Compiling Forward Pass ...")
            t0 = time.time()
            with track_jax_cache() as log_stream:
                fwd_compiled = fwd_lowered.compile()
            t_comp = time.time() - t0
            cache_status = get_cache_status(log_stream.getvalue())
            print(f" - Forwrd XLA Compile Time: {t_comp:.2f} seconds ({cache_status})")

            print("\n2. Tracing Jacobian & Lowering to HLO...")
            t0 = time.time()
            jac_fn = lambda x: jax.jacrev(fwd_fn, has_aux=True)(x)
            jac_lowered = eqx.filter_jit(jac_fn).lower(control_values)
            print(f" - Jacobian Lowering Time: {time.time() - t0:.2f} seconds")

            jac_hlo_text = jac_lowered.as_text()
            jac_graph_length = len(jac_hlo_text.splitlines())
            print(f" - Jacobian XLA HLO Graph Size: {jac_graph_length:,} Lines of Code")

            print(" - Compiling Jacobian (Checking Cache)...")
            t0 = time.time()
            with track_jax_cache() as log_stream:
                jac_compiled = jac_lowered.compile()
            t_comp = time.time() - t0
            cache_status = get_cache_status(log_stream.getvalue())
            print(f" - XLA Compile Time: {t_comp:.2f} seconds ({cache_status})")
            
            # We use a lambda to cleanly pass all arguments to the solver's run method
            if isinstance(self.solver, str):
                pass
            else:
                print("\n3. Tracing Full Optimistix Solver & Lowering...")
                t0 = time.time()
                run_fn = lambda c, st, sy: optx.root_find(
                    fn=get_residuals,
                    solver=self.solver(**solver_options),
                    y0=c,
                    args=(st, sy),
                    max_steps=settings.numerical.max_evaluations,
                )
                solver_lowered = eqx.filter_jit(run_fn).lower(control_values, dyn_state, dyn_system)
                print(f" - Solver Lowering Time : {time.time() - t0:.2f} seconds")

                # 2. Measure the Graph Size
                solver_hlo_text = solver_lowered.as_text()
                solver_graph_length = len(solver_hlo_text.splitlines())
                print(f" - Solver XLA HLO Graph Size: {solver_graph_length:,} Lines of Code")

                if solver_graph_length < settings.numerical.maximum_graph_complexity:
                    print(" - Compiling Solver (Checking Cache)...")
                    t0 = time.time()
                    with track_jax_cache() as log_stream:
                        solver_compiled = solver_lowered.compile()
                    t_compile = time.time() - t0
                    
                    cache_status = get_cache_status(log_stream.getvalue())
                    print(f" - Solver XLA Compile Time: {t_compile:.2f} seconds ({cache_status})")
                    print(f"{'='*60}\n")
                else:
                    sys.exit(f"Graph complexity ({solver_graph_length:,}) higher than "
                             f"settings.numerical.maximum_graph_complexity ({settings.numerical.maximum_graph_complexity:,}). "
                             "Terminating.")

        if settings.DEBUG_MODE:    
            print(f"DEBUG MODE: Executing single forward pass...")
            _, (f_st, f_sys) = get_residuals(control_values, (dyn_state, dyn_system))
            f_ctrls = self._get_control_array(f_st, settings)
            opt_state = None
        
        else:
            if isinstance(self.solver, str):
                run_fn = self._run_scipy_solver
            else:
                run_fn = self._run_optx_solver

            f_ctrls, opt_state, f_st, f_sys = run_fn(
                get_residuals,
                control_values,
                state,
                system,
                settings,
                solver_options
            )

        full_state = eqx.combine(f_st, stat_state)
        full_system = eqx.combine(f_sys, stat_system)
        
        return f_ctrls, opt_state, full_state, full_system
    
    def __call__(self, state: State, system: System, settings: Settings):

        global _analysis_stack, _last_static, _last_shapes, _trace_count
        _analysis_stack.append(self.tag)
        _last_static = None
        _last_shapes = None

        if settings.DEBUG_MODE or settings._DEV_MODE:
            scan_for_invalid_JAX_types(state,  "Analysis State")
            scan_for_invalid_JAX_types(system,  "Analysis System")

        # Get analysis control values 
        self._check_controls_balance(settings)
        initial_control_values = self._get_control_array(state, settings)

        # Run Solver
        with TraceReadout(enabled=not settings.DEBUG_MODE and not settings._DEV_MODE and len(_analysis_stack) == 1,
                     message=f"Tracing {self.tag}..."):
            
            f_ctrls, opt_state, f_st, f_sys = self._run_solver(
                initial_control_values,
                state,
                system,
                settings,
            )

        # Post-Processing
        if settings.verbose and len(_analysis_stack) == 1:
            f_res = self._get_residual_array(f_st, settings)
            self._report_results(f_ctrls, f_res, opt_state)
        
        if settings._DEV_MODE:
            print(f"\n{'='*70}")
            print(f"Full {self.tag} Solver State")
            print(f"{'-'*70}")
            from pprint import pprint
            pprint(opt_state)
            print(f"\n{'='*70}")

        # del _analysis_stack[-1]
        # del _trace_count[-1]

        return f_st, f_sys, settings

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

    def run(self, state: State, system: System, settings:Settings, *, initialize=True, track_history: bool = False):

        if initialize:
            state, system, settings = array_barrier(state, system, settings)
            state, system, settings = self.initialize_controls(state, system, settings)

        if not track_history:
            return self(state, system, settings)

        if settings.verbose:
            print(f"Residual analysis '{self.tag}' called with track_history enabled. "
                  "History returned will be single forward pass with final input values.")
            
        r_st, r_sys, r_setts = self(state, system, settings)
        f_st, f_sys, f_setts, history = self.analyze.run(r_st, r_sys, r_setts, initialize=initialize, track_history=True)
        return f_st, f_sys, f_setts, history
        

        