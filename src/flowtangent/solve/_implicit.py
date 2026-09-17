# flowtangent/Framework/Analyses/residual.py
# (c) Copyright 2026 Aerospace Research Community LLC
#
# Created: Jun 2026, J. Smart
# Modified: Jun 2026, J. Smart

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .. import Settings, State, System

import contextlib
import io
import logging
import os
import sys
import threading
import time
import warnings
from collections import Counter
from typing import Any, Callable, Literal, Optional, overload

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from jax.core import Tracer
from scipy.optimize import root

from .. import utils as ftu
from ..core._processes import Process, array_barrier
from ..utils import Module, update

jax.config.update("jax_enable_x64", True)

# ----------------------------------------------------------------------------------------------------------------------
#  Helper/Diagnostic Functions
# ----------------------------------------------------------------------------------------------------------------------


class Readout:
    def __init__(self, message="Tracing ...", enabled=True):
        self.spinner_chars = "|/-\\"
        self.message = message
        self.enabled = enabled
        self.running = False
        self.thread = None

        self.start_time = None
        self.null_fd = None
        self.saved_stderr_fd = None

    def run(self):
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
            self.saved_stderr_fd = os.dup(2)  # Save the real stderr
            os.dup2(self.null_fd, 2)  # Point stderr to black hole

            self.start_time = time.time()
            self.running = True
            self.thread = threading.Thread(target=self.run)
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
                os.close(self.null_fd)  # type: ignore

            elapsed = int(time.time() - self.start_time)  # type: ignore
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
        dynamic,
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
            print("  [TREEDEF CHANGED] Input PyTree structure mutated.")
            differences_found = True
        else:
            for (path, old_val), (_, new_val) in zip(old_stat, new_stat):
                # Check both value and exact type
                if type(old_val) is not type(new_val) or old_val != new_val:
                    print(f"  [STATIC VALUE CHANGED] {jax.tree_util.keystr(path)}")
                    print(f"    Old: {type(old_val)} {old_val} | New: {type(new_val)} {new_val}")
                    differences_found = True

        if not differences_found:
            print("  [IDENTICAL INPUTS] PyTree structure is identical.")

    _last_static = static
    _last_shapes = shapes


def analyze_compute_graph(func, *args):
    print("Tracing AD graph to count operations...")

    # Flowtangent the Jacobian
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

                    short_file = file_name.split("/")[-1].split("\\")[-1]
                    user_location = f"{func_name} ({short_file}:{line_num})"
                    break  # Found the user code, stop walking up the stack

            source_counts[user_location] += 1
        else:
            source_counts["Unknown Source"] += 1

    print("\n" + "=" * 60)
    print("Top 20 Functions by Node Count")
    print("=" * 60)

    total_nodes = sum(source_counts.values())

    for loc, count in source_counts.most_common(20):
        percentage = (count / total_nodes) * 100
        print(f"{count:8d} nodes ({percentage:4.1f}%) | {loc}")

    print("=" * 60)
    print(f"Total Nodes Analyzed: {total_nodes}")


# ----------------------------------------------------------------------------------------------------------------------
#  Variables and Residuals
# ----------------------------------------------------------------------------------------------------------------------


class Variable(Module):
    """
    State variable scaled for root-finding and optimization solvers.

    Scaling Methods:
    - 'linear': Direct proportional scaling. Best for unconstrained, O(1) variables.
    - 'log_bounded': Exponential sigmoid. Best for variables with strict upper/lower bounds.
    - 'algebraic_bounded': Slower-decaying sigmoid. Best for tight bounds where gradient starvation is a risk.
    - 'softplus': One-sided bound (y > 0). Best for mass, pressure, or physical properties without an upper limit.
    - 'logarithmic': Base-10 scaling. Best for strictly positive vars. spanning multiple orders of magnitude (e.g., Re).
    """

    state_path: ftu.TreePath = ftu.static_field(ftu.TreePath)
    initial_value: ftu.TimeScalar | ftu.ScalarFloat = ftu.field(1.0)
    bounds: tuple[ftu.ScalarFloat, ftu.ScalarFloat] = ftu.static_field((-1e6, 1e6))

    scaling: Literal[
        "linear",
        "log_bounded",
        "algebraic_bounded",
        "softplus",
        "logarithmic",
    ] = ftu.static_field("log_bounded")

    # ==========================================================================
    # Unscaling (Solver space -> Physical space)
    # ==========================================================================

    def _linear_unscale(self, val):
        return val * self.initial_value

    def _log_bounded_unscale(self, val):
        lb, ub = self.bounds
        return lb + (ub - lb) * jax.nn.sigmoid(val)

    def _algebraic_bounded_unscale(self, val):
        lb, ub = self.bounds
        # Algebraic sigmoid maps (-inf, inf) to (-0.5, 0.5)
        alg_sig = (val / jnp.sqrt(1.0 + val**2)) / 2.0
        return lb + (ub - lb) * (alg_sig + 0.5)

    def _softplus_unscale(self, val):
        return jax.nn.softplus(val)

    def _logarithmic_unscale(self, val):
        return 10.0**val

    def unscale(self, val: ftu.TimeScalar | ftu.ScalarFloat) -> ftu.TimeScalar | ftu.ScalarFloat:
        func = getattr(self, f"_{self.scaling}_unscale")
        return func(val)

    # ==========================================================================
    # Scaling (Physical space -> Solver space ~1.0)
    # ==========================================================================

    def _linear_scale(self, val):
        return val / self.initial_value

    def _log_bounded_scale(self, val):
        lb, ub = self.bounds
        norm = jnp.clip((val - lb) / (ub - lb), 1e-6, 1.0 - 1e-6)
        return jnp.log(norm / (1.0 - norm))

    def _algebraic_bounded_scale(self, val):
        lb, ub = self.bounds
        norm = jnp.clip((val - lb) / (ub - lb), 1e-6, 1.0 - 1e-6)
        # Shift domain from [0, 1] to [-0.5, 0.5]
        m = norm - 0.5
        return (2.0 * m) / jnp.sqrt(1.0 - 4.0 * m**2)

    def _softplus_scale(self, val):
        # Inverse softplus: x = log(exp(y) - 1). Use expm1 for numerical stability.
        val_safe = jnp.clip(val, 1e-6, None)
        return jnp.log(jnp.expm1(val_safe))

    def _logarithmic_scale(self, val):
        val_safe = jnp.clip(val, 1e-6, None)
        return jnp.log10(val_safe)

    def scale(self, val: ftu.TimeScalar | ftu.ScalarFloat) -> ftu.TimeScalar | ftu.ScalarFloat:
        func = getattr(self, f"_{self.scaling}_scale")
        return func(val)

    # ==========================================================================
    # Initialization
    # ==========================================================================

    def __post_init__(self):

        if self.bounds[0] > self.bounds[1]:
            warnings.warn(f"Variable '{self.name}' initialized with out-of-order bounds: {self.bounds}. Reversing...")
            object.__setattr__(self, "bounds", (self.bounds[1], self.bounds[0]))

        if self.scaling == "linear":
            if self.initial_value is None:
                raise ValueError(f"Variable '{self.name}' uses 'linear' scaling but has no initial_value.")
            if jnp.any(self.initial_value == 0.0):
                raise ValueError(f"Variable '{self.name}' cannot use 'linear' scaling with an initial_value of 0.")

        if self.initial_value is not None:
            safe_init = jnp.clip(self.initial_value, self.bounds[0] * 1.10, self.bounds[1] * 0.90)
            object.__setattr__(self, "initial_value", safe_init)

            # --- Emit a warning if JAX clipped the value at runtime ---
            needs_clip = jnp.any(safe_init != self.initial_value)
            jax.lax.cond(
                needs_clip,
                lambda: jax.debug.print(f"Warning: initial_value for '{self.name}' was outside bounds and clipped."),
                lambda: None,
            )


class Residual(Module):
    state_path: Optional[ftu.TreePathLike] = ftu.static_field(None)
    value_func: Optional[Callable[["State"], ftu.TimeScalar]] = ftu.method_field(None)

    def __post_init__(self):
        has_path = self.state_path is not None
        has_func = self.value_func is not None

        if has_path and has_func:
            raise ValueError(f"Residual '{self.name}' cannot have both a state_path and a value_func defined.")
        if not has_path and not has_func:
            raise ValueError(f"Residual '{self.name}' must have either a state_path or a value_func defined.")

    def get_value(self, state: "State") -> ftu.TimeScalar | ftu.ScalarFloat:
        if self.state_path is not None:
            return ftu.get_target(state, ftu.TreePath.cast(self.state_path))
        else:
            return self.value_func(state)  # type: ignore


# ----------------------------------------------------------------------------------------------------------------------
#  Implicit Analysis
# ----------------------------------------------------------------------------------------------------------------------


class ImplicitAnalysis(Process):

    analyze: Process = ftu.field(Process)
    solver: Any | str = ftu.method_field(optx.LevenbergMarquardt)
    solver_options: Optional[dict] = ftu.static_field(None)

    variables: tuple[Variable, ...] = ftu.field(tuple)
    residuals: tuple[Residual, ...] = ftu.field(tuple)

    def __init__(
        self,
        analyze: Process = Process(name="Implicit Analysis Forward Pass"),
        name: ftu.NameType = "ImplicitAnalysis",
        solver: Any | str = optx.LevenbergMarquardt,
        solver_options: Optional[dict] = None,
        variables: tuple[Variable, ...] = (),
        residuals: tuple[Residual, ...] = (),
        *,
        _initial_state: Optional[State] = None,
        _initial_system: Optional[System] = None,
        _initial_settings: Optional[Settings] = None,
        _filter_map: Optional[dict] = None,

    ) -> None:

        # Standard field assignments
        self.name = name
        self.function = ftu.null_step
        self.initial_step = 0
        self._initial_state = _initial_state
        self._initial_system = _initial_system
        self._initial_settings = _initial_settings

        # Handle mutable dictionary default safely
        self._filter_map = (
            _filter_map
            if _filter_map is not None
            else {
                "energy": r"state\.energy\.nodes\.\[*\].",
            }
        )

        # Implicit specific attributes
        self.analyze = analyze
        self.solver = solver
        self.solver_options = solver_options
        self.variables = variables
        self.residuals = residuals

    def _report_results(self, f_vars: jax.Array, f_res: jax.Array, opt_stats=None):

        print(f"\n{'=' * 70}")
        print(f"Final {self.name} Solver State")
        print(f"{'-' * 70}")

        if opt_stats:
            try:
                if isinstance(self.solver, str):
                    solver_name = f"Scipy Root; Method: {self.solver}"
                    iter_num = opt_stats.nit
                    avg_res = np.mean(np.asarray(f_res)).item()
                else:
                    solver_name = f"Optimistix Least Squares; Method: {self.solver.__name__}"
                    iter_num = opt_stats["num_steps"].item()
                    avg_res = np.mean(np.asarray(f_res)).item()

                print(f"  Solver          : {solver_name}")
                print(f"  Num. Iterations : {iter_num}")
                print(f"  Avg. Residual   : {avg_res:.4e}")
            except Exception as e:
                print(f"  ERROR: Optimizer state parsing error: {e} Printing raw results...")
                print(opt_stats)

        # Determine the maximum name length
        active_variables = self.variables
        active_residuals = self.residuals

        all_names = [v.name for v in active_variables] + [r.name for r in active_residuals]
        # Default to 20 if empty, otherwise add 2 spaces of buffer to the longest name
        pad = max((len(str(t)) for t in all_names), default=20) + 2

        # Run the forward pass one last time
        print("\n  Final Variable Values:")
        for idx, var in enumerate(active_variables):
            print(f"    {var.name:<{pad}}: {ftu.format_array(var.unscale(f_vars[idx]))}")

        print("\n  Final Residual Values:")
        for i, res in enumerate(self.residuals):
            print(f"    {res.name:<{pad}}: {ftu.format_array(f_res[i])}")

        print(f"{'=' * 70}\n")

    def _check_variables_balance(self, settings: Settings) -> bool:
        """
        Checks that the number of active variables is equal to the number of active dynamics residuals.
        """

        valid_variables = len(self.variables) == len(self.residuals)

        if settings.verbose:
            print("\n")
            print("=" * 70)
            print(f" {self.name} Variables Setup")
            print("-" * 70)

            active_variables = self.variables
            active_residuals = self.residuals

            all_names = [v.name for v in active_variables] + [r.name for r in active_residuals]
            # Default to 20 if empty, otherwise add 2 spaces of buffer to the longest name
            pad = max((len(str(t)) for t in all_names), default=20) + 2

            print(f"\n{'Active Variables':<{pad + 2}}| {'Init. Values':<13}| Bounds")
            print("-" * 65)
            for variable in active_variables:
                print(
                    f"- {variable.name:<{pad}}| "
                    f"{ftu.format_array(variable.initial_value, width=12):>12} | "
                    f"{ftu.format_array(jnp.asarray(variable.bounds))}"
                )

            print("\nActive Residuals")
            print("-" * 65)
            for residual in active_residuals:
                if residual.state_path is not None:
                    print(f"- {residual.name:<{pad}}| path: {residual.state_path}")
                elif residual.value_func is not None:
                    print(f"- {residual.name:<{pad}}| func: {residual.value_func.__name__}")
                else:
                    print(f"- {residual.name:<{pad}}| WARNING: NO PATH OR FUNCTION SET")
            print("=" * 70)
            print("\n")

        return valid_variables

    def _update_variables(self, state: State, variable_values: jax.Array, settings: Settings) -> State:

        var_state = state
        if settings.numerical.sum_residuals:
            N = 1
        else:
            N = state.time.N
        var_idx = 0

        for var in self.variables:
            solver_logit = variable_values[var_idx : var_idx + N]
            new_val = var.unscale(solver_logit[:N])
            var_state = update(
                var_state,
                lambda s: ftu.get_target(s, var.state_path),
                jnp.atleast_2d(new_val).reshape((-1, 1)),
            )
            var_idx += N

        return var_state

    def initialize_variables(self, state: State, system: System, settings: Settings) -> tuple[State, System, Settings]:
        var_values = []

        for var in self.variables:
            n_cp = state.time.N
            var_values.append(jnp.full((n_cp, 1), var.scale(var.initial_value)))

        var_state = self._update_variables(state, jnp.concatenate(var_values, axis=0), settings)

        return var_state, system, settings

    def _get_variable_array(self, state: State, settings: Settings) -> jax.Array:
        var_vals = []
        for var in self.variables:
            current_val = ftu.get_target(state, var.state_path)
            logit_val = var.scale(current_val)
            if settings.numerical.sum_residuals:
                logit_val = jnp.atleast_2d(logit_val[0])  # Only take a batch instance as the variable value
            var_vals.append(logit_val)

        return jnp.concatenate(var_vals, axis=0).flatten()

    def _get_residual_array(self, state: State, settings: Settings) -> jax.Array:

        residual_values = [r.get_value(state) for r in self.residuals]
        if settings.numerical.sum_residuals:
            residual_values = [jnp.sum(r, axis=0) for r in residual_values]
        return jnp.concatenate(residual_values, axis=0).flatten()

    def _run_scipy_solver(
        self,
        get_residuals: Callable,
        variable_values: jax.Array,
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
            x0=np.array(variable_values),
            jac=scipy_jac,
            method=self.solver,
            options=solver_options,
        )

        if settings.DEBUG_MODE:
            print(f"\n--- {str(self.name).upper()} CLOSEOUT PASS ---")
        _, (f_st, f_sys) = get_residuals(jnp.array(results.x), args)

        return results.x, results, f_st, f_sys

    @eqx.filter_jit
    def _run_optx_solver(
        self,
        get_residuals: Callable,
        variable_values: jax.Array,
        state: State,
        system: System,
        settings: Settings,
        solver_options: dict,
    ):
        assert isinstance(self.solver, Callable)
        results = optx.least_squares(
            fn=get_residuals,
            solver=self.solver(**solver_options),
            y0=variable_values,
            args=(state, system),
            max_steps=settings.numerical.max_evaluations,
            has_aux=True,
        )

        final_state, final_system = results.aux

        return results.value, results.stats, final_state, final_system

    def _run_solver(
        self,
        variable_values,
        state: State,
        system: System,
        settings: Settings,
    ):

        dyn_state, stat_state, state_mask, dyn_system, stat_system, system_mask = self._partition_inputs(
            state, system, settings
        )

        # Residual closure defined in _run_solver scope to avoid tracing self argument if it were a bound method
        @eqx.filter_jit
        def get_residuals(variable_values, args):

            r_state, r_system = args

            full_state = eqx.combine(r_state, stat_state)
            full_system = eqx.combine(r_system, stat_system)

            if settings.DEBUG_MODE:
                global _analysis_stack, _trace_count
                if len(_analysis_stack) > len(_trace_count):
                    _trace_count.append(0)
                if _trace_count[_analysis_stack.index(self.name)] > 1:
                    diff_args((variable_values, full_state, full_system, settings))
                _trace_count[_analysis_stack.index(self.name)] += 1
                print(f"\n--- {str(self.name).upper()} PASS {_trace_count[_analysis_stack.index(self.name)]} ---")

            variable_state = self._update_variables(full_state, variable_values, settings)
            analysis_state, analysis_system, analysis_settings = self.analyze(variable_state, full_system, settings)

            res = self._get_residual_array(analysis_state, analysis_settings)
            updated_r_state, _ = eqx.partition(analysis_state, state_mask)
            updated_r_system, _ = eqx.partition(analysis_system, system_mask)

            return res, (updated_r_state, updated_r_system)

        # Run solver w/ dev mode profiling -----------------------------------------------------------------------------
        if self.solver_options is None:
            if isinstance(self.solver, str):
                if self.solver == "hybr":
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
                    "atol": settings.numerical.absolute_tolerance,
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

            print(f"\n{'=' * 60}")
            print("Starting JAX AOT Compilation Profiler...")
            print(f"{'-' * 60}")

            leaves = jax.tree_util.tree_leaves((variable_values, (dyn_state, dyn_system)))
            print(f"Total Input Leaves: {len(leaves)}\n")

            print("1. Tracing Forward Pass and Lowering to HLO...")
            t0 = time.time()

            fwd_fn = lambda x: get_residuals(x, (dyn_state, dyn_system))  # noqa: E731
            fwd_lowered = eqx.filter_jit(fwd_fn).lower(variable_values)  # type: ignore
            print(f" - Forward Lowering Time: {time.time() - t0:.2f} seconds")

            fwd_hlo_text = fwd_lowered.as_text()
            fwd_graph_length = len(fwd_hlo_text.splitlines())
            print(f" - Forward XLA HLO Graph Size: {fwd_graph_length:,} Lines of Code")

            print(" - Compiling Forward Pass ...")
            t0 = time.time()
            with track_jax_cache() as log_stream:
                fwd_compiled = fwd_lowered.compile()  # noqa: F841
            t_comp = time.time() - t0
            cache_status = get_cache_status(log_stream.getvalue())
            print(f" - Forwrd XLA Compile Time: {t_comp:.2f} seconds ({cache_status})")

            print("\n2. Tracing Jacobian & Lowering to HLO...")
            t0 = time.time()
            jac_fn = lambda x: jax.jacrev(fwd_fn, has_aux=True)(x)  # noqa: E731
            jac_lowered = eqx.filter_jit(jac_fn).lower(variable_values)  # type: ignore
            print(f" - Jacobian Lowering Time: {time.time() - t0:.2f} seconds")

            jac_hlo_text = jac_lowered.as_text()
            jac_graph_length = len(jac_hlo_text.splitlines())
            print(f" - Jacobian XLA HLO Graph Size: {jac_graph_length:,} Lines of Code")

            print(" - Compiling Jacobian (Checking Cache)...")
            t0 = time.time()
            with track_jax_cache() as log_stream:
                jac_compiled = jac_lowered.compile()  # noqa: F841
            t_comp = time.time() - t0
            cache_status = get_cache_status(log_stream.getvalue())
            print(f" - XLA Compile Time: {t_comp:.2f} seconds ({cache_status})")

            # We use a lambda to cleanly pass all arguments to the solver's run method
            if isinstance(self.solver, str):
                pass
            else:
                print("\n3. Tracing Full Optimistix Solver & Lowering...")
                t0 = time.time()
                run_fn = lambda c, st, sy: optx.root_find(  # noqa: E731
                    fn=get_residuals,
                    solver=self.solver(**solver_options),  # type: ignore
                    y0=c,
                    args=(st, sy),
                    max_steps=settings.numerical.max_evaluations,
                )
                solver_lowered = eqx.filter_jit(run_fn).lower(variable_values, dyn_state, dyn_system)  # type: ignore
                print(f" - Solver Lowering Time : {time.time() - t0:.2f} seconds")

                # 2. Measure the Graph Size
                solver_hlo_text = solver_lowered.as_text()
                solver_graph_length = len(solver_hlo_text.splitlines())
                print(f" - Solver XLA HLO Graph Size: {solver_graph_length:,} Lines of Code")

                if solver_graph_length < settings.numerical.maximum_graph_complexity:
                    print(" - Compiling Solver (Checking Cache)...")
                    t0 = time.time()
                    with track_jax_cache() as log_stream:
                        solver_compiled = solver_lowered.compile()  # noqa: F841
                    t_compile = time.time() - t0

                    cache_status = get_cache_status(log_stream.getvalue())
                    print(f" - Solver XLA Compile Time: {t_compile:.2f} seconds ({cache_status})")
                    print(f"{'=' * 60}\n")
                else:
                    sys.exit(
                        f"Graph complexity ({solver_graph_length:,}) higher than "
                        f"maximum_graph_complexity ({settings.numerical.maximum_graph_complexity:,}). "
                        "Terminating."
                    )

        if settings.DEBUG_MODE:
            print("DEBUG MODE: Executing single forward pass...")
            _, (f_st, f_sys) = get_residuals(variable_values, (dyn_state, dyn_system))
            f_vars = self._get_variable_array(f_st, settings)
            opt_state = None

        else:
            if isinstance(self.solver, str):
                run_fn = self._run_scipy_solver
            else:
                run_fn = self._run_optx_solver

            f_vars, opt_state, _sol_st, _sol_sys = run_fn(
                get_residuals,
                variable_values,
                state,
                system,
                settings,
                solver_options,
            )

            # Closeout pass (mandatory for adjoints not passing through variables)
            _, (f_st, f_sys) = get_residuals(f_vars, (dyn_state, dyn_system))

        full_state = eqx.combine(f_st, stat_state)
        full_system = eqx.combine(f_sys, stat_system)

        return f_vars, opt_state, full_state, full_system

    def __call__(self, state: State, system: System, settings: Settings):

        global _analysis_stack, _last_static, _last_shapes, _trace_count
        _analysis_stack.append(self.name)
        _last_static = None
        _last_shapes = None

        if settings.DEBUG_MODE or settings._DEV_MODE:
            ftu.scan_for_invalid_JAX_types(state)
            ftu.scan_for_invalid_JAX_types(system)

        # Get analysis variable values
        self._check_variables_balance(settings)
        initial_variable_values = self._get_variable_array(state, settings)

        # Run Solver
        with Readout(
            enabled=settings.verbose and not settings.DEBUG_MODE and not settings._DEV_MODE and len(_analysis_stack) == 1,
            message=f"Tracing {self.name}...",
        ):
            f_vars, opt_state, f_st, f_sys = self._run_solver(
                initial_variable_values,
                state,
                system,
                settings,
            )

        # Post-Processing
        if settings.verbose and len(_analysis_stack) == 1:
            f_res = self._get_residual_array(f_st, settings)
            self._report_results(f_vars, f_res, opt_state)

        if settings._DEV_MODE:
            print(f"\n{'=' * 70}")
            print(f"Full {self.name} Solver State")
            print(f"{'-' * 70}")
            from pprint import pprint

            pprint(opt_state)
            print(f"\n{'=' * 70}")

        if settings.DEBUG_MODE:
            del _trace_count[_analysis_stack.index(self.name)]
        del _analysis_stack[-1]

        return f_st, f_sys, settings

    @property
    def steps(self): #type: ignore
        return self.analyze.steps

    def initialize(self, state: State, system: System, settings:Settings):
        state, system, settings = array_barrier(state, system, settings)
        state, system, settings = self.initialize_variables(state, system, settings)
        return state, system, settings

    @overload
    def run(
        self,
        state: State,
        system: System,
        settings: Settings,
        *,
        track_history: Literal[True],
    ) -> tuple[State, System, Settings, Process]: ...

    @overload
    def run(
        self,
        state: State,
        system: System,
        settings: Settings,
        *,
        track_history: Literal[False] = ...,
    ) -> tuple[State, System, Settings]: ...

    def run(self, state: State, system: System, settings: Settings, *, track_history: bool = False):

        state, system, settings = self.initialize(state, system, settings)

        r_st, r_sys, r_setts = self(state, system, settings)

        if not track_history:
            return r_st, r_sys, r_setts
        else:
            f_st, f_sys, f_setts, history = self.analyze.run(
                r_st, r_sys, r_setts, track_history=True
            )
            return f_st, f_sys, f_setts, history
