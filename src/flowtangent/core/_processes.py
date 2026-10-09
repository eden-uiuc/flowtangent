# flowtangent/Framework/Process.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Jul 2024, FlowTangent Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Callable,
    Generator,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypeAlias,
    overload,
)

if TYPE_CHECKING:
    from .. import Settings, State, System
    from ..solve import JacobianMap

    ProcessFunc: TypeAlias = Callable[[State, System, Settings], Tuple[State, System, Settings]]

import inspect
import os
import re
import time
import warnings
from collections import Counter
from dataclasses import replace
from datetime import datetime
from pathlib import Path

# package imports
import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np

from ..utils import (
    MERMAID_STYLES,
    Module,
    NameType,
    Partial,
    TreePath,
    compute_tree_delta,
    field,
    static_field,
    method_field,
    get_target,
    id_partition,
    inspect_leaves,
    is_array_like,
    null_step,
    update,
)
from ..utils.typing import _Placeholder

# ----------------------------------------------------------------------------------------------------------------------
#  ProcessStep
# ----------------------------------------------------------------------------------------------------------------------


class ProcessStep(Module):
    function: ProcessFunc = method_field(null_step)

    _state_delta: Optional[State] = field(None)
    _system_delta: Optional[System] = field(None)
    _settings_delta: Optional[Settings] = field(None)

    def __init__(
        self,
        function: ProcessFunc | ProcessStep = null_step,
        name: NameType = None,
        *,
        _state_delta: Optional[State] = None,
        _system_delta: Optional[System] = None,
        _settings_delta: Optional[Settings] = None,
    ):

        self.function = function
        if name is not None:
            self.name = name
        else:
            self.name = self.function.__name__
        self._state_delta = _state_delta
        self._system_delta = _system_delta
        self._settings_delta = _settings_delta

    @classmethod
    def cast(cls, step: Callable) -> ProcessStep:
        if isinstance(step, ProcessStep):
            return step

        safe_step = step if isinstance(step, Partial) else Partial(step)
        step_name = getattr(safe_step.func, "__name__", "Unnamed Function")
        sig = inspect.signature(safe_step.func)
        if len(sig.parameters) != 3:
            raise ValueError(
                f"Process functions must take and return (State, System, Settings). "
                f"Found function '{step_name}' with signature '{sig}'."
            )
        return cls(name=step_name, function=safe_step)

    def _profile_complexity(self, state: State, system: System, settings: Settings, top_n=5):
        try:
            assert isinstance(self.function, Callable)
            jaxpr_obj = jax.make_jaxpr(self.function)(state, system, settings)
        except Exception as e:
            return f" - {self.name} | Could not trace ({e})"

        source_counts = Counter()

        # 1. Define everything we want the profiler to IGNORE
        exclude_strings = [
            "jax/",
            "jax\\",
            "equinox",
            "jaxtyping",  # Core internals
            "residual.py",  # Orchestrator
            "graph_network.py",
        ]

        for eqn in jaxpr_obj.jaxpr.eqns:
            if eqn.source_info.traceback:
                user_location = "Unknown Source"

                for frame in reversed(eqn.source_info.traceback.frames):
                    file_name = getattr(frame, "file_name", None) or getattr(frame, "file", "")

                    # 2. Check if this frame is in our ignore list
                    is_ignored = any(bad_string in file_name for bad_string in exclude_strings)

                    # 3. Also ignore the wrapper function by name, just to be safe
                    func_name = getattr(frame, "code_name", None) or getattr(frame, "name", "")
                    is_wrapper_func = func_name in ["make_node_function", "transmit", "net_transmit"]

                    if file_name != "" and not is_ignored and not is_wrapper_func:
                        line_num = getattr(frame, "line_num", None) or getattr(frame, "lineno", "?")
                        short_file = file_name.split("/")[-1].split("\\")[-1]
                        user_location = f"{func_name} ({short_file}:{line_num})"
                        break

                source_counts[user_location] += 1
            else:
                source_counts["Unknown Source"] += 1

        total_ops = len(jaxpr_obj.jaxpr.eqns)
        report = [f" - {self.name} | Total Ops: {total_ops}"]
        for loc, count in source_counts.most_common(top_n):
            pct = (count / total_ops) * 100
            report.append(f" - - {count:4d} ops ({pct:4.1f}%) : {loc}")

        return "\n".join(report)

    def __call__(self, state: State, system: System, settings: Settings):
        if settings._DEV_MODE and settings.verbose:
            print(self._profile_complexity(state, system, settings))
        if not settings._DEV_MODE and settings.DEBUG_MODE:
            print(f" - {self.name}")
        # Default calling behavior, assumes function is callable.
        # String overwrite only for steps with __call__ override
        return self.function(state, system, settings)  # type: ignore

    def run(self, state, system, settings):
        return self(state, system, settings)

    def _run_with_history(self, state, system, settings):
        return *self(state, system, settings), None

    def __repr__(self):
        return str(self.name)

    @property
    def inputs(self) -> set:
        return getattr(self.function, "_inputs", set())

    @property
    def outputs(self) -> set:
        return getattr(self.function, "_outputs", set())


# ----------------------------------------------------------------------------------------------------------------------
#  Process Class
# ----------------------------------------------------------------------------------------------------------------------


def array_barrier(state: State, system: System, settings: Settings):
    """
    Forces every numerical leaf of state (at least 2d) and system (at least 1d) to become JAX arrays.
    This both standardizes the shape of the arrays and ensures they don't share memory
    (e.g. Python may cache every instance of 0.0 to be the same memory address) so that when
    they're partitioned based on memory ID for tracing, there's no collisions.
    """

    def _to_array(leaf, ndim: int = 1):

        if isinstance(leaf, _Placeholder):
            leaf_arr = jnp.zeros((1,) * ndim)
            return leaf_arr
        # Check if it's a raw scalar, a list/tuple of scalars, OR already an array
        is_scalar = isinstance(leaf, (float, int, complex))
        is_iterable = isinstance(leaf, (list, tuple)) and all(isinstance(i, (float, int, complex)) for i in leaf)
        is_array = isinstance(leaf, (jax.Array, np.ndarray))

        if is_scalar or is_iterable or is_array:
            # Convert to JAX array (jnp.asarray is a no-op if it's already a JAX array)
            # Using standard float allows JAX to respect its 32/64-bit config settings naturally
            leaf_arr = jnp.asarray(leaf, dtype=float)

            # 3. Enforce the minimum dimension barrier
            if leaf_arr.ndim < ndim:
                axes_to_add = tuple(range(ndim - leaf_arr.ndim))
                return jnp.expand_dims(leaf_arr, axis=axes_to_add)

            return leaf_arr

        # Leave strings, booleans, empty sentinels, or other metadata alone
        return leaf

    # Apply ndim=2 to State
    arr_state = jax.tree_util.tree_map(lambda x: _to_array(x, ndim=2), state)

    # Apply ndim=1 to System
    arr_system = jax.tree_util.tree_map(lambda x: _to_array(x, ndim=1), system)

    return arr_state, arr_system, settings


class Process(ProcessStep):
    steps: tuple[ProcessStep, ...] = ()

    initial_step: int = static_field(0)

    _initial_state: Optional[State] = field(None)
    _initial_system: Optional[System] = field(None)
    _initial_settings: Optional[Settings] = field(None)

    _filter_map: dict = field(lambda _: {"energy": r"state\.energy\.nodes\.\[*\]."}, static=True)

    def __init__(
        self,
        steps: Sequence[ProcessStep | ProcessFunc] = (),
        name: NameType = "Process",
        *,
        initial_step: int = 0,
        _initial_state: Optional[State] = None,
        _initial_system: Optional[System] = None,
        _initial_settings: Optional[Settings] = None,
        _filter_map: Optional[dict] = None,
    ):
        # Initialize the parent ProcessStep
        super().__init__(name=name)

        # Standard field assignments
        self.initial_step = initial_step
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

        self.steps = tuple(ProcessStep.cast(step) for step in steps)

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.steps[self._index_tag(item)]
        else:
            return self.steps[item]

    def __getattr__(self, key: str):
        if key.startswith("__") and key.endswith("__"):
            raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{key}'")

        try:
            steps = object.__getattribute__(self, "steps")
        except AttributeError:
            raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{key}'")

        #  Search the steps using tracer-safe logic
        for step in steps:
            step_tag = step.name
            if not isinstance(step_tag, str) and hasattr(step_tag, "value"):
                step_tag = step_tag.value

            if isinstance(step_tag, str):
                formatted_tag = step_tag.replace(" ", "_").lower()
                if key == formatted_tag:
                    return step

        raise AttributeError(f"{self.__class__.__name__}: {self.name} has no attribute '{key}'")

    def __call__(self, state: State, system: System, settings: Settings) -> tuple[State, System, Settings]:
        if settings.DEBUG_MODE or settings._DEV_MODE:
            start_time = datetime.fromtimestamp(time.time()).strftime(settings.logging.date_format)
            print(f"Beginning Process: '{self.name}' | {start_time}")

        if settings.numerical.jacobian.calculate:
            jac_map = settings.numerical.jacobian.mapping

            if jac_map is not None:
                # Returns the two distinct flat arrays
                flat_st, flat_sys = jac_map.flatten_inputs(state, system)
                val_and_jac_fn = self._build_value_and_jacobian(jac_map)

                jacobian_matrix, final_st, final_sys, final_setts = val_and_jac_fn(
                    flat_st, flat_sys, state, system, settings
                )

                final_st = update(final_st, "process_jacobian", jacobian_matrix)
                return final_st, final_sys, final_setts
            else:
                warnings.warn(
                    f"Process '{self.name}' Jacobian called with no Jacbian Map set. Jacobian will not be calculated."
                )

        # Standard Execution Path
        for step in self.steps[self.initial_step :]:
            state, system, settings = step(state, system, settings)

        if settings.DEBUG_MODE or settings._DEV_MODE:
            end_time = datetime.fromtimestamp(time.time()).strftime(settings.logging.date_format)
            print(f"Process '{self.name}' Complete. | {end_time}")

        return state, system, settings

    def _run_with_raw_history(self, state, system, settings):
        if settings.DEBUG_MODE or settings._DEV_MODE:
            print(f"Beginning Process: '{self.name}'")
        history = [(state, system, settings)]

        for step in self.steps[self.initial_step :]:
            state, system, settings = step(state, system, settings)
            history.append((state, system, settings))

        return state, system, settings, tuple(history)

    def _build_value_and_jacobian(self, jac_map: JacobianMap):

        def objective_fn(flat_st, flat_sys, base_state, base_system, base_settings):
            st, sys = jac_map.update_inputs(flat_st, flat_sys, base_state, base_system)

            # Prevent recursion by temporarily disabling the Jacobian flag
            inner_setts = update(
                base_settings,
                "numerical.jacobian",
                replace(
                    base_settings.numerical.jacobian,
                    calculate=False,
                ),
            )

            f_st, f_sys, f_setts = self(st, sys, inner_setts)
            out_array = jac_map.flatten_outputs(f_st, f_sys, f_setts)

            # Restore modified setting
            f_setts = update(f_setts, "numerical.jacobian", replace(f_setts.numerical.jacobian, calculate=True))

            return out_array, (f_st, f_sys, f_setts)

        def batched_jacrev_fn(flat_st, flat_sys, base_state, base_system, base_settings):
            out_array, vjp_fn, aux = jax.vjp(
                objective_fn, flat_st, flat_sys, base_state, base_system, base_settings, has_aux=True
            )

            is_coupled_time = getattr(base_settings.numerical, "coupled_time_jacobian", False)

            # 1. Universally parse leading dimensions (L) instead of rigid B and T
            L = out_array.shape[:-1]
            N_L = int(np.prod(L)) if L else 1
            N_o = out_array.shape[-1]

            if not is_coupled_time:
                # =========================================================
                # PATH A: FAST BLOCK-DIAGONAL
                # =========================================================
                # Broadcast the basis to match the leading dimensions dynamically
                basis_st = jnp.broadcast_to(jnp.eye(N_o).reshape((N_o,) + (1,) * len(L) + (N_o,)), (N_o,) + L + (N_o,))
                jac_tuple_st = jax.vmap(vjp_fn)(basis_st)

                jacs = []

                if flat_st.size > 0:
                    N_st = flat_st.shape[-1]
                    if flat_st.shape[:-1] == L:
                        # Input matches leading dims (e.g. batched state)
                        jac_st = jnp.moveaxis(jac_tuple_st[0], 0, -2)
                    else:
                        # Input lacks leading dims (e.g. empty array). Broadcast to match.
                        jac_st = jnp.broadcast_to(jac_tuple_st[0], L + (N_o, N_st))
                    jacs.append(jac_st)

                if flat_sys.size > 0:
                    N_sys = flat_sys.shape[-1]
                    if flat_sys.shape[:-1] == L:
                        jac_sys = jnp.moveaxis(jac_tuple_st[1], 0, -2)
                    else:
                        # System is unbatched. We need N_L * N_o passes to extract block diagonal.
                        basis_sys = jnp.eye(N_L * N_o).reshape((N_L * N_o,) + L + (N_o,))
                        jac_tuple_sys = jax.vmap(vjp_fn)(basis_sys)
                        jac_sys = jac_tuple_sys[1].reshape(L + (N_o, N_sys))

                    jacs.append(jac_sys)

                batched_jacobian = jnp.concatenate(jacs, axis=-1)

            else:
                # =========================================================
                # PATH B: DENSE TEMPORAL (Optimal Control)
                # =========================================================
                basis_st = jnp.eye(N_L * N_o).reshape((N_L * N_o,) + L + (N_o,))
                jac_tuple = jax.vmap(vjp_fn)(basis_st)

                jacs = []

                if flat_st.size > 0:
                    N_st = flat_st.shape[-1]
                    if flat_st.shape[:-1] == L:
                        # Dense coupling requires cross-referencing input and output leading dims
                        jac_st = jac_tuple[0].reshape(L + (N_o,) + L + (N_st,))
                    else:
                        jac_st = jac_tuple[0].reshape(L + (N_o, N_st))
                    jacs.append(jac_st)

                if flat_sys.size > 0:
                    N_sys = flat_sys.shape[-1]
                    if flat_sys.shape[:-1] == L:
                        jac_sys = jac_tuple[1].reshape(L + (N_o,) + L + (N_sys,))
                    else:
                        jac_sys = jac_tuple[1].reshape(L + (N_o, N_sys))

                    jacs.append(jac_sys)

                batched_jacobian = jnp.concatenate(jacs, axis=-1)

            return batched_jacobian, aux[0], aux[1], aux[2]

        return batched_jacrev_fn

    def _partition_inputs(self, state: State, system: System, settings: Settings):
        active_paths = [p.split(":")[0].strip() for p in self.full_io]
        active_ids = set()

        ctx = {"state": state, "system": system}
        for io_str in active_paths:
            try:
                target_obj = eval(io_str, {}, ctx)
                leaves = jax.tree_util.tree_leaves(target_obj)
                for leaf in leaves:
                    if is_array_like(leaf):
                        active_ids.add(id(leaf))
            except Exception as e:
                warnings.warn(f"Failed to evaluate IO dependency '{io_str}': {e}")

        dyn_state, stat_state, state_mask = id_partition(state, active_ids)
        dyn_system, stat_system, system_mask = id_partition(system, active_ids)

        if settings._DEV_MODE:
            inspect_leaves(state, state_mask, settings, tree_name="state", depth=3)
            inspect_leaves(system, system_mask, settings, tree_name="system", depth=3)

        return dyn_state, stat_state, state_mask, dyn_system, stat_system, system_mask

    def initialize(self, state: State, system: System, settings: Settings):
        state, system, settings = array_barrier(state, system, settings)
        return state, system, settings

    def _compute_jacobian(self, state, system, settings, jac_map):

        flat_st, flat_sys = jac_map.flatten_inputs(state, system)

        if jac_map._n_st > 0 and jac_map._n_sys > 0:

            def func(flat_st, flat_sys):
                new_state, new_system = jac_map.update_inputs(flat_st, flat_sys, state, system)
                f_st, f_sys, f_setts = self(new_state, new_system, settings)
                return jac_map.flatten_outputs(f_st, f_sys, f_setts)

            jac_st, jac_sys = jax.jacrev(func, argnums=(0, 1))(flat_st, flat_sys)
            jac = jnp.concatenate((jac_st, jac_sys), axis=-1)

        elif jac_map._n_st > 0:

            def func(flat_st):
                new_state, new_system = jac_map.update_inputs(flat_st, flat_sys, state, system)
                f_st, f_sys, f_setts = self(new_state, new_system, settings)
                return jac_map.flatten_outputs(f_st, f_sys, f_setts)

            jac = jax.jacrev(func)(flat_st)

        elif jac_map._n_sys > 0:

            def func(flat_sys):
                new_state, new_system = jac_map.update_inputs(flat_st, flat_sys, state, system)
                f_st, f_sys, f_setts = self(new_state, new_system, settings)
                return jac_map.flatten_outputs(f_st, f_sys, f_setts)

            jac = jax.jacrev(func)(flat_sys)

        else:
            raise ValueError("JacobianMap contains no inputs.")

        return jac

    @overload
    def run(
        self, state: State, system: System, settings: Settings, *, track_history: Literal[True]
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

        # Direct call if not tracking history
        if not track_history:
            return self(state, system, settings)
        else:
            f_st, f_sys, f_setts, raw_hist = self._run_with_raw_history(state, system, settings)

            logged_process = None
            logged_steps = []

            for i, step in enumerate(self.steps[self.initial_step :]):
                logged_step = update(
                    step,
                    (
                        ("state_delta", compute_tree_delta(raw_hist[i + 1][0], raw_hist[i][0])),
                        ("system_delta", compute_tree_delta(raw_hist[i + 1][1], raw_hist[i][1])),
                        ("settings_delta", compute_tree_delta(raw_hist[i + 1][2], raw_hist[i][2])),
                    ),
                )
                logged_steps.append(logged_step)

                logged_process = update(
                    self,
                    (
                        ("steps", tuple(logged_steps)),
                        ("initial_state", state),
                        ("initial_system", system),
                        ("initial_settings", settings),
                        ("state_delta", compute_tree_delta(f_st, state)),
                        ("system_delta", compute_tree_delta(f_sys, system)),
                        ("settings_delta", compute_tree_delta(f_setts, settings)),
                    ),
                    is_leaf=lambda x: x is None,
                )

                return f_st, f_sys, f_setts, logged_process

    def append(self, step: ProcessStep | Process):
        new_steps = self.steps + (step,)
        return update(self, "steps", new_steps)

    def count(self, step: ProcessStep):
        return self.steps.count(step)

    def _index_tag(self, name: str):
        names = [step.name for step in self.steps]
        if name not in names:
            names = [n.replace(" ", "_").lower() for n in names]
        if name not in names:
            raise AttributeError(f"Unable to locate step {name} in steps of Process {self.name}.")
        index = names.index(name)

        return index

    def _index_function(self, function: Callable):
        functions = [step.function for step in self.steps]
        index = functions.index(function)

        return index

    def index(self, value: str | Callable | ProcessStep | Process):
        if isinstance(value, str):
            return self._index_tag(value)
        elif isinstance(value, Callable):
            return self._index_function(value)
        elif isinstance(value, ProcessStep):
            return self.steps.index(value)

        else:
            raise ValueError("FlowTangent processes can only be indexed by name, function, or ProcessStep object.")

    def insert(self, step: ProcessStep, index: int):
        new_steps = self.steps[:index] + (step,) + self.steps[index:]
        return update(self, "steps", new_steps)

    def pop(self, index: int):
        new_steps = self.steps[:index] + self.steps[index + 1 :]
        return update(self, "steps", new_steps)

    def _remove_tag(self, name: str):
        return self.pop(self._index_tag(name))

    def _remove_function(self, function: Callable):
        return self.pop(self._index_function(function))

    def remove(self, value: str | Callable | ProcessStep | Process):
        idx_to_remove = self.index(value)
        return self.pop(idx_to_remove)

    @property
    def inputs(self) -> set:
        required_inputs = set()
        available_inputs = set()

        for step in self.steps:
            unmet_needs = step.inputs - available_inputs
            required_inputs |= unmet_needs
            available_inputs |= step.outputs

        return required_inputs

    @property
    def outputs(self) -> set:
        return set.union(*[step.outputs for step in self.steps]) if self.steps else set()

    @property
    def full_io(self) -> set:
        all_io = set()
        for step in self.steps:
            all_io |= step.inputs
            all_io |= step.outputs
        return all_io

    def _get_flattened_steps(self, prefix: str = "") -> Generator[Tuple[str, ProcessStep], None, None]:
        """
        Recursively yields (node_name, step_obj) for all steps.
        The prefix tracks the hierarchical origin (e.g., '0_InitializeVLM.2_ProcessGeometry').
        """
        for step in self.steps:
            node_name = step.name

            # Check if this step is itself a nested Process containing other steps
            if hasattr(step, "steps") and step.steps is not None:
                # Recursively yield its internal steps
                yield from step._get_flattened_steps(prefix=f"{prefix}{node_name}.")
            else:
                yield node_name, step

    def _format_ascii_tree(self, paths: set[str] | list[str]) -> str:
        """
        Parses a list of variable paths and returns a formatted ASCII tree string.
        Handles iterators ([Item]), dictionaries (['key']), and pseudo-types (: Type).
        """
        if not paths:
            return ""

        import re
        tree = {}
        type_hints = {}
        # Regex extracts bracketed items (with or without quotes) and normal text, ignoring dots
        pattern = re.compile(r"\[.*?\]|[^.\[\]]+")

        for path in paths:
            # 1. Strip out and store the type hint if it exists
            if ":" in path:
                base_path, hint = path.split(":", 1)
                base_path = base_path.strip()
                hint = hint.strip()
            else:
                base_path = path.strip()
                hint = None

            # 2. Split the base path into parts
            parts = tuple(pattern.findall(base_path))

            if hint:
                type_hints[parts] = hint

            # 3. Build the structural tree
            current_level = tree
            for part in parts:
                current_level = current_level.setdefault(part, {})

        # 4. Recursively build the string representation
        lines = []
        def traverse(current_tree: dict, current_parts: tuple = (), depth: int = 0):
            keys = sorted(current_tree.keys())
            for i, key in enumerate(keys):
                node_parts = current_parts + (key,)
                display_name = str(key)

                if node_parts in type_hints:
                    display_name += f": {type_hints[node_parts]}"

                # Flag dictionary children
                has_dict_children = any(str(k).startswith("['") or str(k).startswith('["') for k in current_tree[key].keys())
                if has_dict_children:
                    display_name += ": {dict}"

                if depth == 0:
                    lines.append(display_name)
                else:
                    padding = "  " * (depth - 1)
                    branch = "└─ " if i == len(keys) - 1 else "├─ "
                    lines.append(f"{padding}{branch}{display_name}")

                traverse(current_tree[key], node_parts, depth + 1)

        traverse(tree)
        return "\n".join(lines)

    def graph(self, recursive: bool = False) -> nx.DiGraph:
        """
        Constructs a Directed Acyclic Graph (DAG) of the process.

        Args:
            recursive: If True, flattens nested Processes into their atomic base steps.
                       If False, treats nested Processes as single black-box nodes.
        """
        G = nx.DiGraph()
        latest_producers = {}

        if recursive:
            step_iterator = self._get_flattened_steps()
        else:
            step_iterator = ((f"{i}_{step.name}", step) for i, step in enumerate(self.steps))

        # 2. Build the chronological DAG
        for step_node, step in step_iterator:
            G.add_node(step_node, step_obj=step)

            # Resolve Inputs
            for in_var in step.inputs:
                if in_var in latest_producers:
                    producer_node = latest_producers[in_var]
                    if G.has_edge(producer_node, step_node):
                        G.edges[producer_node, step_node]["variables"].append(in_var)
                    else:
                        G.add_edge(producer_node, step_node, variables=[in_var])
                else:
                    global_node = "User Inputs"
                    if not G.has_node(global_node):
                        G.add_node(global_node)

                    if G.has_edge(global_node, step_node):
                        G.edges[global_node, step_node]["variables"].append(in_var)
                    else:
                        G.add_edge(global_node, step_node, variables=[in_var])

            # Resolve Outputs
            for out_var in step.outputs:
                latest_producers[out_var] = step_node

        return G

    def to_mermaid(
        self,
        recursive: bool = False,
        show_edges: bool = True,
        layout: str = "LR",
        exclude: Optional[list[str]] = None,
        save_path: Optional[str | Path] = None,
        style: str = "modern",
    ) -> str:
        """
        Generates a Mermaid.js flowchart string from the Process DAG.

        Args:
            recursive: Whether to flatten nested Processes.
            show_edges: Whether to label the edges with the variables passed between steps.
            layout: "LR" (Left-to-Right) or "TD" (Top-Down).
            exclude: List of predefined domains to hide from the edges (e.g., ['energy']).
            save_path: Path to write the output file.
            style: The visual style preset to use from MERMAID_STYLES.
        """

        # 1. Setup the exclusion filters
        if exclude is None:
            exclude = ["energy"]

        compiled_patterns = [
            re.compile(self._filter_map[k]) 
            for k in exclude if k in self._filter_map
        ]

        def is_filtered(var_name: str) -> bool:
            return any(pat.search(var_name) for pat in compiled_patterns)

        # 2. Grab the graph
        G = self.graph(recursive=recursive)
        mermaid_lines = []

        # Note: Ensure the MERMAID_STYLES dictionary uses strict JSON (double quotes) 
        # inside the init block string so Mermaid can parse it correctly!
        if style in MERMAID_STYLES and MERMAID_STYLES[style]:
            mermaid_lines.append(MERMAID_STYLES[style])

        mermaid_lines.append(f"graph {layout}")

        # 3. Build safe Node IDs and visual shapes
        node_id_map = {}
        for i, node_name in enumerate(G.nodes()):
            safe_id = f"N{i}"
            node_id_map[node_name] = safe_id

            if node_name == "User Inputs":
                mermaid_lines.append(f"    {safe_id}([{node_name}])")
            else:
                step_obj = G.nodes[node_name].get("step_obj")
                display_label = step_obj.name if step_obj else str(node_name)
                
                # Sanitize characters that break Mermaid node syntax
                display_label = display_label.replace('"', "").replace("[", "(").replace("]", ")")
                mermaid_lines.append(f"    {safe_id}[{display_label}]")

        # 4. Build edges and apply filters
        for u, v, data in G.edges(data=True):
            raw_vars = data.get("variables", [])
            vars_list = [var for var in raw_vars if not is_filtered(var)]

            if show_edges and vars_list:
                if len(vars_list) > 6:
                    label = f"{len(vars_list)} variables"
                elif len(vars_list) == 1:
                    # Don't try to build a tree for a single variable
                    label = vars_list[0]
                else:
                    # 1. Split into components to find the common prefix
                    split_vars = [v.split(".") for v in vars_list]
                    min_len = min(len(v) for v in split_vars)
                    
                    common_idx = 0
                    for i in range(min_len):
                        if len(set(v[i] for v in split_vars)) == 1:
                            common_idx += 1
                        else:
                            break
                            
                    # 2. Build the tree string
                    if common_idx > 0 and common_idx < min_len:
                        prefix = ".".join(split_vars[0][:common_idx])
                        suffixes = [".".join(v[common_idx:]) for v in split_vars]
                        
                        # Added the bold tags back in
                        tree_lines = [f"<b>{prefix}</b>"]
                        for i, suffix in enumerate(suffixes):
                            branch = "└─ " if i == len(suffixes) - 1 else "├─ "
                            tree_lines.append(f"{branch}{suffix}")
                            
                        # Use Mermaid's native literal "\n" token (requires \\n in Python)
                        label = "\\n".join(tree_lines)
                    else:
                        label = "\\n".join(vars_list)

                # Robust sanitization
                label = label.replace('"', "").replace("'", "").replace("|", "/")
                
                # Keep the escaped double quotes to protect the tree formatting
                mermaid_lines.append(f"    {node_id_map[u]} -->|\"{label}\"| {node_id_map[v]}")
        
        mermaid_str = "\n".join(mermaid_lines)

        # 5. Handle File Output
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            with open(save_path, "w", encoding="utf-8") as f:
                if save_path.suffix.lower() == ".md":
                    f.write("```mermaid\n")
                    f.write(mermaid_str)
                    f.write("\n```\n")
                else:
                    f.write(mermaid_str)

        return mermaid_str

    def to_cytoscape_json(self, recursive: bool = False, exclude: Optional[list[str]] = None) -> str:
        import json
        import re
        
        if exclude is None:
            exclude = ["energy"]

        compiled_patterns = [re.compile(self._filter_map[k]) for k in exclude if k in self._filter_map]
        def is_filtered(var_name: str) -> bool:
            return any(pat.search(var_name) for pat in compiled_patterns)

        G = self.graph(recursive=recursive)
        nodes = []
        edges = []

        # -- Pre-calculate all user inputs --
        user_input_vars = set()
        for u, v, data in G.edges(data=True):
            if u == "User Inputs":
                for var in data.get("variables", []):
                    if not is_filtered(var):
                        user_input_vars.add(var)
        
        # USE THE NEW FORMATTER
        ui_tree_str = self._format_ascii_tree(user_input_vars)
        ui_full_tree = f"USER INPUTS\n{ui_tree_str}" if ui_tree_str else "User Inputs"

        # 1. Build Primary Nodes
        name_to_id = {}
        for i, node_name in enumerate(G.nodes()):
            safe_id = f"N{i}"
            name_to_id[node_name] = safe_id
            
            if node_name == "User Inputs":
                nodes.append({
                    "data": {"id": safe_id, "label": node_name, "full_tree": ui_full_tree, "node_type": "input"}
                })
            else:
                step_obj = G.nodes[node_name].get("step_obj")
                label = step_obj.name if step_obj else str(node_name)
                nodes.append({
                    "data": {"id": safe_id, "label": label, "node_type": "process"}
                })

        # 2. Build Edges and Intermediate Variable Nodes
        for u, v, data in G.edges(data=True):
            raw_vars = data.get("variables", [])
            vars_list = [var for var in raw_vars if not is_filtered(var)]
            
            if not vars_list:
                edges.append({"data": {"source": name_to_id[u], "target": name_to_id[v], "edge_type": "direct"}})
                continue
                
            short_label = str(len(vars_list))
            
            # USE THE NEW FORMATTER
            full_tree = self._format_ascii_tree(vars_list)

            var_node_id = f"var_{name_to_id[u]}_{name_to_id[v]}"
            
            nodes.append({
                "data": {"id": var_node_id, "short_label": short_label, "full_tree": full_tree, "node_type": "variable"}
            })
            
            edges.append({"data": {"source": name_to_id[u], "target": var_node_id, "edge_type": "incoming"}})
            edges.append({"data": {"source": var_node_id, "target": name_to_id[v], "edge_type": "outgoing"}})

        return json.dumps({"nodes": nodes, "edges": edges}, indent=2)
    
    def print_io_tree(self, exclude: Optional[list[str]] = None):
        """
        Extracts the inputs and outputs of the Process and prints them
        in a hierarchical, human-readable ASCII tree structure.
        """
        import re
        
        if exclude is None:
            exclude = ["energy"]

        exclude_patterns = [self._filter_map[k] for k in exclude if k in self._filter_map]

        def filter_paths(paths: set[str]) -> set[str]:
            if not exclude_patterns or not paths:
                return paths
            compiled_patterns = [re.compile(p) for p in exclude_patterns]
            return {p for p in paths if not any(pat.search(p) for pat in compiled_patterns)}

        display_inputs = filter_paths(self.inputs)
        display_outputs = filter_paths(self.outputs)

        print("=== Process Inputs ===")
        print(self._format_ascii_tree(display_inputs) if display_inputs else "  (None)")

        print("\n=== Process Outputs ===")
        print(self._format_ascii_tree(display_outputs) if display_outputs else "  (None)")

    def find_variable_usage(self, search_term: str):
        """
        Searches recursively through all steps and prints a report of
        which steps consume (input) or produce (output) a specific variable.
        """
        print(f"=== Usage Report for: '{search_term}' ===")

        producers = []
        consumers = []

        # Leverage our recursive flattener to get every atomic step
        for step_name, step in self._get_flattened_steps():
            # Check Inputs (Consumed)
            for in_var in step.inputs:
                # Strip type hints for a clean comparison
                base_var = in_var.split(":")[0].strip()
                if search_term in base_var:
                    consumers.append((step_name, in_var))

            # Check Outputs (Produced)
            for out_var in step.outputs:
                base_var = out_var.split(":")[0].strip()
                if search_term in base_var:
                    producers.append((step_name, out_var))

        if not producers and not consumers:
            print("  (No usage found in any step)")
            return

        print("\nProduced by (Outputs):")
        if not producers:
            print("  (None)")
        else:
            for step_name, var in producers:
                # We print the raw 'var' to show if it had a type hint or bracket attached
                print(f"  - [{step_name}] -> {var}")

        print("\nConsumed by (Inputs):")
        if not consumers:
            print("  (None)")
        else:
            for step_name, var in consumers:
                print(f"  - [{step_name}] <- {var}")

    @property
    def details(self) -> str:
        steps = getattr(self, "steps", None)
        if not steps:
            return f"{self.name} (Empty Process)"

        step_tags = []
        step_func_names = []

        for step in steps:
            # Handle tracer proxies safely
            name = step.name
            if not isinstance(name, str) and hasattr(name, "value"):
                name = name.value
            step_tags.append(str(name))

            # Safely get the name whether it's a function or a class instance
            if isinstance(step, Process):
                step_func_names.append(f"<Process>: {len(step.steps)} Step(s)")
            elif isinstance(step, ProcessStep):
                func = step.function
                name = getattr(func, "__name__", func.__class__.__name__)
                step_func_names.append(name)

        # Handle edge case where process has steps but they have empty names
        max_tag_length = max([len(t) for t in step_tags]) if step_tags else 0

        process_str = self.name
        for idx in range(len(step_tags)):
            process_str += f"\n\t{idx + 1:>2}) {step_tags[idx]:<{max_tag_length}} : {step_func_names[idx]}"

        return process_str


# ----------------------------------------------------------------------------------------------------------------------
#  Legacy Optimizer Interface
# ----------------------------------------------------------------------------------------------------------------------


class OptimizerInterface:
    """Interface with legacy optimizers to separate value and gradient function for FlowTangent Processes."""

    def __init__(
        self,
        process: Process,
        base_state: State,
        base_system: System,
        base_settings: Settings,
        grad_map: JacobianMap,
        objective_path: TreePath,
        **kwargs,
    ):

        self.process = process

        self.base_state = base_state
        self.base_system = base_system
        self.base_settings = base_settings

        self.grad_map = grad_map
        self.objective_path = objective_path

        self.last_x = None
        self.last_val = None
        self.last_jac = None

        # Store additonal optimizer-specific settings
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _update_cache(self, x):
        if self.last_x is None or not np.allclose(x, self.last_x):
            state, system, settings = self.grad_map.update_inputs(
                x, self.base_state, self.base_system, self.base_settings
            )
            f_st, f_sys, f_setts, jac = self.process.run(state, system, settings)
            token = dict(state=f_st, system=f_sys, settings=f_setts)

            self.last_val = get_target(token, self.objective_path)
            self.last_jac = np.array(jac)
            self.last_x = x

    def fun(self, x):
        self._update_cache(x)
        return self.last_val

    def jac(self, x):
        self._update_cache(x)
        return self.last_jac
