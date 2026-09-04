# Trace/Framework/Process.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Jul 2024, Trace Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Generator, Optional, Self, Tuple, TypeAlias, Sequence, Literal, overload
if TYPE_CHECKING:
    from .settings import JacobianMap

import os
import re
import math
import time
import inspect
import warnings

from collections import Counter
from dataclasses import replace
from datetime import datetime

import equinox as eqx

# package imports
import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np  # Used only for OptimizerInterface class w/ legacy optimizers

# Trace imports
from .state import State
from .systems import System
from .settings import Settings
from ..utils import MERMAID_STYLES, DataPath, init_field, compute_tree_delta, null_step, get_target


# ----------------------------------------------------------------------------------------------------------------------
#  ProcessStep
# ----------------------------------------------------------------------------------------------------------------------

TraceFunction: TypeAlias = Callable[[State, System, Settings], Tuple[State, System, Settings]]


class ProcessStep(eqx.Module):
    function: TraceFunction = init_field(null_step, static=True, as_value=True)
    tag: str = init_field("Process Step", static=True)

    _state_delta: Optional[State] = init_field(None)
    _system_delta: Optional[System] = init_field(None)
    _settings_delta: Optional[Settings] = init_field(None)

    def __init__(
        self,
        tag: str = "Process Step",
        function: TraceFunction | ProcessStep = null_step,
        _state_delta: Optional[State] = None,
        _system_delta: Optional[System] = None,
        _settings_delta: Optional[Settings] = None,
    ):
        
        self.function = function
        self.tag = tag
        self._state_delta = _state_delta
        self._system_delta = _system_delta
        self._settings_delta = _settings_delta

    @classmethod
    def from_function(cls, step: Any) -> ProcessStep:
        if isinstance(step, ProcessStep):
            return step
        elif callable(step):
            step_name = getattr(step, "__name__", "Unnamed TraceFunction")
            sig = inspect.signature(step)
            if len(sig.parameters) != 3: raise ValueError(
                f"Process functions must take and return (State, System, Settings). "
                f"Found function '{step_name}' with signature '{sig}'.")
            return cls(tag=step_name, function=step)
        else:
            raise ValueError(f"Cannot create a ProcessStep from instance of '{type(step)}'.")
        

    def _profile_complexity(self, state: State, system: System, settings: Settings, top_n=5):
        try:
            assert(isinstance(self.function, Callable))
            jaxpr_obj = jax.make_jaxpr(self.function)(state, system, settings)
        except Exception as e:
            return f" - {self.tag} | Could not trace ({e})"

        source_counts = Counter()
        
        # 1. Define everything we want the profiler to IGNORE
        exclude_strings = [
            "jax/", "jax\\", "equinox", "jaxtyping", # Core internals
            "residual.py",                          # Orchestrator
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
                    is_wrapper_func = func_name in [
                        "make_node_function",
                        "transmit",
                        "net_transmit"
                    ]
                    
                    if file_name != "" and not is_ignored and not is_wrapper_func:
                        line_num = getattr(frame, "line_num", None) or getattr(frame, "lineno", "?")
                        short_file = file_name.split('/')[-1].split('\\')[-1]
                        user_location = f"{func_name} ({short_file}:{line_num})"
                        break 
                
                source_counts[user_location] += 1
            else:
                source_counts["Unknown Source"] += 1

        total_ops = len(jaxpr_obj.jaxpr.eqns)
        report = [f" - {self.tag} | Total Ops: {total_ops}"]
        for loc, count in source_counts.most_common(top_n):
            pct = (count / total_ops) * 100
            report.append(f" - - {count:4d} ops ({pct:4.1f}%) : {loc}")
            
        return "\n".join(report)
    
    def __call__(self, state: State, system: System, settings: Settings):
        if settings._DEV_MODE and settings.verbose:
            print(self._profile_complexity(state, system, settings))
        if not settings._DEV_MODE and settings.DEBUG_MODE:
            print(f" - {self.tag}")
        # Default calling behavior, assumes function is callable.
        # String overwrite only for steps with __call__ override
        return self.function(state, system, settings)  # type: ignore

    def run(self, state, system, settings):
        return self(state, system, settings)

    def _run_with_history(self, state, system, settings):
        return *self(state, system, settings), None

    def __repr__(self):
        return self.tag

    @property
    def inputs(self) -> set:
        return getattr(self.function, "_inputs", set())

    @property
    def outputs(self) -> set:
        return getattr(self.function, "_outputs", set())

# ----------------------------------------------------------------------------------------------------------------------
#  Process Class
# ----------------------------------------------------------------------------------------------------------------------

def array_barrier(state:State, system:System, settings:Settings):

    """
    Forces every numerical leaf of state (at least 2d) and system (at least 1d) to become JAX arrays.
    This both standardizes the shape of the arrays and ensures they don't share memory
    (e.g. Python may cache every instance of 0.0 to be the same memory address) so that when
    they're partitioned based on memory ID for tracing, there's no collisions.
    """

    def _to_array(leaf, ndim: int = 1):
        # 1. Check if it's a raw scalar, a list/tuple of scalars, OR already an array
        is_scalar = isinstance(leaf, (float, int, complex))
        is_iterable = isinstance(leaf, (list, tuple)) and all(isinstance(i, (float, int, complex)) for i in leaf)
        is_array = isinstance(leaf, (jax.Array, np.ndarray))

        if is_scalar or is_iterable or is_array:
            # 2. Convert to JAX array (jnp.asarray is a no-op if it's already a JAX array)
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

    tag: str = init_field("Process", static=True)
    steps: tuple[ProcessStep, ...] = ()
    
    initialize: TraceFunction = init_field(null_step, static=True)
    initial_step: int = init_field(0, static=True)

    _initial_state: Optional[State] = init_field(None)
    _initial_system: Optional[System] = init_field(None)
    _initial_settings: Optional[Settings] = init_field(None)

    _val_and_jac_fn: Optional[Callable] = init_field(None, static=True)
    _cached_grad_map: Optional[JacobianMap] = init_field(None, static=True)
    _filter_map: dict = init_field(lambda _:{
            "energy": r"state\.energy\.nodes\.\[*\]."
        }, static=True)

    def __init__(
        self,
        steps: Sequence[ProcessStep | TraceFunction] = (),
        tag: str = "Process",
        initialize: TraceFunction = null_step,
        initial_step: int = 0,
        _initial_state: Optional[State] = None,
        _initial_system: Optional[System] = None,
        _initial_settings: Optional[Settings] = None,
        _val_and_jac_fn: Optional[Callable] = None,
        _cached_grad_map: Optional[JacobianMap] = None,
        _filter_map: Optional[dict] = None,
    ):
        # Initialize the parent ProcessStep
        super().__init__(
            function=null_step, 
            tag=tag,
            _state_delta=None,
            _system_delta=None,
            _settings_delta=None,
        )

        # Standard field assignments
        self.tag = tag
        self.initialize = initialize
        self.initial_step = initial_step
        self._initial_state = _initial_state
        self._initial_system = _initial_system
        self._initial_settings = _initial_settings
        self._val_and_jac_fn = _val_and_jac_fn
        self._cached_grad_map = _cached_grad_map
        
        # Handle mutable dictionary default safely
        self._filter_map = _filter_map if _filter_map is not None else {
            "energy": r"state\.energy\.nodes\.\[*\]."
        }
                
        self.steps = tuple(ProcessStep.from_function(step) for step in steps)

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
            step_tag = step.tag
            if not isinstance(step_tag, str) and hasattr(step_tag, "value"):
                step_tag = step_tag.value

            if isinstance(step_tag, str):
                formatted_tag = step_tag.replace(" ", "_").lower()
                if key == formatted_tag:
                    return step

        raise AttributeError(f"{self.__class__.__name__}: {self.tag} has no attribute '{key}'")

    def __call__(self, state: State, system: System, settings: Settings) -> tuple[State, System, Settings]:
        if settings.DEBUG_MODE or settings._DEV_MODE:

            start_time = datetime.fromtimestamp(time.time()).strftime(settings.logging.date_format)
            print(f"Beginning Process: '{self.tag}' | {start_time}") 
        
        if settings.numerical.calculate_jacobian:
            jac_map = settings.numerical.jacobian_map
            
            if jac_map is not None:
                # Returns the two distinct flat arrays
                flat_st, flat_sys = jac_map.flatten_inputs(state, system)
                val_and_jac_fn = self._build_value_and_jacobian(jac_map)

                jacobian_matrix, final_st, final_sys, final_setts = val_and_jac_fn(
                    flat_st, flat_sys, state, system, settings
                )
                
                final_st = eqx.tree_at(lambda s: s.process_jacobian, final_st, jacobian_matrix)
                return final_st, final_sys, final_setts
            else: warnings.warn(f"Process '{self.tag}' Jacobian called with no Jacbian Map set. Jacobian will not be calculated.")

        # Standard Execution Path
        for step in self.steps[self.initial_step :]:
            state, system, settings = step(state, system, settings)

        if settings.DEBUG_MODE or settings._DEV_MODE:
            end_time = datetime.fromtimestamp(time.time()).strftime(settings.logging.date_format)
            print(f"Process '{self.tag}' Complete. | {end_time}") 

        return state, system, settings

    def _run_with_raw_history(self, state, system, settings):
        if settings.DEBUG_MODE or settings._DEV_MODE:
            print(f"Beginning Process: '{self.tag}'")
        history = [(state, system, settings)]

        for step in self.steps[self.initial_step :]:
            state, system, settings = step(state, system, settings)
            history.append((state, system, settings))

        return state, system, settings, tuple(history)

    def _build_value_and_jacobian(self, grad_map: JacobianMap):
        def objective_fn(flat_st, flat_sys, base_state, base_system, base_settings):
            st, sys = grad_map.update_inputs(flat_st, flat_sys, base_state, base_system)

            # Prevent recursion by temporarily disabling the Jacobian flag
            inner_num = replace(base_settings.numerical, calculate_jacobian=False)
            inner_setts = replace(base_settings, numerical=inner_num)

            f_st, f_sys, f_setts = self(st, sys, inner_setts)
            out_array = grad_map.flatten_outputs(f_st, f_sys, f_setts)

            # Restore modified setting
            restored_num = replace(f_setts.numerical, calculate_jacobian=True)
            f_setts = replace(f_setts, numerical=restored_num)

            return out_array, (f_st, f_sys, f_setts)

        def batched_jacrev_fn(flat_st, flat_sys, base_state, base_system, base_settings):
            out_array, vjp_fn, aux = jax.vjp(
                objective_fn, flat_st, flat_sys, base_state, base_system, base_settings, has_aux=True
            )
            
            is_coupled_time = getattr(base_settings.numerical, 'coupled_time_jacobian', False)
            
            # Determine shapes based on your strict (B, T, F) or (T, F) rules
            ndim = out_array.ndim
            N_o = out_array.shape[-1]
            has_B = (ndim == 3)
            
            B = out_array.shape[0] if has_B else 1
            T = out_array.shape[1] if has_B else out_array.shape[0]

            if not is_coupled_time:
                # =========================================================
                # PATH A: FAST BLOCK-DIAGONAL (Independent Time Steps)
                # Cost: O(N_o) VJP passes
                # =========================================================
                basis_st = jnp.eye(N_o).reshape(N_o, 1, 1, N_o) if has_B else jnp.eye(N_o).reshape(N_o, 1, N_o)
                basis_st = jnp.broadcast_to(basis_st, (N_o, B, T, N_o) if has_B else (N_o, T, N_o))
                
                jac_tuple = jax.vmap(vjp_fn)(basis_st)
                
                jac_st = jnp.moveaxis(jac_tuple[0], 0, -2) # -> (B, T, N_o, N_st) or (T, N_o, N_st)
                
                if flat_sys.size > 0:
                    # System is usually dense across the batch
                    B_total = B * T
                    basis_sys = jnp.eye(B_total * N_o).reshape(B_total * N_o, B, T, N_o) if has_B else jnp.eye(B_total * N_o).reshape(B_total * N_o, T, N_o)
                    jac_sys_tuple = jax.vmap(vjp_fn)(basis_sys)
                    
                    jac_sys = jac_sys_tuple[1].reshape(B, T, N_o, -1) if has_B else jac_sys_tuple[1].reshape(T, N_o, -1)
                    batched_jacobian = jnp.concatenate([jac_st, jac_sys], axis=-1)
                else:
                    batched_jacobian = jac_st

            else:
                # =========================================================
                # PATH B: DENSE TEMPORAL (Optimal Control)
                # Cost: O(T * N_o) VJP passes
                # =========================================================
                # We map over (T * N_o) to capture cross-time sensitivities
                T_No = T * N_o
                
                if has_B:
                    # Independent across batches, fully dense across time
                    basis_st = jnp.eye(T_No).reshape(T_No, 1, T, N_o)
                    basis_st = jnp.broadcast_to(basis_st, (T_No, B, T, N_o))
                    
                    jac_tuple = jax.vmap(vjp_fn)(basis_st) # Output: (T*N_o, B, T, N_st)
                    jac_st = jnp.moveaxis(jac_tuple[0], 1, 0) # -> (B, T*N_o, T, N_st)
                    jac_st = jac_st.reshape(B, T, N_o, T, -1) # -> (B, T_out, N_o, T_in, N_st)
                else:
                    basis_st = jnp.eye(T_No).reshape(T_No, T, N_o)
                    jac_tuple = jax.vmap(vjp_fn)(basis_st) # Output: (T*N_o, T, N_st)
                    jac_st = jac_tuple[0].reshape(T, N_o, T, -1) # -> (T_out, N_o, T_in, N_st)

                if flat_sys.size > 0:
                    # System is dense across everything
                    B_total = B * T if has_B else T
                    basis_sys = jnp.eye(B_total * N_o).reshape(B_total * N_o, B, T, N_o) if has_B else jnp.eye(B_total * N_o).reshape(B_total * N_o, T, N_o)
                    jac_sys_tuple = jax.vmap(vjp_fn)(basis_sys)
                    
                    # You may need to adapt this reshape depending on if System variables are constant over time or time-varying
                    jac_sys = jac_sys_tuple[1].reshape(B, T, N_o, -1) if has_B else jac_sys_tuple[1].reshape(T, N_o, -1)
                    # Note: Concatenating State (5D) with System (4D) requires flattening the time dimensions for the solver
                    batched_jacobian = (jac_st, jac_sys) # Returned as a tuple for optimal control solvers
                else:
                    batched_jacobian = jac_st

            return batched_jacobian, aux[0], aux[1], aux[2]


        return batched_jacrev_fn

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
    
    def run(
        self,
        state: State, system: System, settings: Settings, *,
        initialize: bool = True, track_history: bool = False
    ):

        state, system, settings = array_barrier(state, system, settings)

        if initialize:
            state, system, settings = self.initialize(state, system, settings)

        # Direct call if not tracking history
        if not track_history:
            return self(state, system, settings)

        
        f_st, f_sys, f_setts, raw_hist = self._run_with_raw_history(state, system, settings)

        logged_process = None
        logged_steps = []

        for i, step in enumerate(self.steps[self.initial_step :]):
            logged_step = eqx.tree_at(
                lambda s: (s.state_delta, s.system_delta, s.settings_delta),
                step,
                (
                    compute_tree_delta(raw_hist[i + 1][0], raw_hist[i][0]),
                    compute_tree_delta(raw_hist[i + 1][1], raw_hist[i][1]),
                    compute_tree_delta(raw_hist[i + 1][2], raw_hist[i][2]),
                ),
            )
            logged_steps.append(logged_step)

        logged_process = eqx.tree_at(
            lambda p: (
                p.steps,
                p.initial_state,
                p.initial_system,
                p.initial_settings,
                p.state_delta,
                p.system_delta,
                p.settings_delta,
            ),
            self,
            (
                tuple(logged_steps),
                state,
                system,
                settings,
                compute_tree_delta(f_st, state),
                compute_tree_delta(f_sys, system),
                compute_tree_delta(f_setts, settings),
            ),
            is_leaf=lambda x: x is None,
        )

        return f_st, f_sys, f_setts, logged_process

    def append(self, step: ProcessStep | Self):
        new_steps = self.steps + (step,)
        return eqx.tree_at(lambda p: p.steps, self, new_steps)

    def count(self, step: ProcessStep):
        return self.steps.count(step)

    def _index_tag(self, tag: str):
        tags = [step.tag for step in self.steps]
        if tag not in tags:
            tags = [t.replace(" ", "_").lower() for t in tags]
        if tag not in tags:
            raise AttributeError(f"Unable to locate step {tag} in steps of Process {self.tag}.")
        index = tags.index(tag)

        return index

    def _index_function(self, function: Callable):
        functions = [step.function for step in self.steps]
        index = functions.index(function)

        return index

    def index(self, value: str | Callable | ProcessStep | Self):
        if isinstance(value, str):
            return self._index_tag(value)
        elif isinstance(value, Callable):
            return self._index_function(value)
        elif isinstance(value, ProcessStep):
            return self.steps.index(value)

        else:
            raise ValueError("Trace processes can only be indexed by name, function, or ProcessStep object.")

    def insert(self, step: ProcessStep, index: int):
        new_steps = self.steps[:index] + (step,) + self.steps[index:]
        return eqx.tree_at(lambda c: c.steps, self, new_steps)

    def pop(self, index: int):
        new_steps = self.steps[:index] + self.steps[index + 1 :]
        return eqx.tree_at(lambda p: p.steps, self, new_steps)

    def _remove_tag(self, tag: str):
        return self.pop(self._index_tag(tag))

    def _remove_function(self, function: Callable):
        return self.pop(self._index_function(function))

    def remove(self, value: str | Callable | ProcessStep | Self):
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
            node_name = step.tag

            # Check if this step is itself a nested Process containing other steps
            if hasattr(step, "steps") and step.steps is not None:
                # Recursively yield its internal steps
                yield from step._get_flattened_steps(prefix=f"{prefix}{node_name}.")
            else:
                yield node_name, step

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
            step_iterator = ((f"{i}_{step.tag}", step) for i, step in enumerate(self.steps))

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
        exclude: list[str] = None,
        save_path: str = None,
        style: str = "modern",
    ) -> str:
        """
        Generates a Mermaid.js flowchart string from the Process DAG.

        Args:
            recursive: Whether to flatten nested Processes.
            show_edges: Whether to label the edges with the variables passed between steps.
            layout: "LR" (Left-to-Right) or "TD" (Top-Down).
            exclude: List of predefined domains to hide from the edges (e.g., ['energy']).
        """
        # 1. Setup the exclusion filters using the shared class attribute
        if exclude is None:
            exclude = ["energy"]

        compiled_patterns = [re.compile(self._filter_map[k]) for k in exclude if k in self._filter_map]

        def is_filtered(var_name: str) -> bool:
            return any(pat.search(var_name) for pat in compiled_patterns)

        # 2. Grab the graph
        G = self.graph(recursive=recursive)
        mermaid_lines = []

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
                display_label = step_obj.tag if step_obj else str(node_name)
                mermaid_lines.append(f"    {safe_id}[{display_label}]")

        # 4. Build edges and apply filters to the variable lists
        for u, v, data in G.edges(data=True):
            raw_vars = data.get("variables", [])

            # Apply the filter to strip out unwanted phantom paths
            vars_list = [var for var in raw_vars if not is_filtered(var)]

            if show_edges and vars_list:
                if len(vars_list) > 4:
                    label = f"{len(vars_list)} variables"
                else:
                    clean_vars = [var.split(".")[-1] for var in vars_list]
                    label = "<br>".join(clean_vars)

                # Sanitize any stray double quotes to single quotes
                label = label.replace('"', "'")

                # Wrap the final label in double quotes for Mermaid's parser
                mermaid_lines.append(f'    {node_id_map[u]} -->|"{label}"| {node_id_map[v]}')
            else:
                mermaid_lines.append(f"    {node_id_map[u]} --> {node_id_map[v]}")

        mermaid_str = "\n".join(mermaid_lines)

        if save_path:
            # Ensure the target directory exists
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

            with open(save_path, "w", encoding="utf-8") as f:
                # If it's a markdown file, wrap it in the mermaid code block
                if save_path.lower().endswith(".md"):
                    f.write("```mermaid\n")
                    f.write(mermaid_str)
                    f.write("\n```\n")
                else:
                    # For .mmd or .txt, just write the raw string
                    f.write(mermaid_str)

        return mermaid_str

    def print_io_tree(self, exclude: list[str] = None):
        """
        Extracts the inputs and outputs of the Process and prints them
        in a hierarchical, human-readable ASCII tree structure.
        Handles iterators ([Item]), dictionaries (['key']), and pseudo-types (: Type).

        Args:
            exclude: List of predefined domains to hide from the I/O tree (e.g., ['energy']).
        """

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

        def build_tree_and_metadata(paths: set[str]) -> tuple[dict, dict]:
            """
            Converts strings into a nested dict and extracts type hints.
            Returns the structural tree and a dictionary of type hints keyed by path tuples.
            """
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

                # 3. Store the hint keyed by the exact structural path
                if hint:
                    type_hints[parts] = hint

                # 4. Build the structural tree
                current_level = tree
                for part in parts:
                    current_level = current_level.setdefault(part, {})

            return tree, type_hints

        def display_tree(tree: dict, type_hints: dict, depth: int = 0, current_parts: tuple = ()):
            """Recursively prints the nested dictionary with ASCII indentation."""
            for key in sorted(tree.keys()):
                node_parts = current_parts + (key,)
                display_name = str(key)

                # Apply Type Hint if one was registered for this exact node
                if node_parts in type_hints:
                    display_name += f": {type_hints[node_parts]}"

                # Look ahead: if a child is a dictionary key (has quotes inside brackets), flag this node
                # Iterators like [Wing] have no quotes, so they won't trigger this.
                has_dict_children = any(str(k).startswith("['") or str(k).startswith('["') for k in tree[key].keys())
                if has_dict_children:
                    display_name += ": {dict}"

                # Print with formatting
                if depth == 0:
                    print(display_name)
                else:
                    prefix = "|" + "-" * (2 * depth)
                    print(f"{prefix}{display_name}")

                # Recurse
                display_tree(tree[key], type_hints, depth + 1, node_parts)

        print("=== Process Inputs ===")
        if not display_inputs:
            print("  (None)")
        else:
            in_tree, in_hints = build_tree_and_metadata(display_inputs)
            display_tree(in_tree, in_hints)

        print("\n=== Process Outputs ===")
        if not display_outputs:
            print("  (None)")
        else:
            out_tree, out_hints = build_tree_and_metadata(display_outputs)
            display_tree(out_tree, out_hints)

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
            return f"{self.tag} (Empty Process)"

        step_tags = []
        step_func_names = []

        for step in steps:
            # Handle tracer proxies safely
            tag = step.tag
            if not isinstance(tag, str) and hasattr(tag, "value"):
                tag = tag.value
            step_tags.append(str(tag))

            # Safely get the name whether it's a function or a class instance
            if isinstance(step, Process):
                step_func_names.append(f"<Process>: {len(step.steps)} Step(s)")
            elif isinstance(step, ProcessStep):
                func = step.function
                name = getattr(func, "__name__", func.__class__.__name__)
                step_func_names.append(name)

        # Handle edge case where process has steps but they have empty tags
        max_tag_length = max([len(t) for t in step_tags]) if step_tags else 0

        process_str = self.tag
        for idx in range(len(step_tags)):
            process_str += f"\n\t{idx + 1:>2}) {step_tags[idx]:<{max_tag_length}} : {step_func_names[idx]}"

        return process_str


# ----------------------------------------------------------------------------------------------------------------------
#  Legacy Optimizer Interface
# ----------------------------------------------------------------------------------------------------------------------


class OptimizerInterface:
    """Interface with legacy optimizers to separate value and gradient function for Trace Processes."""

    def __init__(
        self,
        process: Process,
        base_state: State,
        base_system: System,
        base_settings: Settings,
        grad_map: JacobianMap,
        objective_path: DataPath,
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
