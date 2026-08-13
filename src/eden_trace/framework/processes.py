# Trace/Framework/Process.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Jul 2024, Trace Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Generator, Optional, Self, Tuple
if TYPE_CHECKING:
    from eden_trace.framework import Settings, State, System

import os
import re
import time
from datetime import datetime

from collections import Counter

import equinox as eqx

# package imports
import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np  # Used only for OptimizerInterface class w/ legacy optimizers

import eden_trace.utils as tu

# Trace imports
from eden_trace.utils import MERMAID_STYLES, DataPath, Token, init_field

# ----------------------------------------------------------------------------------------------------------------------
#  ProcessStep
# ----------------------------------------------------------------------------------------------------------------------


def null_step(*args):
    return args


class ProcessStep(eqx.Module):
    function: Callable | str = init_field(null_step, static=True)
    tag: str = init_field("Process Step", static=True)

    state_delta: State | None = None
    system_delta: System | None = None
    settings_delta: Settings | None = None

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
#  Gradient Map
# ----------------------------------------------------------------------------------------------------------------------


class GradientMap:
    def __init__(
        self,
        state_inputs: tuple = (),
        state_outputs: tuple = (),
        system_inputs: tuple = (),
        system_outputs: tuple = (),
        settings_inputs: tuple = (),
        settings_outputs: tuple = (),
    ):

        # Sanitize inputs/oututs to PathTuples
        self.state_inputs = tuple(DataPath(p) for p in state_inputs)
        self.system_inputs = tuple(DataPath(p) for p in system_inputs)
        self.settings_inputs = tuple(DataPath(p) for p in settings_inputs)

        self.state_outputs = tuple(DataPath(p) for p in state_outputs)
        self.system_outputs = tuple(DataPath(p) for p in system_outputs)
        self.settings_outputs = tuple(DataPath(p) for p in settings_outputs)

        # Count inputs
        self._n_st = len(self.state_inputs)
        self._n_sys = len(self.system_inputs)
        self._n_setts = len(self.settings_inputs)

        self.unravel_function = null_step  # Default, updates when inputs are grabbed

    def flatten_inputs(
        self,
        base_state,
        base_system,
        base_settings,
    ):
        import numpy as np  # Standard numpy for calculating split indices during tracing

        inputs = []
        if len(self.state_inputs) > 0:
            inputs.extend(tu.get_all_targets(base_state, self.state_inputs))
        if len(self.system_inputs) > 0:
            inputs.extend(tu.get_all_targets(base_system, self.system_inputs))
        if len(self.settings_inputs) > 0:
            inputs.extend(tu.get_all_targets(base_settings, self.settings_inputs))

        # 1. Dynamically read the batch size from the first array
        B = inputs[0].shape[0]

        # 2. Record the inner shapes and sizes for unraveling later
        shapes = [inp.shape[1:] for inp in inputs]
        sizes = [int(np.prod(s)) if s else 1 for s in shapes]

        # 3. Reshape all arrays to strictly (Batch, Features) and concatenate
        # E.g., a (4,) array becomes (4, 1). A (4, 3, 3) matrix becomes (4, 9).
        flat_inputs = [inp.reshape(B, -1) for inp in inputs]
        flat_input_array = jnp.concatenate(flat_inputs, axis=-1)

        # 4. Create a custom unravel function that acts on axis=-1
        def unravel_function(flat_array):
            # flat_array shape is (Batch, Total_N_i)
            split_indices = np.cumsum(sizes)[:-1]
            splits = jnp.split(flat_array, split_indices, axis=-1)

            # Restore to original shapes, preserving the leading Batch dimension
            return [s.reshape((B,) + shape) for s, shape in zip(splits, shapes)]

        self.unravel_function = unravel_function
        return flat_input_array

    def update_inputs(
        self,
        input_array,
        base_state: Optional[State] = None,
        base_system: Optional[System] = None,
        base_settings: Optional[Settings] = None,
    ):

        st, sys, setts = base_state, base_system, base_settings

        reshaped_inputs = self.unravel_function(input_array)

        def stitch_parents(base_tree, paths, new_slices):
            parents = tu.get_all_parents(base_tree, paths)
            updated_parents = []

            for parent, new_val, path in zip(parents, new_slices, paths):
                if path.slice_obj != slice(None):
                    # Stitch the updated slice into the original parent array
                    updated_parents.append(parent.at[path.slice_obj].set(new_val))
                else:
                    # No slice, just use the whole new value
                    updated_parents.append(new_val)

            return tuple(updated_parents)

        # Inject flat array of inputs into the PyTrees
        # (Wrapped in lambdas, and replace inputs cast to tuples, slices stiched back into parents)
        if self._n_st > 0:
            st_slices = reshaped_inputs[: self._n_st]
            updated_st_parents = stitch_parents(st, self.state_inputs, st_slices)

            st = eqx.tree_at(lambda t: tu.get_all_parents(t, self.state_inputs), st, updated_st_parents)

        if self._n_sys > 0:
            sys_slices = reshaped_inputs[self._n_st : self._n_st + self._n_sys]
            updated_sys_parents = stitch_parents(sys, self.system_inputs, sys_slices)

            sys = eqx.tree_at(lambda t: tu.get_all_parents(t, self.system_inputs), sys, updated_sys_parents)

        if self._n_setts > 0:
            setts_slices = reshaped_inputs[self._n_st + self._n_sys :]
            updated_setts_parents = stitch_parents(setts, self.settings_inputs, setts_slices)

            setts = eqx.tree_at(lambda t: tu.get_all_parents(t, self.settings_inputs), setts, updated_setts_parents)

        return st, sys, setts

    def flatten_outputs(self, f_st, f_sys, f_setts):
        outputs = []
        if self.state_outputs:
            outputs.extend(tu.get_all_targets(f_st, self.state_outputs))
        if self.system_outputs:
            outputs.extend(tu.get_all_targets(f_sys, self.system_outputs))
        if self.settings_outputs:
            outputs.extend(tu.get_all_targets(f_setts, self.settings_outputs))

        # Preserve Batch, flatten the inner features
        B = outputs[0].shape[0]
        out_array = jnp.concatenate([out.reshape(B, -1) for out in outputs], axis=-1)

        return out_array


# ----------------------------------------------------------------------------------------------------------------------
#  Process Class
# ----------------------------------------------------------------------------------------------------------------------


class Process(ProcessStep):
    tag: str = init_field("Process", static=True)

    steps: tuple[ProcessStep, ...] = init_field(tuple)
    initialize: Optional[Process] = init_field(None)

    initial_step: int = init_field(0, static=True)

    initial_state: Optional[State] = None
    initial_system: Optional[System] = None
    initial_settings: Optional[Settings] = None

    _val_and_jac_fn: Optional[Callable] = init_field(None, static=True)
    _cached_grad_map: Optional[GradientMap] = init_field(None, static=True)

    _filter_map: dict = init_field(
        lambda: {
            "energy": r"state\.energy\.nodes\.\[*\].outputs",
        },
        static=True,
    )

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

    def __call__(self, state:State, system:System, settings:Settings) -> tuple[State, System, Settings]:
        if settings.DEBUG_MODE or settings._DEV_MODE:
            start_time = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S")
            print(f"Beginning Process: '{self.tag}' | {start_time}")

        for step in self.steps[self.initial_step :]:
            state, system, settings = step(state, system, settings)

        if settings.DEBUG_MODE or settings._DEV_MODE:
            end_time = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S")
            print(f"Process '{self.tag}' Complete.  | {end_time}")
        return state, system, settings

    def _run_with_raw_history(self, state, system, settings):
        if settings.DEBUG_MODE or settings._DEV_MODE:
            print(f"Beginning Process: '{self.tag}'")
        history = [(state, system, settings)]

        for step in self.steps[self.initial_step :]:
            state, system, settings = step(state, system, settings)
            history.append((state, system, settings))

        return state, system, settings, tuple(history)

    def _build_value_and_jacobian(self, grad_map: GradientMap, track_history: bool):
        """Compiles closed-form Jacobian for specified input and output paths."""

        def objective_fn(input_array, base_state, base_system, base_settings):
            st, sys, setts = grad_map.update_inputs(input_array, base_state, base_system, base_settings)

            if track_history:
                f_st, f_sys, f_setts, raw_hist = self._run_with_raw_history(st, sys, setts)
                aux = (f_st, f_sys, f_setts, raw_hist)
            else:
                f_st, f_sys, f_setts = self(st, sys, setts)
                aux = (f_st, f_sys, f_setts, None)

            out_array = grad_map.flatten_outputs(f_st, f_sys, f_setts)
            return out_array, aux

        def batched_jacrev_fn(input_array, base_state, base_system, base_settings):
            # 1. Forward pass + VJP function generation
            out_array, vjp_fn, aux = jax.vjp(
                objective_fn, input_array, base_state, base_system, base_settings, has_aux=True
            )
            B, N_o = out_array.shape

            # 2. Build batched cotangent basis: (N_o, B, N_o)
            basis = jnp.broadcast_to(jnp.eye(N_o)[:, None, :], (N_o, B, N_o))

            # 3. Pullback through vmap
            # vjp_fn receives (B, N_o) cotangents and outputs (B, N_i) gradients.
            # out_axes=(1, 0, 0, 0) stacks the target gradient natively to (B, N_o, N_i)
            jac_tuple = jax.vmap(vjp_fn, out_axes=(1, 0, 0, 0))(basis)

            # 4. Extract target gradient
            batched_jacobian = jac_tuple[0]

            return batched_jacobian, aux

        return batched_jacrev_fn

    @staticmethod
    def _sanitize_inputs(tree):
        """Cast all numeric leaves to 0D JAX scalars for gradient computations."""

        def _to_array(leaf):
            if isinstance(leaf, (float, int)) or (
                isinstance(leaf, list) and all(isinstance(i, (float, int)) for i in leaf)
            ):
                return jnp.array(leaf, dtype=jnp.float64)
            else:
                return leaf

        return jax.tree_util.tree_map(_to_array, tree)

    def run(
        self, state, system, settings, initialize=False, track_history: bool = False
    ) -> Tuple[State, System, Settings, Optional[jnp.ndarray], Optional[Self]]:

        # Sanitize inputs (map floats/ints to JAX arrays)
        state = self._sanitize_inputs(state)
        system = self._sanitize_inputs(system)
        settings = self._sanitize_inputs(settings)

        # Save original/intial state
        if self.initialize is not None and initialize:
            i_st, i_sys, i_setts = self.initialize(state, system, settings)
        else:
            i_st, i_sys, i_setts = state, system, settings

        # Prep for gradient calcuation/history tracking
        jacobian_matrix = None
        raw_hist = None

        # Grad map acts as flag to get gradients
        grad_map = settings.analysis.gradient_map
        if grad_map is not None:
            # Flatten inputs for Jacobian calculation
            flat_input_array = grad_map.flatten_inputs(i_st, i_sys, i_setts)

            # Build Value and Jacobian function only if it doesn't exist or the cached grad_map is outdated
            if self._val_and_jac_fn is None or self._cached_grad_map != grad_map:
                object.__setattr__(self, "_val_and_jac_fn", self._build_value_and_jacobian(grad_map, track_history))
                object.__setattr__(self, "_cached_grad_map", grad_map)

            jacobian_matrix, aux = self._val_and_jac_fn(flat_input_array, i_st, i_sys, i_setts)  # type: ignore
            f_st, f_sys, f_setts, raw_hist = aux

        else:
            if track_history:
                f_st, f_sys, f_setts, raw_hist = self._run_with_raw_history(i_st, i_sys, i_setts)
            else:
                f_st, f_sys, f_setts = self(i_st, i_sys, i_setts)

        logged_process = None
        if track_history and raw_hist is not None:
            logged_steps = []
            for i, step in enumerate(self.steps[self.initial_step :]):
                logged_step = eqx.tree_at(
                    lambda s: (s.state_delta, s.system_delta, s.settings_delta),
                    step,
                    (
                        tu.compute_tree_delta(raw_hist[i + 1][0], raw_hist[i][0]),
                        tu.compute_tree_delta(raw_hist[i + 1][1], raw_hist[i][1]),
                        tu.compute_tree_delta(raw_hist[i + 1][2], raw_hist[i][2]),
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
                    i_st,
                    i_sys,
                    i_setts,
                    tu.compute_tree_delta(f_st, i_st),
                    tu.compute_tree_delta(f_sys, i_sys),
                    tu.compute_tree_delta(f_setts, i_setts),
                ),
                is_leaf=lambda x: x is None,
            )

        # Always return final State, System, Settings, optionally return Jacobian matrix and logged Process
        out_vals = (f_st, f_sys, f_setts)
        if jacobian_matrix is not None:
            out_vals += (jacobian_matrix,)
        if logged_process is not None:
            out_vals += (logged_process,)

        return out_vals  # type: ignore

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
        grad_map: GradientMap,
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
            token = Token(state=f_st, system=f_sys, settings=f_setts)

            self.last_val = tu.get_target(token, self.objective_path)
            self.last_jac = np.array(jac)
            self.last_x = x

    def fun(self, x):
        self._update_cache(x)
        return self.last_val

    def jac(self, x):
        self._update_cache(x)
        return self.last_jac
