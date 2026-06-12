# RCAIDE/Framework/Process.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Jul 2024, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Callable, Self, Generator, Tuple
if TYPE_CHECKING:
    from RCAIDE.Framework import State, System, Settings

import logging
from pathlib import Path
from collections import defaultdict
from itertools import product

# package imports
import zarr
import jax
import jax.numpy as jnp
import equinox as eqx
import networkx as nx

import numpy as np  # Used only for OptimizerInterface class w/ legacy optimizers

from jax.flatten_util import ravel_pytree
from numcodecs import Blosc
from tqdm import trange

# RCAIDE imports
from RCAIDE.utils import Token, PathTuple, init_field
import RCAIDE.utils as ru
# ----------------------------------------------------------------------------------------------------------------------
#  ProcessStep
# ----------------------------------------------------------------------------------------------------------------------

def null_step(*args):
    return args

class ProcessStep(eqx.Module):

    function:       Callable | str     = init_field(null_step, static=True)
    tag:            str                = init_field("Process Step", static=True)

    state_delta:          State | None     = None
    system_delta:         System | None    = None
    settings_delta:       Settings | None  = None

    def __call__(self, state, system, settings):
        if settings.DEBUG_MODE: print(f"  Step: '{self.tag}'")
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
        state_inputs:       tuple = (),
        state_outputs:      tuple = (),
        system_inputs:      tuple = (),
        system_outputs:     tuple = (),
        settings_inputs:    tuple = (),
        settings_outputs:   tuple = (),
    ):

        # Sanitize inputs/oututs to PathTuples  
        self.state_inputs     = tuple(PathTuple(p) for p in state_inputs)
        self.system_inputs    = tuple(PathTuple(p) for p in system_inputs)
        self.settings_inputs  = tuple(PathTuple(p) for p in settings_inputs)
        
        self.state_outputs    = tuple(PathTuple(p) for p in state_outputs)
        self.system_outputs   = tuple(PathTuple(p) for p in system_outputs)
        self.settings_outputs = tuple(PathTuple(p) for p in settings_outputs)

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
            inputs.extend(ru.get_all_targets(base_state, self.state_inputs))
        if len(self.system_inputs) > 0:
            inputs.extend(ru.get_all_targets(base_system, self.system_inputs))
        if len(self.settings_inputs) > 0:
            inputs.extend(ru.get_all_targets(base_settings, self.settings_inputs))

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
            base_settings: Optional[Settings] =  None,
        ):

        st, sys, setts = base_state, base_system, base_settings

        reshaped_inputs = self.unravel_function(input_array)

        def stitch_parents(base_tree, paths, new_slices):
            parents = ru.get_all_parents(base_tree, paths)
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
            st_slices = reshaped_inputs[:self._n_st]
            updated_st_parents = stitch_parents(st, self.state_inputs, st_slices)
            
            st = eqx.tree_at(
                lambda t: ru.get_all_parents(t, self.state_inputs),
                st,
                updated_st_parents
            )

        if self._n_sys > 0:
            sys_slices = reshaped_inputs[self._n_st: self._n_st + self._n_sys]
            updated_sys_parents = stitch_parents(sys, self.system_inputs, sys_slices)
            
            sys = eqx.tree_at(
                lambda t: ru.get_all_parents(t, self.system_inputs), 
                sys,
                updated_sys_parents
            )

        if self._n_setts > 0:
            setts_slices = reshaped_inputs[self._n_st + self._n_sys:]
            updated_setts_parents = stitch_parents(setts, self.settings_inputs, setts_slices)
            
            setts = eqx.tree_at(
                lambda t: ru.get_all_parents(t, self.settings_inputs), 
                setts,
                updated_setts_parents
            )
            
        return st, sys, setts
    
    def flatten_outputs(self, f_st, f_sys, f_setts):
        outputs = []
        if self.state_outputs:
            outputs.extend(ru.get_all_targets(f_st, self.state_outputs))
        if self.system_outputs:
            outputs.extend(ru.get_all_targets(f_sys, self.system_outputs))
        if self.settings_outputs:
            outputs.extend(ru.get_all_targets(f_setts, self.settings_outputs))

        # Preserve Batch, flatten the inner features
        B = outputs[0].shape[0]
        out_array = jnp.concatenate([out.reshape(B, -1) for out in outputs], axis=-1)

        return out_array

# ----------------------------------------------------------------------------------------------------------------------
#  Process Class
# ----------------------------------------------------------------------------------------------------------------------

class Process(ProcessStep):

    tag:                str                     = init_field("Process", static=True)

    steps:              tuple[ProcessStep, ...] = init_field(tuple)

    initial_step:       int                     = init_field(0, static=True)

    initial_state:      Optional[State]     = None
    initial_system:     Optional[System]    = None
    initial_settings:   Optional[Settings]  = None

    _val_and_jac_fn:    Optional[Callable]      = init_field(None, static=True)
    _cached_grad_map:   Optional[GradientMap]   = init_field(None, static=True)

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
            if not isinstance(step_tag, str) and hasattr(step_tag, 'value'):
                step_tag = step_tag.value
            
            if isinstance(step_tag, str):
                formatted_tag = step_tag.replace(' ', '_').lower()
                if key == formatted_tag:
                    return step

        raise AttributeError(f"{self.__class__.__name__}: {self.tag} has no attribute '{key}'")

    def __call__(self, state, system, settings) -> tuple[State, System, Settings]:
        if settings.DEBUG_MODE: print(f"Beginning Process: '{self.tag}'")

        for step in self.steps[self.initial_step:]:
            state, system, settings = step(state, system, settings)

        if settings.DEBUG_MODE: print(f"Process '{self.tag}' Complete.")
        return state, system, settings

    def _run_with_raw_history(self, state, system, settings):
        if settings.DEBUG_MODE: print(f"Beginning Process: '{self.tag}'")
        history = [(state, system, settings)]

        for step in self.steps[self.initial_step:]:
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
                objective_fn,
                input_array, base_state, base_system, base_settings,
                has_aux=True
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
            if (
                isinstance(leaf, (float, int)) or
                (isinstance(leaf, list) and all(isinstance(i, (float, int)) for i in leaf))
            ):
                return jnp.array(leaf, dtype=jnp.float64)
            else:
                return leaf

        return jax.tree_util.tree_map(_to_array, tree)

    def run(self,
            state, system, settings,
            track_history: bool=False) -> Tuple[State, System, Settings, Optional[jnp.ndarray], Optional[Self]]:

        # Sanitize inputs (map floats/ints to JAX arrays)
        state = self._sanitize_inputs(state)
        system = self._sanitize_inputs(system)
        settings = self._sanitize_inputs(settings)

        # Save original/intial state
        initial_state = state
        initial_system = system
        initial_settings = settings

        # Prep for gradient calcuation/history tracking
        jacobian_matrix = None
        raw_hist = None

        # Grad map acts as flag to get gradients
        grad_map = settings.analysis.gradient_map
        if grad_map is not None:

            # Flatten inputs for Jacobian calculation
            flat_input_array = grad_map.flatten_inputs(state, system, settings)

            # Build Value and Jacobian function only if it doesn't exist or the cached grad_map is outdated
            if self._val_and_jac_fn is None or self._cached_grad_map != grad_map:

                object.__setattr__(self, '_val_and_jac_fn', self._build_value_and_jacobian(grad_map, track_history))
                object.__setattr__(self, '_cached_grad_map', grad_map)

            jacobian_matrix, aux = self._val_and_jac_fn(flat_input_array, state, system, settings)  # type: ignore
            f_st, f_sys, f_setts, raw_hist = aux

        else:
            if track_history:
                f_st, f_sys, f_setts, raw_hist = self._run_with_raw_history(state, system, settings)
            else:
                f_st, f_sys, f_setts = self(state, system, settings)

        logged_process = None
        if track_history and raw_hist is not None:
            logged_steps = []
            for i, step in enumerate(self.steps[self.initial_step:]):
                logged_step = eqx.tree_at(lambda s: (s.state_delta, s.system_delta, s.settings_delta), step,
                                          (ru.compute_tree_delta(raw_hist[i+1][0], raw_hist[i][0]),
                                           ru.compute_tree_delta(raw_hist[i+1][1], raw_hist[i][1]),
                                           ru.compute_tree_delta(raw_hist[i+1][2], raw_hist[i][2])
                                           ))
                logged_steps.append(logged_step)

            logged_process = eqx.tree_at(
                lambda p: (
                    p.steps,
                    p.initial_state, p.initial_system, p.initial_settings,
                    p.state_delta, p.system_delta, p.settings_delta
                ),
                self,
                (
                    tuple(logged_steps),
                    initial_state, initial_system, initial_settings,
                    ru.compute_tree_delta(state, initial_state),
                    ru.compute_tree_delta(system, initial_system),
                    ru.compute_tree_delta(settings, initial_settings)
                ),
                is_leaf=lambda x: x is None
            )

        # Always return final State, System, Settings, optionally return Jacobian matrix and logged Process
        out_vals = (f_st, f_sys, f_setts)
        if jacobian_matrix is not None:
            out_vals += (jacobian_matrix,)
        if logged_process is not None:
            out_vals += (logged_process,)

        return out_vals # type: ignore

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
            raise ValueError("RCAIDE processes can only be indexed by name, function, or ProcessStep object.")
    
    def insert(self, step: ProcessStep, index: int):
        new_steps = self.steps[:index] + (step,) + self.steps[index:]
        return eqx.tree_at(lambda c: c.steps, self, new_steps)

    def pop(self, index: int):
        new_steps = self.steps[:index] + self.steps[index + 1:]
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
        for i, step in enumerate(self.steps):
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
                        G.edges[producer_node, step_node]['variables'].append(in_var)
                    else:
                        G.add_edge(producer_node, step_node, variables=[in_var])
                else:
                    global_node = "Global Inputs"
                    if not G.has_node(global_node):
                        G.add_node(global_node)

                    if G.has_edge(global_node, step_node):
                        G.edges[global_node, step_node]['variables'].append(in_var)
                    else:
                        G.add_edge(global_node, step_node, variables=[in_var])

            # Resolve Outputs
            for out_var in step.outputs:
                latest_producers[out_var] = step_node

        return G

    def print_io_tree(self):
        """
        Extracts the inputs and outputs of the Process and prints them
        in a hierarchical, human-readable ASCII tree structure.
        Handles iterators ([Item]), dictionaries (['key']), and pseudo-types (: Type).
        """
        import re

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
        if not self.inputs:
            print("  (None)")
        else:
            in_tree, in_hints = build_tree_and_metadata(self.inputs)
            display_tree(in_tree, in_hints)

        print("\n=== Process Outputs ===")
        if not self.outputs:
            print("  (None)")
        else:
            out_tree, out_hints = build_tree_and_metadata(self.outputs)
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
                base_var = in_var.split(':')[0].strip()
                if search_term in base_var:
                    consumers.append((step_name, in_var))

            # Check Outputs (Produced)
            for out_var in step.outputs:
                base_var = out_var.split(':')[0].strip()
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
            if not isinstance(tag, str) and hasattr(tag, 'value'):
                tag = tag.value
            step_tags.append(str(tag))

            # Safely get the name whether it's a function or a class instance
            if isinstance(step, Process):
                step_func_names.append(f"<Process>: {len(step.steps)} Step(s)")
            elif isinstance(step, ProcessStep):
                func = step.function
                name = getattr(func, '__name__', func.__class__.__name__)
                step_func_names.append(name)

        # Handle edge case where process has steps but they have empty tags
        max_tag_length = max([len(t) for t in step_tags]) if step_tags else 0

        process_str = self.tag
        for idx in range(len(step_tags)):
            process_str += f"\n\t{idx+1:>2}) {step_tags[idx]:<{max_tag_length}} : {step_func_names[idx]}"

        return process_str

#-----------------------------------------------------------------------------------------------------------------------
# Batch Process
#-----------------------------------------------------------------------------------------------------------------------

class BatchAnalysis:
    def __init__(
            self,
            tag: str="BatchAnalysis",
            initialize: Process=Process(),
            compute: Process=Process(),
            inputs: dict={},
            outputs: dict={},
            db_path: Optional[str | Path] = None
        ):
        
        self.tag = tag
        
        # Path mapping and default settings.
        self.input_mappings = inputs
        self.output_mappings = outputs

        self.initialization_process = initialize
        self.compute_process = compute
        self._compiled_step = eqx.filter_jit(self.compute_process.run)

        self.db_path = db_path

    def run(
        self,
        system: System,
        settings: Settings,
        mode="zip",
        batch_size: Optional[int]=None,
        logger_handle: Optional[str]=None,
        **kwargs
    ):

        if logger_handle is not None:  # Inherit logger from dataset generator
            logger = logging.getLogger(logger_handle)
        else:  # Self logging
            logger = logging.getLogger(self.tag+"_Logger")
            
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            
            formatter = logging.Formatter('[%(asctime)s] - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            ch.setFormatter(formatter)
            logger.addHandler(ch)

        # Set up base state
        from RCAIDE.Framework.Missions.Conditions import Numerics
        state       = State(numerics=Numerics(number_of_control_points=1, calculate_integration=False))
        initials    = eqx.tree_at(lambda s: s.initials, state, None, is_leaf=lambda x: x is None)
        base_state  = eqx.tree_at(lambda s: s.initials, state, initials, is_leaf=lambda x: x is None)

        active_keys = []
        target_map  = []
        raw_arrays  = []

        # Validate inputs, convert to JAX arrays
        for k, v in kwargs.items():
            if k.lower() not in self.input_mappings:
                logger.warning(f"Unrecognized variable {k} ignored. "
                              f"Allowed variables: {list(self.input_mappings.keys())}")
            else:
                active_keys.append(k.lower())
                target_map.append(self.input_mappings[k.lower()][0])
                raw_arrays.append(jnp.atleast_1d(v))

        if len(active_keys) == 0:
            raise ValueError("No valid inputs provided.")
        for k, v in self.input_mappings.items():
            if k not in active_keys:
                active_keys.append(k)
                target_map.append(v[0])
                raw_arrays.append(jnp.atleast_1d(v[1]))

        # Get all flight states
        if mode == "zip":
            processed_arrays = jnp.broadcast_arrays(*raw_arrays)
        elif mode == "mesh":
            grids = jnp.meshgrid(*raw_arrays, indexing="ij")
            processed_arrays = [g.ravel().reshape(-1, 1) for g in grids]
        else:
            raise ValueError(f"Invalid mode {mode}. Supported modes: 'zip', 'mesh'.")
        
        total_states = len(processed_arrays[0])
        all_outputs = {k: [] for k in self.output_mappings.keys()}
        all_grads = defaultdict(list)
        jac_arr = None
        
        # Prepare for grads if provided
        if settings.analysis.gradient_map is not None:
            g_map = settings.analysis.gradient_map
            inp = g_map.state_inputs
            out = g_map.state_outputs
            
            grad_pairs = product(out, inp)
            grad_keys = [f"d{p[0].tag}_d{p[1].tag}" for p in grad_pairs]
            grad_idxs = list(product(range(len(out)), range(len(inp))))

        # Initialize VORJAX once
        init_results = self.initialization_process.run(base_state.expand_rows(batch_size), system, settings)
        state = init_results[0]
        system = init_results[1]
        settings = init_results[2]

        # Batch over computation
        for i in trange(0, total_states, batch_size, desc=f"Running {self.tag} Analysis"):
            batch_arrays = tuple(arr[i:i+batch_size].reshape(-1, 1) for arr in processed_arrays)
            actual_size  = len(batch_arrays[0])

            if actual_size < batch_size:
                pad_length = ((0, batch_size - actual_size), (0, 0))
                batch_arrays = tuple(jnp.pad(arr, pad_length, mode="edge") for arr in batch_arrays)
            
            batch_state = eqx.tree_at(lambda s: ru.get_all_targets(s, target_map), state, batch_arrays)
            
            try:    
                res = self._compiled_step(batch_state, system, settings)

                raw_coeff_arrs = jax.device_get(ru.get_all_targets(res[0], self.output_mappings.values()))
                clean_coeff_arrs = [arr[:actual_size] for arr in raw_coeff_arrs]
                
                for j, key in enumerate(self.output_mappings.keys()):
                    all_outputs[key].append(clean_coeff_arrs[j])
                
                if settings.analysis.gradient_map is not None:
                    jac_arr = jax.device_get(res[3])
                    for i, key in enumerate(grad_keys):
                        out_idx, in_idx = grad_idxs[i]
                        v_np = jac_arr[:actual_size, out_idx, in_idx]
                        all_grads[key].append(v_np)
            
            except Exception as e:
                logger.error(f"Failed at states {i} to {i + batch_size}. Injecting NaNs...", exc_info=True)
                nan_array = np.full((actual_size, 1), np.nan, dtype=np.float64)
                
                for key in self.output_mappings.keys():
                    all_outputs[key].append(nan_array)
                    
                if settings.analysis.gradient_map is not None:
                    for key in grad_keys:
                        all_grads[key].append(nan_array)
            
        merged_results = all_outputs | all_grads
        
        if self.db_path is not None:
            db_root = zarr.open_group(self.db_path, mode='a', zarr_format=2)
            for key, list_of_arrays in merged_results.items():
                # Concatenate the thousands of 256-length arrays into one 3,000,000-length array
                full_array = np.concatenate(list_of_arrays, axis=0)
                
                if key not in db_root:
                    db_root.create_array(
                        name=key,
                        shape=(0,) + full_array.shape[1:],
                        chunks=(100_000,) + full_array.shape[1:],
                        dtype=full_array.dtype,
                        compressor=Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
                    )
                db_root[key].append(full_array, axis=0)

        return merged_results

# ----------------------------------------------------------------------------------------------------------------------
#  Legacy Optimizer Interface
# ----------------------------------------------------------------------------------------------------------------------

class OptimizerInterface:
    """Interface with legacy optimizers to separate value and gradient function for RCAIDE Processes."""

    def __init__(self, process: Process,
                 base_state: State, base_system: System, base_settings: Settings,
                 grad_map: GradientMap, objective_path: PathTuple, 
                 **kwargs):
        
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
            state, system, settings = self.grad_map.update_inputs(x, self.base_state, self.base_system, self.base_settings)
            f_st, f_sys, f_setts, jac = self.process.run(state, system, settings, grad_map=self.grad_map)
            token = Token(state=f_st, system=f_sys, settings=f_setts)
            
            self.last_val = ru.get_target(token, self.objective_path)
            self.last_jac = np.array(jac)
            self.last_x = x
        
    def fun(self, x):
        self._update_cache(x)
        return self.last_val
    
    def jac(self, x):
        self._update_cache(x)
        return self.last_jac


