# RCAIDE/Framework/Process.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Jul 2024, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import Callable, Self, TYPE_CHECKING, Generator, Tuple

import re

# package imports
import equinox as eqx
import networkx as nx

# RCAIDE imports
if TYPE_CHECKING:
    from RCAIDE.Framework import State, System, Settings


# ----------------------------------------------------------------------------------------------------------------------
#  Argument Passer
# ----------------------------------------------------------------------------------------------------------------------


def null_step(*args):
    return args


class ProcessStep(eqx.Module):

    function:       Callable | str     = eqx.field(static=True, default=null_step)
    tag:            str                = eqx.field(static=True, default="Process Step")
    
    initial_state:        State | None     = None
    initial_system:       System | None    = None
    initial_settings:     Settings | None  = None

    final_state:          State | None     = None
    final_system:         System | None    = None
    final_settings:       Settings | None  = None

    def __call__(self, state, system, settings):
        if settings.DEBUG_MODE: print(f"  Step: '{self.tag}'")
        # Default calling behavior, assumes function is callable.
        # String overwrite only for steps with __call__ override
        return self.function(state, system, settings)  # type: ignore

    def run(self, state, system, settings):
        return self(state, system, settings)
    
    def run_with_history(self, state, system, settings):
        return *self(state, system, settings), None
    
    def __repr__(self):
        return self.tag

    def record_history(
            self,
            initial_state,
            initial_system,
            initial_settings,
            final_state,
            final_system,
            final_settings,
    ) -> ProcessStep:
        
        return eqx.tree_at(
            lambda s: (
                s.initial_state,
                s.initial_system,
                s.initial_settings,
                s.final_state,
                s.final_system,
                s.final_settings,
            ), self,
            (
                initial_state,
                initial_system,
                initial_settings,
                final_state,
                final_system,
                final_settings,
            ),
            is_leaf=lambda x: x is None
        )

    @property
    def inputs(self) -> set:
        return getattr(self.function, "_inputs", set())

    @property
    def outputs(self) -> set:
        return getattr(self.function, "_results", set())


class Process(ProcessStep):

    tag:                str                     = eqx.field(static=True, default="Process")

    steps:              tuple[ProcessStep, ...] = eqx.field(default_factory=tuple)

    initial_step:       int                     = eqx.field(static=True, default=0)

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

    def run(self, state, system, settings):
        return self(state, system, settings)

    def run_with_history(self, state, system, settings):
        if settings.DEBUG_MODE: print(f"Beginning Process: '{self.tag}'")

        original_state = state
        original_system = system
        original_settings = settings

        logged_steps = []

        for step in self.steps:
            # 1. Run the step
            new_state, new_system, new_settings, history = step.run_with_history(state, system, settings)
            
            # 2. Record the pre-step (or post-step) conditions into the history
            if not history: # Single Process Step
                logged_step = step.record_history(
                    initial_state=state,
                    initial_system=system,
                    initial_settings=settings,
                    final_state=new_state,
                    final_system=new_system,
                    final_settings=new_settings,
                )
                logged_steps.append(logged_step)
            else: # Multi-Step Process
                logged_steps.append(history)
            
            # 3. Advance to next step
            state, system, settings = new_state, new_system, new_settings

        # Return the final states AND the structurally new, logged Process
        logged_process = eqx.tree_at(
            lambda p: (
                p.steps, 
                p.initial_state, p.initial_system, p.initial_settings,
                p.final_state, p.final_system, p.final_settings
            ),
            self,
            (
                tuple(logged_steps),
                original_state, original_system, original_settings,
                state, system, settings # The final ones
            ),
            is_leaf=lambda x: x is None
        )
        if settings.DEBUG_MODE: print(f"Process '{self.tag}' Complete.")
        
        return state, system, settings, logged_process

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
                yield from step._get_flattened_steps(prefix=f"{node_name}.")
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


if __name__ == "__main__":

    def lib_sq_func(x):
        return x**2

    def fw_sq_func(state, system, settings):
        x = state.x
        results = lib_sq_func(x)
        state.x = results

        return state, system, settings


    def lib_add_func(x, y):
        return x + y

    def fw_add_func(state, system, settings):
        x = state.x
        y = state.y
        results = lib_add_func(x, y)
        state.x = results

        return state, system, settings

    square_and_add = Process(
        steps=(ProcessStep(function=fw_sq_func), ProcessStep(function=fw_add_func)),
        initial_step=0
    )

    def sq_add_func(x, y):
        square_and_add.initial_state.x = x
        square_and_add.initial_state.y = y
        st, set, sys = square_and_add.run(State(), System(), Settings())
        return st.x

    from jax import value_and_grad

    sq_grad = value_and_grad(sq_add_func)

    results = sq_grad(3., 6.)

    print(square_and_add.details)
    print(results)

