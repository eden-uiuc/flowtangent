# RCAIDE/Framework/Process.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Jul 2024, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import Callable, Self, TYPE_CHECKING

# package imports
import equinox as eqx

# RCAIDE imports
if TYPE_CHECKING:
    from RCAIDE.Framework import State, System, Settings


# ----------------------------------------------------------------------------------------------------------------------
#  Argument Passer
# ----------------------------------------------------------------------------------------------------------------------


def skip(*args):
    return args


class ProcessStep(eqx.Module):

    function:       Callable | str     = eqx.field(static=True, default=skip)
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
        return self.function(state, system, settings) #type: ignore
    
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
            )
        )

class Process(eqx.Module):

    tag:                str                     = eqx.field(static=True, default="Process")

    steps:              tuple[ProcessStep, ...] = eqx.field(default_factory=tuple)

    initial_step:       int                     = eqx.field(static=True, default=0)

    initial_state:      "State | None"          = None
    initial_system:     "System | None"         = None
    initial_settings:   "Settings | None"       = None

    final_state:        "State | None"          = None
    final_system:       "System | None"         = None
    final_settings:     "Settings | None"       = None


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
        
        original_state = state
        original_system = system
        original_settings = settings

        logged_steps = []

        for step in self.steps:
            # 1. Run the step
            new_state, new_system, new_settings = step(state, system, settings)
            
            # 2. Record the pre-step (or post-step) conditions into the history
            logged_step = step.record_history(
                initial_state=state,
                initial_system=system,
                initial_settings=settings,
                final_state=new_state,
                final_system=new_system,
                final_settings=new_settings,
            )
            logged_steps.append(logged_step)
            
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
            )
        )
        
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

    def __repr__(self):

        return self.tag
    
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
            if isinstance(step, ProcessStep):
                func = step.function
                name = getattr(func, '__name__', func.__class__.__name__)
                step_func_names.append(name)
            elif isinstance(step, Process):
                step_func_names.append(f"<Process>: {len(step.steps)} Step(s)")

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

