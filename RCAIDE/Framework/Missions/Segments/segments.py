# RCAIDE/Framework/Missions/Mission.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Jul 2024, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
# IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from __future__ import annotations

import threading
import timeit
import time
import sys

from typing import Callable, Literal, TYPE_CHECKING
from dataclasses import replace

# package imports

import jax
jax.config.update("jax_enable_x64", True)

first_structure = None
first_treedef = None

import equinox as eqx
import jax.numpy as jnp

from jaxopt import ScipyRootFinding, Broyden, Bisection, GaussNewton
from scipy.optimize import fsolve

# RCAIDE imports

from .Profiles import *

from RCAIDE.Library import Units

from RCAIDE.Framework import Process, ProcessStep
from RCAIDE.Framework.Process import skip
from RCAIDE.Framework.Missions.Initialize import *
from RCAIDE.Framework.Missions.Update     import *
from RCAIDE.Framework.Missions.Conditions.Controls import ControlVariable, DynamicResidual, ResidualNames

if TYPE_CHECKING:
    from RCAIDE.Framework import State, Settings, System

#-----------------------------------------------------------------------------------------------------------------------
# Mission Spinner
#-----------------------------------------------------------------------------------------------------------------------

class Spinner:
    def __init__(self, message="JIT compiling and solving...", enabled=True):
        self.spinner_chars = "|/-\\"
        self.message = message
        self.enabled = enabled  # Flag to easily turn it off during debugging
        self.running = False
        self.thread = None

    def spin(self):
        i = 0
        while self.running:
            sys.stdout.write(f"\r{self.message} {self.spinner_chars[i % 4]}")
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1

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
            sys.stdout.write(f"\r{self.message} Done!    \n")
            sys.stdout.flush()

# ----------------------------------------------------------------------------------------------------------------------
# Segment Subfunctions
# ----------------------------------------------------------------------------------------------------------------------

def _activate_control(control: str | ControlVariable, state):
    
    if isinstance(control, str):
        control_name = control.replace(' ', '_').lower()
        
        if control_name not in state.controls.__dataclass_fields__:
            # It's a custom control:
            new_ctrl = ControlVariable(tag=control, active=True)
            new_controls = state.controls.add_control_variable(new_ctrl)
        else:
            # It's a pre-existing control: grab the existing one, activate it, and replace it
            existing_ctrl = getattr(state.controls, control_name)
            active_ctrl = eqx.tree_at(lambda c: c.active, existing_ctrl, True)
            
            # Use getattr to map the path
            new_controls = eqx.tree_at(lambda p: getattr(p, control_name), state.controls, active_ctrl)
            
    elif isinstance(control, ControlVariable):
        active_ctrl = eqx.tree_at(lambda c: c.active, control, True)
        new_controls = state.controls.add_control_variable(active_ctrl)
        
    return eqx.tree_at(lambda s: s.controls, state, new_controls)



def _activate_residual(residual_name: ResidualNames, state):
    
    current_residual = getattr(state.dynamics, residual_name)
    active_residual = replace(current_residual, active=True)
    
    # Safely inject it using getattr path tracing
    new_dynamics = eqx.tree_at(
        lambda d: getattr(d, residual_name), 
        state.dynamics, 
        active_residual
    )
    
    return eqx.tree_at(lambda s: s.dynamics, state, new_dynamics).expand_rows(state.numerics.number_of_control_points)


# ----------------------------------------------------------------------------------------------------------------------
# Initialize Segment
# ----------------------------------------------------------------------------------------------------------------------

def _initialization_steps():
    return (
        ProcessStep(tag="Time",                 function=initialize_time),
        ProcessStep(tag="Mass",                 function=initialize_mass),
        ProcessStep(tag="Energy",               function=initialize_energy),
        ProcessStep(tag="Inertial Position",    function=initialize_inertial_position),
        ProcessStep(tag="Planetary Position",   function=initialize_planetary_position),
    )

class InitializeSegment(Process):

    tag: str = 'Segment Initialization'

    active_controls:   tuple[str|ControlVariable, ...]  = eqx.field(default_factory=tuple)
    active_residuals:  tuple[ResidualNames, ...]        = eqx.field(default_factory=tuple)

    controls_initial_guess: tuple[jnp.ndarray|float,...] = (0., 0.)

    steps: tuple[ProcessStep, ...] = eqx.field(default_factory=_initialization_steps)
    

    def __call__(self, state, system, settings):

        current_state = eqx.tree_at(lambda s: s.frames.planet.true_course, state, self.true_course)
        current_state = state.expand_rows(state.numerics.number_of_control_points)

        for ctrl in self.active_controls:
            current_state = _activate_control(ctrl, current_state)

        # Set up static routing for active controls
        active_controls = current_state.controls.get_active_controls()
        routing_table = tuple((ctrl.path, ctrl.path_indices) for ctrl in active_controls)
        new_controls = replace(current_state.controls, active_routing_table=routing_table)
        current_state = eqx.tree_at(lambda s: s.controls, current_state, new_controls)

        for res_name in self.active_residuals:
            current_state = _activate_residual(res_name, current_state)
        
        n_cp = int(current_state.numerics.number_of_control_points)
        
        if self.controls_initial_guess is not None and len(self.controls_initial_guess) > 0:
            new_unknowns = jnp.concatenate([jnp.full((n_cp,), v) for v in self.controls_initial_guess])
        else:
            new_unknowns = jnp.zeros((n_cp * len(self.active_controls)))

        new_residuals = jnp.zeros((n_cp, len(self.active_residuals)))
        
        current_state = eqx.tree_at(
            lambda s: (s.unknowns, s.residuals),
            current_state,
            (new_unknowns, new_residuals)
        )

        # 5. Validation
        assert current_state.check_controls(verbose=False), (
            f"During initialization of {self.tag} the number of active controls "
            "did not match the number of active residuals.\n"
        )

        current_state = current_state.unpack_unknowns(current_state.unknowns)

        # 6. Run the sub-steps
        current_state, system, settings = super().__call__(current_state, system, settings)

        return current_state, system, settings

# ----------------------------------------------------------------------------------------------------------------------
# Analyze Segment
# ----------------------------------------------------------------------------------------------------------------------

def _default_analyses():
    return (
        ProcessStep(tag="Time Differentials",   function=update_time_differentials),
        ProcessStep(tag="Acceleration",         function=update_acceleration),
        ProcessStep(tag="Angular Acceleration", function=update_angular_acceleration),
        ProcessStep(tag="Freestream",           function=update_freestream),
        ProcessStep(tag="Orientations",         function=update_orientations),
        ProcessStep(tag="Energy",               function=skip),
        ProcessStep(tag="Aerodynamics",         function=skip),
        ProcessStep(tag="Stability",            function=skip),
        ProcessStep(tag="Mass",                 function=update_mass_and_weight),
        ProcessStep(tag="Forces",               function=update_forces),
        ProcessStep(tag="Moments",              function=update_moments),
        ProcessStep(tag="Planetary Position",   function=update_planetary_position),
        ProcessStep(tag="Calculate Residuals",  function=flight_dynamics_residuals)
    )

class AnalyzeSegment(Process):

    tag: str = eqx.field(static=True, default="Segment Analysis Specification")

    steps: tuple[ProcessStep, ...] = eqx.field(default_factory=_default_analyses)


def find_circular_references(obj, path="root", visited=None): 
    if visited is None:
        visited = set()
    
    # Skip basic types and arrays (they don't hold other objects)
    if obj is None or isinstance(obj, (int, float, str, bool, tuple, frozenset)):
        return
    if type(obj).__name__ in ('ndarray', 'ArrayImpl', 'DynamicJaxprTracer'):
        return

    obj_id = id(obj)
    
    # If we've seen this exact object ID in this branch, we found the loop
    if obj_id in visited:
        print(f"CIRCULARITY FOUND:")
        print(f"Path: {path} loops back to an already visited {type(obj).__name__}")
        return True

    # Add this object's ID to the visited set for this branch
    visited.add(obj_id)

    # Recursively check dictionaries
    if isinstance(obj, dict):
        for k, v in obj.items():
            if find_circular_references(v, f"{path}['{k}']", visited.copy()):
                return True
            
    # Recursively check lists
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if find_circular_references(v, f"{path}[{i}]", visited.copy()):
                return True
            
    # Recursively check custom objects and dataclasses
    elif hasattr(obj, '__dict__'):
        for k, v in vars(obj).items():
            if find_circular_references(v, f"{path}.{k}", visited.copy()):
                return True
    
    return False

# ----------------------------------------------------------------------------------------------------------------------
# Iterate Segment
# ----------------------------------------------------------------------------------------------------------------------

def fsolve_results_parser(
        fsolve_result: tuple,
        state: State,
        system: System,
        settings: Settings,
) -> tuple[State, System, Settings]:

        unknowns:       jnp.ndarray     = jnp.array(fsolve_result[0])
        infodict:       dict            = fsolve_result[1]
        ier:            int             = fsolve_result[2]
        mesg:           str             = fsolve_result[3]

        current_state = state

        if ier != 1:
            print("Segment Convergence Failed:", mesg)
            unconverged_numerics = eqx.tree_at(lambda n:n.converged, state.numerics, False)
            current_state = eqx.tree_at(lambda s: s.numerics, state, unconverged_numerics)
        else:
            print("Segment Converged.")
            print("Number of function evaluations:", infodict['nfev'])
            converged_numerics = eqx.tree_at(lambda n:n.converged, state.numerics, True)
            current_state = eqx.tree_at(lambda s: s.numerics, state, converged_numerics)
        
        current_state = eqx.tree_at(lambda s: s.unknowns, current_state, unknowns)
        current_state = current_state.unpack_unknowns(unknowns)
        
        return current_state, system, settings

RootFinders = Literal[fsolve, ScipyRootFinding, Broyden, Bisection, GaussNewton]

class IterateSegment(Process):

    tag:            str     = eqx.field(static=True, default='Segment Convergence')
    analyze:        Process = eqx.field(default_factory=AnalyzeSegment)

    # Root-finder arguments and parser
    root_finder:    RootFinders = eqx.field(static=True, default=GaussNewton) # type: ignore
    results_parser: Callable    = eqx.field(static=True, default=fsolve_results_parser) # type: ignore
    
    def _get_residuals(self, unknowns, state: "State", system: "System", settings: "Settings"):

        current_state = state.unpack_unknowns(unknowns)
        current_state, _, _ = self.analyze(current_state, system, settings)
        current_state = current_state.pack_residuals()

        return current_state.residuals

    @eqx.filter_jit
    def _run_broyden_solver(self, x0, state, system, settings):
        root = Broyden(
            fun=self._get_residuals,
            tol=state.numerics.solution_tolerance,
            maxiter=state.numerics.max_evaluations,
        )

        return root.run(x0, state, system, settings)
    
    @eqx.filter_jit
    def _run_gauss_newton_solver(self, x0, state, system, settings):
        # Note the argument is `residual_fun` for the minimizer
        root = GaussNewton(
            residual_fun=self._get_residuals,
            tol=state.numerics.solution_tolerance,
            maxiter=state.numerics.max_evaluations
        )
        
        return root.run(x0, state, system, settings)

    def __call__(self, state, system, settings):
        with Spinner():
            if self.root_finder is fsolve:
                
                root_finder_kwargs = {
                    'func': self._get_residuals,
                    'x0': state.unknowns,
                    'args': (state, system, settings),
                    'xtol': state.numerics.solution_tolerance,
                    'maxfev': state.numerics.max_evaluations,
                    'epsfcn': state.numerics.step_size,
                    'full_output': True
                }

                results = self.root_finder(**root_finder_kwargs)

                return fsolve_results_parser(results, state, system, settings)

            elif self.root_finder in (ScipyRootFinding, Bisection):

                x0 = state.unknowns

                root = ScipyRootFinding(
                    method='hybr',
                    optimality_fun=self._get_residuals,
                    tol = state.numerics.solution_tolerance,
                    jit=True
                )
                if settings.mission.debugging:
                    if any([
                        find_circular_references(state, path="state"),
                        find_circular_references(system, path="system"),
                        find_circular_references(settings, path="settings")
                    ]):
                        raise RecursionError("Circularity found in mission data structures. Terminating mission.")

                t0 = timeit.default_timer()

                unknowns, _ = root.run(x0, state, system, settings)

                unknowns.block_until_ready()

                t1 = timeit.default_timer()

                print(f"--- Hybrid JAX/SciPy Solver: {t1 - t0:.6f} seconds ---")

                current_state = eqx.tree_at(lambda s: s.unknowns, state, unknowns)
                current_state = current_state.unpack_unknowns(unknowns)

                return self.analyze(current_state, system, settings)

            elif self.root_finder is Broyden:
                x0 = state.unknowns

                # 2. Start the clock
                t0 = timeit.default_timer()

                # 3. Fire the compiled GPU kernel
                # (Note: jaxopt returns an (unknowns, state) tuple, we just need the unknowns)
                unknowns, _ = self._run_broyden_solver(x0, state, system, settings)
                
                # 4. Wait for the final answer to come back across the PCIe bus
                unknowns.block_until_ready()

                # 5. Stop the clock
                t1 = timeit.default_timer()
                print(f"--- Pure GPU Broyden Solver: {t1 - t0:.6f} seconds ---")

                # 6. Unpack and continue
                current_state = eqx.tree_at(lambda s: s.unknowns, state, unknowns)
                current_state = current_state.unpack_unknowns(unknowns)

                return self.analyze(current_state, system, settings)
            
            elif self.root_finder is GaussNewton:
                x0 = state.unknowns

                # 2. Start the clock
                # t0 = timeit.default_timer()

                # 3. Fire the compiled GPU kernel
                # (Note: jaxopt returns an (unknowns, state) tuple, we just need the unknowns)
                unknowns, opt_state = self._run_gauss_newton_solver(x0, state, system, settings)
                
                # 4. Wait for the final answer to come back across the PCIe bus
                unknowns.block_until_ready()

                print("\n" + "="*30)
                print("JAXOPT SOLVER DIAGNOSTICS")
                print("="*30)
                # JAX arrays need to be cast or formatted to print cleanly
                print(f"Iterations taken: {opt_state.iter_num}")
                print(f"Final Error/Residual: {opt_state.error}")
                
                # Depending on the specific JAXopt solver, you might also have:
                if hasattr(opt_state, 'stepsize'):
                    print(f"Final Stepsize: {opt_state.stepsize}")
                if hasattr(opt_state, 'value'):
                    print(f"Objective Value: {opt_state.value}")
                print("="*30 + "\n")

                # 5. Stop the clock
                # t1 = timeit.default_timer()
                # print(f"--- Pure GPU Gauss-Newton Solver: {t1 - t0:.6f} seconds ---")

                # 6. Unpack and continue
                current_state = eqx.tree_at(lambda s: s.unknowns, state, unknowns)
                current_state = current_state.unpack_unknowns(unknowns)

                return self.analyze(current_state, system, settings)
            
            else:
                return state, system, settings     

# ----------------------------------------------------------------------------------------------------------------------
# Finalize Segment
# ----------------------------------------------------------------------------------------------------------------------


def _reset_controls_and_residuals(
            state: State,
            system: System,
            settings: Settings,
    ):
        current_state = state

        def _turn_off_control(node):
            if isinstance(node, ControlVariable):
                return eqx.tree_at(lambda c: c.active, node, False)
            return node        

        def _turn_off_residual(node):
            if isinstance(node, DynamicResidual):
                return replace(node, active=False)
            return node
        
        new_controls = jax.tree_util.tree_map(
            _turn_off_control, 
            current_state.controls, 
            is_leaf=lambda x: isinstance(x, ControlVariable)
        )

        new_dynamics = jax.tree_util.tree_map(
            _turn_off_residual, 
            current_state.dynamics, 
            is_leaf=lambda x: isinstance(x, DynamicResidual)
        )

        current_state = eqx.tree_at(
            lambda s: (s.controls, s.dynamics), 
            current_state, 
            (new_controls, new_dynamics)
        )
        
        return current_state, system, settings

def _default_finalize():
    return (
        ProcessStep(tag="Deactivate Controls & Residuals", function=_reset_controls_and_residuals),
    )

class FinalizeSegment(Process):

    tag: str = eqx.field(static=True, default='Segment Finalization')
    steps: tuple[ProcessStep, ...] = eqx.field(default_factory=_default_finalize)
    

# ----------------------------------------------------------------------------------------------------------------------
# Converged Segments
# ----------------------------------------------------------------------------------------------------------------------


class Segment(Process):

    tag: str = eqx.field(static=True, default="Segment")

    # Pass-through configuration for InitializeSegment
    active_controls:  tuple[str | ControlVariable, ...]     = eqx.field(default_factory=tuple)
    active_residuals: tuple[ResidualNames, ...]             = eqx.field(default_factory=tuple)
    controls_initial_guess: tuple[jnp.ndarray|float, ...]   = (0., 0.)

    position_profile:   PositionProfile = eqx.field(default_factory=ConstantAltitude)
    speed_profile:      SpeedProfile    = eqx.field(default_factory=ConstantSpeed)
    velocity_profile:   VelocityProfile = eqx.field(default_factory=ConstantAltitudeChangeRate)
    duration_profile:   DurationProfile = eqx.field(default_factory=FixedDistance)

    # Global dynamics variables
    sideslip_angle:         float = 0.0
    temperature_deviation:  float = 0.0
    true_course:            float = 0.0

    # Start with an empty tuple. We will populate it securely in __post_init__
    steps: tuple = eqx.field(default_factory=tuple)

    def __post_init__(self):
        # Only build the default steps if the user didn't explicitly provide custom ones
        if len(self.steps) == 0:
            
            # 1. Build the steps, passing the controls configuration directly into InitializeSegment
            init_step = InitializeSegment(
                tag=f"Initialize {self.tag}",
                active_controls=self.active_controls,
                active_residuals=self.active_residuals,
                controls_initial_guess=self.controls_initial_guess
            )
            
            # Add profile initialization
            init_step = eqx.tree_at(
                lambda i:i.steps, init_step, 
                init_step.steps + (self.position_profile, self.speed_profile, self.velocity_profile, self.duration_profile))
            
            iter_step = IterateSegment(tag=f"Iterate {self.tag}")
            fin_step  = FinalizeSegment(tag=f"Finalize {self.tag}")
            
            # 2. Safely lock them into the frozen object
            object.__setattr__(self, "steps", (init_step, iter_step, fin_step))

    # ----------------------------------------------------------------------------------
    # Quality-of-Life Accessors (Replaces __getattr__)
    # ----------------------------------------------------------------------------------
    @property
    def initialize(self) -> InitializeSegment:
        return self.steps[0]

    @property
    def iterate(self) -> IterateSegment:
        return self.steps[1]

    @property
    def finalize(self) -> FinalizeSegment:
        return self.steps[2]

    @property
    def analyze(self) -> AnalyzeSegment:
        return self.steps[1].analyze

    # ----------------------------------------------------------------------------------
    # Execution
    # ----------------------------------------------------------------------------------
    def __call__(self, state, system, settings) -> tuple["State", "System", "Settings"]:
        
        if settings.mission.verbose:
            for step in self.analyze.steps:
                if isinstance(step, ProcessStep) and step.function is skip:
                    print(f"Warning: Skipping step '{step.tag}' due to missing analysis function.")
        
        return super().__call__(state, system, settings)


#-----------------------------------------------------------------------------------------------------------------------
# Optimal Segments
#-----------------------------------------------------------------------------------------------------------------------


# @chex.dataclass(kw_only=True)
# class OptimalSegment(Process):

#     tag:                    str    = 'Optimize Segment'
#     optimization_method:    str    = 'SLSQP'
#     display_optimization:   bool   = False

#     initialize:             InitializeSegment   = field(default_factory=InitializeSegment)
#     analyze:                AnalyzeSegment      = field(default_factory=AnalyzeSegment)

#     calculate_objective:    Callable    = None
#     bounds:                 List[Any]   = None
#     constraints:            dict        = None

#     function: Callable = scipy.optimize.minimize

#     def _results_parser(self, res):

#         self.state.unknowns.unpack_array(res.x)
#         self.state.objective.unpack_array(res.fun)

#         self.last_result = res

#         return self.state, self.settings, self.system

#     def __call__(self, *args, **kwargs) -> Tuple["rcf.State", "rcf.System", "rcf.Settings"]:

#         # Fix Bounds
#         NCP = self.state.numerics.number_of_control_points
#         new_bounds = []
#         for b in self.bounds:
#             new_bounds.extend([b for _ in range(NCP)])
#         self.bounds = new_bounds

#         # self.state, self.system, self.settings = args[0]
#         self.state.initials = self.state
#         self.update_details()

#         self.initialize.state = self.state
#         self.initialize.system = self.system
#         self.initialize.settings = self.settings

#         self.state, self.system, self.settings = self.initialize((self.state, self.system, self.settings))

#         def _obj(U):
#             self.state.unknowns.unpack_array(U)
#             self.analyze.state = self.state
#             self.analyze.system = self.system
#             self.analyze.settings = self.settings
#             self.state, self.system, self.settings = self.analyze()
#             return self.calculate_objective(self.state, self.system, self.settings)

#         _obj_fcn    = jit(_obj)
#         _obj_grad   = jit(grad(_obj_fcn))

#         res = minimize(
#             fun=_obj_fcn,
#             jac=_obj_grad,
#             x0=self.state.unknowns.pack_array(),
#             method=self.optimization_method,
#             bounds=self.bounds,
#             constraints=self.constraints,
#             options={'disp': self.display_optimization,
#                      'maxiter': self.state.numerics.max_evaluations},
#             tol=self.state.numerics.solution_tolerance,
#         )

#         self.state, self.system, self.settings = self._results_parser(res)

#         self.state, self.system, self.settings = self.finalize(self.state, self.system, self.settings)

#         return self.state, self.system, self.settings


# #-----------------------------------------------------------------------------------------------------------------------
# # Energy Optimal Segments

# def energy_use(
#         state: "rcf.State",
#         system: "rcf.System",
#         settings: "rcf.Settings"
# ):

#     energy_start    = state.energy.total_energy[0]
#     energy_end      = state.energy.total_energy[-1]
#     energy_used     = energy_end - energy_start

#     return energy_used[0]


# @chex.dataclass(kw_only=True)
# class EnergyOptimalCruise(OptimalSegment):

#     tag: str = 'Energy Optimal Cruise'

#     altitude: float = 0.0
#     distance: float = 0.0

#     calculate_objective: Callable = energy_use

#     def __post_init__(self):
#         distance_check = lambda x: self.state.frames.inertial.position_vector[-1, 0]
#         self.bounds = [(-np.pi/12, np.pi/12), (0., 1.)]
#         self.constraints = [NonlinearConstraint(distance_check, lb=self.distance, ub=self.distance)]


# @chex.dataclass(kw_only=True)
# class EnergyOptimalAltitudeChange(OptimalSegment):

#     tag: str = 'Energy Optimal Altitude Change'

#     altitude_start: float = 0.0
#     altitude_end:   float = 0.0

#     calculate_objective: Callable = energy_use

#     def __post_init__(self):
#         start_check = lambda x: self.state.frames.inertial.position_vector[0, 2]
#         end_check = lambda x: self.state.frames.inertial.position_vector[-1, 2]
#         self.bounds = [(-np.pi/4, np.pi/4), (0., 1.)]
#         self.constraints = [NonlinearConstraint(start_check, lb=self.altitude_start, ub=self.altitude_start),
#                             NonlinearConstraint(end_check, lb=self.altitude_end, ub=self.altitude_end)]
