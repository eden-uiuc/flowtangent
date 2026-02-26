# RCAIDE/Framework/Missions/Mission.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Jul 2024, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
# IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from typing import Callable, List, Any, Tuple
from dataclasses import field

# package imports

import jax
import chex
import scipy

first_structure = None
first_treedef = None

import optimistix as optx
from jaxopt import ScipyRootFinding

jax.config.update("jax_enable_x64", True)

from jax import jit, grad

#import numpy as np
import jax.numpy as np
from scipy.optimize import minimize, NonlinearConstraint, fsolve

# RCAIDE imports

import RCAIDE.Framework as rcf
from RCAIDE.Framework import Process, ProcessStep
from RCAIDE.Framework.Process import skip
from RCAIDE.Framework.Missions.Initialize import *
from RCAIDE.Framework.Missions.Update     import *
from RCAIDE.Framework.Missions.Converge   import fsolve_results_parser, fsolve_update_kwargs
from RCAIDE.Framework.Missions.Conditions.Controls import ControlVariable

# ----------------------------------------------------------------------------------------------------------------------
# Segment Subfunctions
# ----------------------------------------------------------------------------------------------------------------------


@chex.dataclass(kw_only=True)
class InitializeSegment(Process):

    tag: str = 'Segment Initialization'

    active_controls:   Tuple[str|ControlVariable, ...] = field(default_factory=tuple)
    active_residuals:  Tuple[str|ControlVariable, ...] = field(default_factory=tuple)

    controls_initial_guess: np.ndarray = None

    @staticmethod
    def _activate_control(control: str | ControlVariable, state: "rcf.State") -> None:
        
        current_state = state
        
        if isinstance(control, str):
            control = control.replace(' ', '_').lower()  # Normalize control name to lowercase and replace spaces with underscores
            if control not in vars(current_state.controls).keys():
                current_state.controls.add_control_variable(ControlVariable(tag=control))
            current_state.controls[control].active = True
        else:
            current_state.controls.add_control_variable(control)
        
        return current_state

    @staticmethod
    def _activate_residual(residual_name: str, state) -> None:
        
        current_state = state
        
        residual_name = residual_name.replace(' ', '_').lower()  # Normalize residual name to lowercase and replace spaces with underscores
        current_state.dynamics[residual_name].active = True

        return current_state

    def __post_init__(self):

        default_steps = [
            # Step Name              Step Functions
            ("Expand State",         expand_state),
            ("Time",                 initialize_time),
            ("Mass",                 initialize_mass),
            ("Energy",               initialize_energy),
            ("Inertial Position",    initialize_inertial_position),
            ("Planetary Position",   initialize_planetary_position),
        ]

        for name, function in default_steps:
            self.append(ProcessStep(tag=name, function=function))

    def __call__(self, state, system, settings):
        
        current_state = state

        for ctrl_name in self.active_controls:
            current_state = self._activate_control(ctrl_name, current_state)
        if isinstance(self.controls_initial_guess, np.ndarray):
            current_state.unknowns = self.controls_initial_guess
        elif isinstance(self.controls_initial_guess, tuple):
            n_cp = int(current_state.numerics.number_of_control_points)
            current_state.unknowns = np.concatenate([np.full((n_cp,), v) for v in self.controls_initial_guess])
        else:
            current_state.unknowns = np.zeros((self.state.numerics.number_of_control_points * len(self.active_controls)))

        for res_name in self.active_residuals:
            current_state = self._activate_residual(res_name, current_state)
        current_state.residuals = np.zeros((current_state.numerics.number_of_control_points, len(self.active_residuals)))

        assert current_state.check_controls(verbose=False), (
            f"During initialization of {self.tag} the number of active controls"
            "did not match the number of active residuals.\n"
            "Please run State.check_controls(verbose=True) to see details"
        )

        current_state = current_state.unpack_unknowns(current_state.unknowns)

        return super(InitializeSegment, self).__call__(current_state, system, settings)


@chex.dataclass(kw_only=True)
class AnalyzeSegment(Process):

    tag: str = "Segment Analysis Specification"

    def __post_init__(self):

        default_steps = [
            ("Time Differentials",   update_time_differentials),
            ("Acceleration",         update_acceleration),
            ("Angular Acceleration", update_angular_acceleration),
            ("Altitude",             update_altitude),
            ("Gravity",              update_gravity),
            ("Freestream",           update_freestream),
            ("Orientations",         update_orientations),
            ("Energy",               skip),
            ("Aerodynamics",         skip),
            ("Stability",            skip),
            ("Mass",                 update_mass_and_weight),
            ("Forces",               update_forces),
            ("Moments",              update_moments),
            ("Planetary Position",   skip),
            ("Calculate Residuals",  flight_dynamics_residuals)
        ]

        for name, function in default_steps:
            self.append(ProcessStep(tag=name, function=function))

def static_pure_residuals(unknowns, segment, state, system, settings):

    return segment._get_pure_residuals(unknowns, state, system, settings)

def find_circular_references(obj, path="root", visited=None):
    if visited is None:
        visited = set()
        
    # Skip basic types and arrays (they don't hold other objects)
    if obj is None or isinstance(obj, (int, float, str, bool, tuple, frozenset)):
        return
    if type(obj).__name__ in ('ndarray', 'ArrayImpl', 'DynamicJaxprTracer'):
        return

    obj_id = id(obj)
    
    # If we've seen this exact object ID in this branch, we found the loop!
    if obj_id in visited:
        print(f"CIRCULARITY FOUND:")
        print(f"Path: {path} loops back to an already visited {type(obj).__name__}")
        return

    # Add this object's ID to the visited set for this branch
    visited.add(obj_id)

    # Recursively check dictionaries
    if isinstance(obj, dict):
        for k, v in obj.items():
            find_circular_references(v, f"{path}['{k}']", visited.copy())
            
    # Recursively check lists
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            find_circular_references(v, f"{path}[{i}]", visited.copy())
            
    # Recursively check custom objects and dataclasses
    elif hasattr(obj, '__dict__'):
        for k, v in vars(obj).items():
            find_circular_references(v, f"{path}.{k}", visited.copy())

@chex.dataclass(kw_only=True)
class IterateSegment(Process):

    tag:            str     = 'Segment Convergence'

    analyze:        Process = None

    # Root-finder arguments and parser
    root_finder:            Callable    = ScipyRootFinding
    root_finder_args:       Tuple       = None
    root_finder_kwargs:     dict        = None

    update_args:            Callable    = None
    update_kwargs:          Callable    = fsolve_update_kwargs
    results_parser:         Callable[[Any], Tuple["rcf.State", "rcf.System", "rcf.Settings"]] = fsolve_results_parser

    def _get_fsolve_residuals(self, unknowns, state, system, settings):
        """
        Wraps the analysis step to calculate residuals for fsolve.
        """

        current_state = state.unpack_unknowns(unknowns)

        current_state, current_system, current_settings = self.analyze(
            current_state, system, settings
        )

        current_state.pack_residuals()

        if self.update_args:
            self.root_finder_args = self.update_args(self.root_finder_args, self.state, self.system, self.settings)
        if self.update_kwargs:
            self.root_finder_kwargs = self.update_kwargs(self.root_finder_kwargs, self.state, self.system, self.settings)

        return current_state.residuals.ravel(order='F')
    
    def _get_pure_residuals(self, unknowns, state, system, settings):
        """
        Wraps the analysis step to calculate residuals for fsolve.
        """

        current_state = state.unpack_unknowns(unknowns)

        current_state, _, _ = self.analyze(current_state, system, settings)

        current_state = current_state.pack_residuals()

        return current_state.residuals.ravel(order='F')
                
        # Call your actual pure residual function
        return self._get_pure_residuals(unknowns, state, system, settings)

    def __call__(self, state, system, settings):
    

        if self.root_finder is fsolve:
            if self.root_finder_kwargs is None:

                self.root_finder_kwargs = {
                    'func': self._get_fsolve_residuals,
                    'x0': self.state.unknowns,
                    'args': (state, system, settings),
                    'xtol': self.state.numerics.solution_tolerance,
                    'maxfev': self.state.numerics.max_evaluations,
                    'epsfcn': self.state.numerics.step_size,
                    'full_output': True
                }


            results = self.root_finder(**self.root_finder_kwargs)

            self.state, self.system, self.settings = fsolve_results_parser(results, self.state, self.system, self.settings)


        if self.root_finder is ScipyRootFinding:

            self.update_kwargs = False

            current_state = state
            current_system = system
            current_settings = settings

            x0 = current_state.unknowns

            combined_args = (self, current_state, current_system, current_settings)

            root = ScipyRootFinding(
                method='hybr',
                optimality_fun=static_pure_residuals,
                tol = current_state.numerics.solution_tolerance,
                jit=False,  #TODO: Test JIT compilation
            )

            print("--- Hunting for cycles in State ---")
            find_circular_references(current_state, path="state")

            print("--- Hunting for cycles in System ---")
            find_circular_references(current_system, path="system")

            print("--- Hunting for cycles in Settings ---")
            find_circular_references(current_settings, path="settings")

            unknowns, _ = root.run(x0, combined_args)

            self.state.unknowns = unknowns
            self.state = self.state.unpack_unknowns(self.state.unknowns)

            self.state, self.system, self.settings = self.analyze(self.state, self.system, self.settings)
        
        return self.state, self.system, self.settings
        


@chex.dataclass(kw_only=True)
class FinalizeSegment(Process):

    tag: str = 'Segment Finalization'

    @staticmethod
    def _reset_controls_and_residuals(
            state: "rcf.State",
            system: "rcf.System",
            settings: "rcf.Settings",
    ):

        for name, control_var in vars(state.controls).items():
            if hasattr(control_var, 'active') and control_var.active:
                control_var.active = False

        for name, residual_var in vars(state.dynamics).items():
            if hasattr(residual_var, 'active') and residual_var.active:
                residual_var.active = False

        return state, system, settings

    def __post_init__(self):
        self.append(
            ProcessStep(tag='Reset Controls and Residuals',
                        function=self._reset_controls_and_residuals)
        )


# ----------------------------------------------------------------------------------------------------------------------
# Converged Segments
# ----------------------------------------------------------------------------------------------------------------------


@chex.dataclass(kw_only=True)
class Segment(Process):

    tag: str = "Segment"

    active_controls:    Tuple[str, ...]   = None
    active_residuals:   Tuple[str, ...]   = None

    controls_initial_guess: tuple | np.ndarray = None

    initialize:         InitializeSegment   = field(default_factory=InitializeSegment)
    iterate:            IterateSegment      = field(default_factory=IterateSegment)
    analyze:            AnalyzeSegment      = field(default_factory=AnalyzeSegment)
    finalize:           FinalizeSegment     = field(default_factory=FinalizeSegment)

    # Global dynamics variables
    sideslip_angle:         float = 0.0
    temperature_deviation:  float = 0.0

    def __post_init__(self):

        self.initialize.tag                     = f'Initialize {self.tag}'
        self.initialize.active_controls         = self.active_controls
        self.initialize.active_residuals        = self.active_residuals

        self.initialize.controls_initial_guess  = self.controls_initial_guess

        self.analyze.tag                        = f'Analyze {self.tag}'

        self.iterate.tag                        = f'Iterate {self.tag}'
        self.iterate.analyze                    = self.analyze

        self.finalize.tag                       = f'Finalize {self.tag}'

        self.steps = [
            self.initialize,
            self.iterate,
            self.finalize,
        ]

    def __call__(self, *args, **kwargs) -> Tuple["rcf.State", "rcf.System", "rcf.Settings"]:

        for step in self.analyze.steps:
            if isinstance(step, ProcessStep) and step.function is skip:
                print(f"Skipping step {step.tag} due to missing analysis function.")

        return super(Segment, self).__call__(*args, **kwargs)


#-----------------------------------------------------------------------------------------------------------------------
# Optimal Segments
#-----------------------------------------------------------------------------------------------------------------------


@chex.dataclass(kw_only=True)
class OptimalSegment(Process):

    tag:                    str    = 'Optimize Segment'
    optimization_method:    str    = 'SLSQP'
    display_optimization:   bool   = False

    initialize:             InitializeSegment   = field(default_factory=InitializeSegment)
    analyze:                AnalyzeSegment      = field(default_factory=AnalyzeSegment)

    calculate_objective:    Callable    = None
    bounds:                 List[Any]   = None
    constraints:            dict        = None

    function: Callable = scipy.optimize.minimize

    def _results_parser(self, res):

        self.state.unknowns.unpack_array(res.x)
        self.state.objective.unpack_array(res.fun)

        self.last_result = res

        return self.state, self.settings, self.system

    def __call__(self, *args, **kwargs) -> Tuple["rcf.State", "rcf.System", "rcf.Settings"]:

        # Fix Bounds
        NCP = self.state.numerics.number_of_control_points
        new_bounds = []
        for b in self.bounds:
            new_bounds.extend([b for _ in range(NCP)])
        self.bounds = new_bounds

        # self.state, self.system, self.settings = args[0]
        self.state.initials = self.state
        self.update_details()

        self.initialize.state = self.state
        self.initialize.system = self.system
        self.initialize.settings = self.settings

        self.state, self.system, self.settings = self.initialize((self.state, self.system, self.settings))

        def _obj(U):
            self.state.unknowns.unpack_array(U)
            self.analyze.state = self.state
            self.analyze.system = self.system
            self.analyze.settings = self.settings
            self.state, self.system, self.settings = self.analyze()
            return self.calculate_objective(self.state, self.system, self.settings)

        _obj_fcn    = jit(_obj)
        _obj_grad   = jit(grad(_obj_fcn))

        res = minimize(
            fun=_obj_fcn,
            jac=_obj_grad,
            x0=self.state.unknowns.pack_array(),
            method=self.optimization_method,
            bounds=self.bounds,
            constraints=self.constraints,
            options={'disp': self.display_optimization,
                     'maxiter': self.state.numerics.max_evaluations},
            tol=self.state.numerics.solution_tolerance,
        )

        self.state, self.system, self.settings = self._results_parser(res)

        self.state, self.system, self.settings = self.finalize(self.state, self.system, self.settings)

        return self.state, self.system, self.settings


#-----------------------------------------------------------------------------------------------------------------------
# Energy Optimal Segments

def energy_use(
        state: "rcf.State",
        system: "rcf.System",
        settings: "rcf.Settings"
):

    energy_start    = state.energy.total_energy[0]
    energy_end      = state.energy.total_energy[-1]
    energy_used     = energy_end - energy_start

    return energy_used[0]


@chex.dataclass(kw_only=True)
class EnergyOptimalCruise(OptimalSegment):

    tag: str = 'Energy Optimal Cruise'

    altitude: float = 0.0
    distance: float = 0.0

    calculate_objective: Callable = energy_use

    def __post_init__(self):
        distance_check = lambda x: self.state.frames.inertial.position_vector[-1, 0]
        self.bounds = [(-np.pi/12, np.pi/12), (0., 1.)]
        self.constraints = [NonlinearConstraint(distance_check, lb=self.distance, ub=self.distance)]


@chex.dataclass(kw_only=True)
class EnergyOptimalAltitudeChange(OptimalSegment):

    tag: str = 'Energy Optimal Altitude Change'

    altitude_start: float = 0.0
    altitude_end:   float = 0.0

    calculate_objective: Callable = energy_use

    def __post_init__(self):
        start_check = lambda x: self.state.frames.inertial.position_vector[0, 2]
        end_check = lambda x: self.state.frames.inertial.position_vector[-1, 2]
        self.bounds = [(-np.pi/4, np.pi/4), (0., 1.)]
        self.constraints = [NonlinearConstraint(start_check, lb=self.altitude_start, ub=self.altitude_start),
                            NonlinearConstraint(end_check, lb=self.altitude_end, ub=self.altitude_end)]
