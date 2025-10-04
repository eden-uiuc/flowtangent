# RCAIDE/Framework/Missions/Mission.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Jul 2024, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
# IMPORT
# ----------------------------------------------------------------------------------------------------------------------

import inspect

from functools import reduce
from typing import Callable, List, Any, Tuple
from dataclasses import field

# package imports

import jax
import jax.numpy as jnp
import scipy.optimize

import chex

jax.config.update("jax_enable_x64", True)

from jax import jit, grad

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

# RCAIDE imports

import RCAIDE.Framework as rcf
from RCAIDE.Framework import Process, ProcessStep
from RCAIDE.Framework.Process import skip
from RCAIDE.Framework.Missions.Conditions   import Conditions
from RCAIDE.Framework.Missions.Initialize   import *
from RCAIDE.Framework.Missions.Update       import *
from RCAIDE.Framework.Missions.Converge     import fsolve_results_parser
from RCAIDE.Framework.Missions              import flight_dynamics_residuals

# ----------------------------------------------------------------------------------------------------------------------
# Segment Subfunctions
# ----------------------------------------------------------------------------------------------------------------------


@chex.dataclass(kw_only=True)
class InitializeSegment(Process):

    tag: str = 'Segment Initialization'

    def __post_init__(self):

        default_steps = [
            # Step Name              Step Functions
            ("Expand State",         expand_state),
            ("Time",                 initialize_time),
            ("Mass",                 initialize_mass),
            ("Energy",               initialize_energy),
            ("Inertial Position",    initialize_inertial_position),
            ("Planetary Position",   initialize_planetary_position)
        ]

        for name, function in default_steps:
            self.append(ProcessStep(tag=name, function=function))


@chex.dataclass(kw_only=True)
class AnalyzeSegment(Process):

    tag: str = "Segment Analysis"

    def __post_init__(self):

        default_steps = [
            ("Time Differentials",   update_time_differentials),
            ("Acceleration",         update_acceleration),
            ("Angular Acceleration", update_angular_acceleration),
            ("Altitude",             update_altitude),
            ("Gravity",              skip),
            ("Freestream",           update_freestream),
            ("Orientations",         update_orientations),
            ("Energy",               skip),
            ("Aerodynamics",         skip),
            ("Stability",            skip),
            ("Mass",                 skip),
            ("Forces",               update_forces),
            ("Moments",              update_moments),
            ("Planetary Position",   skip)
        ]

        for name, function in default_steps:
            self.append(ProcessStep(tag=name, function=function))


# ----------------------------------------------------------------------------------------------------------------------
# Converged Segments
# ----------------------------------------------------------------------------------------------------------------------


@chex.dataclass(kw_only=True)
class ConvergedSegment(Process):

    tag: str = "Segment Convergence"

    # Functions for calculating residuals and solving root-finding problem
    calculate_residuals:    Callable = None

    # Root-finder arguments and parser
    root_finder_args:       List = field(default_factory=list)
    root_finder_kwargs:     dict = None
    results_parser:         Callable[[Any], Tuple["rcf.State", "rcf.System", "rcf.Settings"]] = None

    # Global dynamics variables
    sideslip_angle:         float = 0.0
    temperature_deviation:  float = 0.0

    def __post_init__(self):
        if self.calculate_residuals is None:
            self.calculate_residuals = flight_dynamics_residuals

        if self.results_parser is None:
            self.results_parser = fsolve_results_parser

        self.steps = [
            InitializeSegment(tag=f'Initialize {self.tag}'),
            AnalyzeSegment(tag=f'Analyze {self.tag}'),
        ]

        self._initialize = self.steps[0]
        self._analyze = self.steps[1]

    def unpack_unknowns(self):
        """
        Finds the active control variables and assigns the unknowns to their locations in state.
        """

        state       = self.state
        unknowns    = state.unknowns.pack_array()
        controls    = state.controls
        n_points    = state.numerics.number_of_control_points

        control_idx = 0

        for name, control_var in inspect.getmembers(controls):
            if hasattr(control_var, 'active') and control_var.active:
                values = unknowns[control_idx : control_idx + n_points]     # Extract control values from unknowns
                values = np.reshape(values, (-1, 1))               # Reshape to column vector
                destination = reduce(getattr, control_var.path, state)      # Find destination within state
                destination[control_var.path_indices] = values.flatten()    # Assign to destination in state
                control_idx += n_points

        return

    # Special override of process call to handle root finding, still follows process type flow
    def __call__(self) -> Tuple["rcf.State", "rcf.System", "rcf.Settings"]:

        self.update_details()

        self._initialize.state = self.state
        self._initialize.system = self.system
        self._initialize.settings = self.settings

        state, system, settings = self._initialize()

        # Converge root of residuals

        root_finder = settings.root_finder

        if self.root_finder_kwargs is None:
            # Assume fsolve is the default root finder and that the calculate_residuals function is provided
            self.root_finder_kwargs = {
                'func': self.calculate_residuals,
                'x0': state.unknowns.pack_array(),
                'args': (state, system, settings),
                'xtol': state.numerics.solution_tolerance,
                'maxfev': state.numerics.max_evaluations,
                'epsfcn': state.numerics.step_size,
                'full_output': True
            }

        results = root_finder(*self.root_finder_args, **self.root_finder_kwargs)

        state, system, settings = self.results_parser(results, state, system, settings)

        self._analyze.state = state
        self._analyze.system = system
        self._analyze.settings = settings

        state, system, settings = self._analyze()

        self.state = state
        self.system = system
        self.settings = settings

        return


#-----------------------------------------------------------------------------------------------------------------------
# Optimal Segments
#-----------------------------------------------------------------------------------------------------------------------


@chex.dataclass(kw_only=True)
class OptimalSegment(Process):

    tag:                   str     = 'Optimize Segment'
    optimization_method:    str     = 'SLSQP'
    display_optimization:   bool    = False

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
        settings: "rcf.Settings"):

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


if __name__ == '__main__':
    seg = ConvergedSegment()
    print(seg.details)
    print('Done')
