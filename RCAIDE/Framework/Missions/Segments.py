# RCAIDE/Framework/Missions/Mission.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Jul 2024, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
# IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from typing import Callable, List, Any, Tuple
from dataclasses import dataclass, field

# package imports

import jax
import jax.numpy as jnp
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

# ----------------------------------------------------------------------------------------------------------------------
# Segment Subfunctions
# ----------------------------------------------------------------------------------------------------------------------


@dataclass(kw_only=True)
class InitializeSegment(Process):

    name: str = 'Segment Initialization'

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
            self.append(ProcessStep(name=name, function=function))


@dataclass(kw_only=True)
class AnalyzeSegment(Process):

    name: str = "Segment Iteration"

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
            self.append(ProcessStep(name=name, function=function))


@dataclass(kw_only=True)
class FinalizeSegment(Process):

    name: str = "Mission Finalization"

# ----------------------------------------------------------------------------------------------------------------------
# Converge/Optimize Segment
# ----------------------------------------------------------------------------------------------------------------------


@dataclass(kw_only=True)
class ConvergedSegment(Process):

    name: str = "Segment Convergence"

    calculate_residuals:    Callable = None

    root_finder_args:       List = field(default_factory=list)
    root_finder_kwargs:     dict = None
    results_parser:         Callable[[Any], Tuple["rcf.State", "rcf.System", "rcf.Settings"]] = skip

    initial_unknowns:       np.ndarray  = None

    # Special override of process call to handle root finding, still follows process type flow
    def __call__(self,
                 state: "rcf.State",
                 system: "rcf.System",
                 settings: "rcf.Settings") -> Tuple["rcf.State", "rcf.System", "rcf.Settings"]:

        self.update_details()

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

        return self.results_parser(results)


@dataclass(kw_only=True)
class OptimalSegment(Process):

    name:                   str     = 'Optimize Segment'
    optimization_method:    str     = 'SLSQP'
    display_optimization:   bool    = False

    initialize:             InitializeSegment   = field(default_factory=InitializeSegment)
    analyze:                AnalyzeSegment      = field(default_factory=AnalyzeSegment)
    finalize:               FinalizeSegment     = field(default_factory=FinalizeSegment)

    calculate_objective:    Callable = None
    bounds:                 List[Any] = None

    def _results_parser(self, res):

        self.state.unknowns.unpack_array(res.x)
        self.state.objective.unpack_array(res.fun)

        return self.state, self.settings, self.system

    def __call__(self, *args, **kwargs) -> Tuple["rcf.State", "rcf.System", "rcf.Settings"]:

        self.state, self.system, self.settings = args[0]
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
            options={'disp': self.display_optimization,
                     'maxiter': self.state.numerics.max_evaluations},
            tol=self.state.numerics.solution_tolerance,
        )

        self.state, self.system, self.settings = self._results_parser(res)

        self.state, self.system, self.settings = self.finalize(self.state, self.system, self.settings)

        return self.state, self.system, self.settings


def energy_use(
        state: "rcf.State",
        system: "rcf.System",
        settings: "rcf.Settings"):

    energy_start    = state.energy.total_energy[0]
    energy_end      = state.energy.total_energy[-1]
    energy_used     = energy_end - energy_start

    return energy_used[0]


@dataclass(kw_only=True)
class EnergyOptimalCruise(OptimalSegment):

    name: str = 'Energy Optimal Cruise'

    altitude: float = 0.0
    distance: float = 0.0

    calculate_objective: Callable = energy_use


@dataclass(kw_only=True)
class EnergyOptimalAltitudeChange(OptimalSegment):

    name: str = 'Energy Optimal Altitude Change'

    altitude_start: float = 0.0
    altitude_end:   float = 0.0

    calculate_objective: Callable = energy_use

    def __post_init__(self):
        start_check = lambda: self.state.frames.inertial.position_vector[0, 2]
        end_check = lambda: self.state.frames.inertial.position_vector[-1, 2]
        self.bounds = [NonlinearConstraint(start_check, lb=self.altitude_start, ub=self.altitude_start),
                       NonlinearConstraint(end_check, lb=self.altitude_end, ub=self.altitude_end)]



if __name__ == '__main__':
    seg = ConvergedSegment()
    print(seg.details)
    print('Done')
