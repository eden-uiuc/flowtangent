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

from jax import jit, grad, value_and_grad

import numpy as np
from scipy.optimize import fsolve, minimize

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
            # Step Name                         Step Functions
            ("Expand State",                    expand_state),
            ("Initialize Time",                 initialize_time),
            ("Initialize Mass",                 initialize_mass),
            ("Initialize Energy",               initialize_energy),
            ("Initialize Inertial Position",    initialize_inertial_position),
            ("Initialize Planetary Position",   initialize_planetary_position)
        ]

        for name, function in default_steps:
            self.append(ProcessStep(name=name, function=function))


@dataclass(kw_only=True)
class AnalyzeSegment(Process):

    name: str = "Segment Iteration"

    def __post_init__(self):

        default_steps = [
            ("Update Time Differentials",   update_time_differentials),
            ("Update Acceleration",         update_acceleration),
            ("Update Angular Acceleration", update_angular_acceleration),
            ("Update Altitude",             update_altitude),
            ("Update Gravity",              skip),
            ("Update Freestream",           update_freestream),
            ("Update Orientations",         update_orientations),
            ("Update Energy",               skip),
            ("Update Aerodynamics",         skip),
            ("Update Stability",            skip),
            ("Update Mass",                 skip),
            ("Update Forces",               update_forces),
            ("Update Moments",              update_moments),
            ("Update Planetary Position",   skip)
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
    results_parser:         Callable[[Any], Tuple[rcf.State, rcf.Settings, rcf.System]] = skip

    initial_unknowns:       np.ndarray  = None

    # Special override of process call to handle root finding, still follows process type flow
    def __call__(self,
                 state: rcf.State,
                 settings: rcf.Settings,
                 system: rcf.System) -> Tuple[rcf.State, rcf.Settings, rcf.System]:

        self.update_details()

        # Converge root of residuals

        root_finder = Settings.root_finder

        if self.root_finder_kwargs is None:
            # Assume fsolve is the default root finder and that the calculate_residuals function is provided
            self.root_finder_kwargs = {
                'func': self.calculate_residuals,
                'x0': state.unknowns.pack_array(),
                'args': (state, settings, system),
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

    initalize:              InitializeSegment   = field(default_factory=InitializeSegment)
    update:                 AnalyzeSegment      = field(default_factory=AnalyzeSegment)
    finalize:               FinalizeSegment     = field(default_factory=FinalizeSegment)

    calculate_objective:    Callable = None
    bounds:                 List[Tuple[np.array, np.array]] = None

    def _results_parser(self, res):

        self.state.unknowns.unpack_array(res.x)
        self.state.objective.unpack_array(res.fun)

        return self.state, self.settings, self.system

    def __call__(self) -> Tuple[rcf.State, rcf.Settings, rcf.System]:

        self.update_details()

        self.state, self.system, self.settings = self.initalize(self.state, self.settings, self.system)

        def _obj(U):
            self.state.unknowns.unpack_array(U)
            self.state, self.system, self.settings = self.update(self.state, self.system, self.settings)
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


if __name__ == '__main__':
    seg = ConvergedSegment()
    print(seg.details)
    print('Done')
