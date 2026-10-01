# ruff: noqa
# flowtangent/Framework/Missions/Mission.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Jul 2024, FlowTangent Team

# ----------------------------------------------------------------------------------------------------------------------
# IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

import sys
import threading
import time
import timeit
from dataclasses import replace

import equinox as eqx

# package imports
import jax
import jax
import jax.numpy as jnp

from flowtangent.core._processes import null_step
from flowtangent.core._state_data._controls import Control, Residual
from flowtangent.framework import Process, ProcessStep
from flowtangent.framework.simulation.initialize import *  # noqa: F403
from flowtangent.framework.simulation.update import *  # noqa: F403
from flowtangent.utils import field, scan_for_invalid_JAX_types

from . import profiles as pf

jax.config.update("jax_enable_x64", True)

# ----------------------------------------------------------------------------------------------------------------------
# Mission Spinner
# ----------------------------------------------------------------------------------------------------------------------


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
            sys.stdout.write(f"\r{self.message} Done.    \n")
            sys.stdout.flush()


# ----------------------------------------------------------------------------------------------------------------------
# Segment Subfunctions
# ----------------------------------------------------------------------------------------------------------------------


def _activate_control(control: str | Control, state):

    if isinstance(control, str):
        control_name = control.replace(" ", "_").lower()

        if control_name not in state.controls.__dataclass_fields__:
            # It's a custom control:
            new_var = Variable(name=control, _active=True)
            new_controls = state.controls.add_control_variable(new_var)
        else:
            # It's a pre-existing control: grab the existing one, activate it, and replace it
            existing_var = getattr(state.controls, control_name)
            active_var = update(existing_var, "active", True)

            # Use getattr to map the path
            new_controls = update(state.controls, lambda p: getattr(p, control_name), active_var)

    elif isinstance(control, Control):
        active_var = update(control, "active", True)
        new_controls = state.controls.add_control_variable(active_var)

    return update(state, "controls", new_controls)


def _activate_residual(res: str | Residual, state):

    if isinstance(res, str) and res in NamedResidual:
        current_residual = getattr(state.dynamics, res)
        active_residual = replace(current_residual, active=True)

        # Safely inject it using getattr path tracing
        new_dynamics = update(state.dynamics, lambda d: getattr(d, res), active_residual)
    elif isinstance(res, Residual):
        active_res = replace(res, active=True)
        new_dynamics = state.dynamics.add_substate(active_res)

    return update(state, "dynamics", new_dynamics).expand_rows(state.numerics.number_of_control_points)


# ----------------------------------------------------------------------------------------------------------------------
# Initialize Segment
# ----------------------------------------------------------------------------------------------------------------------


def _initialization_steps():
    return (
        ProcessStep(name="State", function=expand_state),
        ProcessStep(name="Time", function=initialize_time),
        ProcessStep(name="Mass", function=initialize_mass),
        ProcessStep(name="Energy", function=initialize_energy),
        ProcessStep(name="Inertial Position", function=initialize_inertial_position),
        ProcessStep(name="Planetary Position", function=initialize_planetary_position),
        ProcessStep(name="Analyses", function=null_step),
    )


class InitializeSegment(Process):
    name: str = "Segment Initialization"

    active_controls: tuple[str | Control, ...] = field(tuple)
    active_residuals: tuple[NamedResidual, ...] = field(tuple)

    controls_initial_guess: tuple[jax.Array | float, ...] = (0.0, 0.0)

    steps: tuple[ProcessStep, ...] = field(_initialization_steps)

    def __call__(self, state: State, system: System, settings: Settings, validate_controls=False):

        if settings.DEBUG_MODE:
            scan_for_invalid_JAX_types(state, f"Pre-{self.name} State")
            scan_for_invalid_JAX_types(system, f"Pre-{self.name} System")

        current_state = state

        for var in self.active_controls:
            current_state = _activate_control(var, current_state)

        # Set up static routing for active controls
        active_controls = current_state.controls.get_active_controls()
        if settings.analysis.energy.use_network_controls:
            for network in system.energy_networks:
                active_controls += network.controls  # type: ignore
        routing_table = tuple((var.path, var.path_indices) for var in active_controls)
        new_controls = replace(current_state.controls, active_routing_table=routing_table)
        current_state = update(current_state, "controls", new_controls)

        active_residuals = self.active_residuals
        if settings.analysis.energy.use_network_controls:
            for network in system.energy_networks:
                active_residuals += network.residuals  # type: ignore
        for res in self.active_residuals:
            current_state = _activate_residual(res, current_state)

        n_cp = int(current_state.numerics.number_of_control_points)

        if self.controls_initial_guess is not None and len(self.controls_initial_guess) > 0:
            new_unknowns = jnp.concatenate([jnp.full((n_cp,), v) for v in self.controls_initial_guess])
        else:
            new_unknowns = jnp.zeros((n_cp * len(self.active_controls)))

        new_residuals = jnp.zeros((n_cp, len(self.active_residuals)))

        current_state = eqx.tree_at(
            lambda s: (s.solver.unknowns, s.solver.residuals), current_state, (new_unknowns, new_residuals)
        )

        if validate_controls:
            assert current_state.check_controls(verbose=False), (
                f"During initialization of {self.name} the number of active controls "
                "did not match the number of active residuals.\n"
            )

        current_state = current_state.unpack_unknowns(current_state.solver.unknowns)

        current_state, system, settings = super().__call__(current_state, system, settings)

        return current_state, system, settings


# ----------------------------------------------------------------------------------------------------------------------
# Analyze Segment
# ----------------------------------------------------------------------------------------------------------------------


def _default_analyses():
    return (
        ProcessStep(name="Time Differentials", function=update_time_differentials),
        ProcessStep(name="Acceleration", function=update_acceleration),
        ProcessStep(name="Angular Acceleration", function=update_angular_acceleration),
        ProcessStep(name="Freestream", function=update_freestream),
        ProcessStep(name="Orientations", function=update_orientations),
        ProcessStep(name="Energy", function=null_step),
        ProcessStep(name="Aerodynamics", function=null_step),
        ProcessStep(name="Stability", function=null_step),
        ProcessStep(name="Mass", function=update_mass_and_weight),
        ProcessStep(name="Forces", function=update_forces),
        ProcessStep(name="Moments", function=update_moments),
        ProcessStep(name="Planetary Position", function=update_planetary_position),
        ProcessStep(name="Calculate Residuals", function=flight_dynamics_residuals),
    )


class AnalyzeSegment(Process):
    name: str = field("Segment Analysis", static=True)

    steps: tuple[ProcessStep, ...] = field(_default_analyses)


def find_circular_references(obj, path="root", visited=None):
    if visited is None:
        visited = set()

    # Skip basic types and arrays (they don't hold other objects)
    if obj is None or isinstance(obj, (int, float, str, bool, tuple, frozenset)):
        return
    if type(obj).__name__ in ("ndarray", "ArrayImpl", "DynamicJaxprFlowTangentr"):
        return

    obj_id = id(obj)

    # If we've seen this exact object ID in this branch, we found the loop
    if obj_id in visited:
        print("CIRCULARITY FOUND:")
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
    elif hasattr(obj, "__dict__"):
        for k, v in vars(obj).items():
            if find_circular_references(v, f"{path}.{k}", visited.copy()):
                return True

    return False


# ----------------------------------------------------------------------------------------------------------------------
# Iterate Segment
# ----------------------------------------------------------------------------------------------------------------------


class IterateSegment(Process):
    name: str = field("Segment Convergence", static=True)
    analyze: Process = field(AnalyzeSegment)

    def _get_residuals(self, unknowns, state: "State", system: "System", settings: "Settings"):

        current_state = state.update_controls()
        current_state, _, _ = self.analyze(current_state, system, settings)
        current_state = current_state.pack_residuals()

        return current_state.solver.residuals

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
            maxiter=state.numerics.max_evaluations,
        )

        return root.run(x0, state, system, settings)

    def __call__(self, state: State, system: System, settings: Settings):

        root_finder = settings.numerical.root_finder

        with Spinner(enabled=not settings.DEBUG_MODE):
            match root_finder:
                case cls if cls == "GaussNewton":
                    x0 = state.solver.unknowns
                    unknowns, opt_state = self._run_gauss_newton_solver(x0, state, system, settings)

                    if settings.DEBUG_MODE:
                        unknowns.block_until_ready()

                        print("\n" + "=" * 30)
                        print("JAXOPT SOLVER DIAGNOSTICS")
                        print("=" * 30)
                        # JAX arrays need to be cast or formatted to print cleanly
                        print(f"Iterations taken: {opt_state.iter_num}")
                        print(f"Final Error/Residual: {opt_state.error: .3e}")

                        # Depending on the specific JAXopt solver, you might also have:
                        if hasattr(opt_state, "stepsize"):
                            print(f"Final Stepsize: {opt_state.stepsize}")
                        if hasattr(opt_state, "value"):
                            print(f"Objective Value: {opt_state.value: .3e}")
                        print("=" * 30 + "\n")

                    current_state = update(state, "solver.unknowns", unknowns)
                    current_state = current_state.unpack_unknowns()

                    return self.analyze(current_state, system, settings)

                case cls if cls == "ScipyRootFinding":
                    x0 = state.solver.unknowns

                    root = ScipyRootFinding(
                        method="hybr",
                        optimality_fun=self._get_residuals,
                        tol=state.time.solution_tolerance,
                        jit=True,
                    )
                    if settings.DEBUG_MODE:
                        if any(
                            [
                                find_circular_references(state, path="state"),
                                find_circular_references(system, path="system"),
                                find_circular_references(settings, path="settings"),
                            ]
                        ):
                            raise RecursionError("Circularity found in mission data structures. Terminating mission.")

                    t0 = timeit.default_timer()

                    unknowns, _ = root.run(x0, state, system, settings)

                    unknowns.block_until_ready()

                    t1 = timeit.default_timer()

                    print(f"--- Hybrid JAX/SciPy Solver: {t1 - t0:.6f} seconds ---")

                    current_state = update(state, "solver.unknowns", unknowns)
                    current_state = current_state.unpack_unknowns(unknowns)

                    return self.analyze(current_state, system, settings)

                case cls if cls == "Broyden":
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
                    current_state = update(state, "solver.unknowns", unknowns)
                    current_state = current_state.unpack_unknowns(unknowns)

                    return self.analyze(current_state, system, settings)

                case _:
                    return state, system, settings

    def run_with_history(self, state, system, settings):
        conv_state, conv_system, conv_settings = self(state, system, settings)
        return self.analyze.run_with_history(conv_state, conv_system, conv_settings)


# ----------------------------------------------------------------------------------------------------------------------
# Finalize Segment
# ----------------------------------------------------------------------------------------------------------------------


def _reset_controls_and_residuals(
    state: State,
    system: System,
    settings: Settings,
):
    current_state = state

    def _turn_off_Variable(node):
        if isinstance(node, Control):
            return update(node, "active", False)
        return node

    def _turn_off_residual(node):
        if isinstance(node, Residual):
            return replace(node, active=False)
        return node

    new_controls = jax.tree_util.tree_map(
        _turn_off_control, current_state.controls, is_leaf=lambda x: isinstance(x, Control)
    )

    new_dynamics = jax.tree_util.tree_map(
        _turn_off_residual, current_state.dynamics, is_leaf=lambda x: isinstance(x, Residual)
    )

    current_state = update(current_state, lambda s: (s.controls, s.dynamics), (new_controls, new_dynamics))

    return current_state, system, settings


def _default_finalize():
    return (ProcessStep(name="Deactivate Controls & Residuals", function=_reset_controls_and_residuals),)


class FinalizeSegment(Process):
    name: str = field("Segment Finalization", static=True)
    steps: tuple[ProcessStep, ...] = field(_default_finalize)


# ----------------------------------------------------------------------------------------------------------------------
# Converged Segments
# ----------------------------------------------------------------------------------------------------------------------


class Segment(Process):
    name: str = field("Segment", static=True)

    # Pass-through configuration for InitializeSegment
    active_controls: tuple[str | Control, ...] = field(tuple)
    active_residuals: tuple[NamedResidual, ...] = field(tuple)
    controls_initial_guess: tuple[jax.Array | float, ...] = (0.0, 0.0)

    course_profile: pf.CourseProfile = field(pf.ConstantCourse)
    position_profile: pf.PositionProfile = field(pf.ConstantAltitude)
    speed_profile: pf.SpeedProfile = field(pf.ConstantSpeed)
    velocity_profile: pf.VelocityProfile = field(pf.ConstantAltitudeChangeRate)
    duration_profile: pf.DurationProfile = field(pf.FixedDistance)

    # Global dynamics variables
    sideslip_angle: float = 0.0
    temperature_deviation: float = 0.0
    true_course: float = 0.0

    # Start with an empty tuple. We will populate it securely in __post_init__
    steps: tuple = field(tuple)

    def __post_init__(self):
        # Only build the default steps if the user didn't explicitly provide custom ones
        if len(self.steps) == 0:
            # 1. Build the steps, passing the controls configuration directly into InitializeSegment
            init_step = InitializeSegment(
                name=f"{self.name} Initialization",
                active_controls=self.active_controls,
                active_residuals=self.active_residuals,
                controls_initial_guess=self.controls_initial_guess,
            )

            # Add profile initialization
            init_step = eqx.tree_at(
                lambda i: i.steps,
                init_step,
                init_step.steps
                + (
                    self.course_profile,
                    self.position_profile,
                    self.speed_profile,
                    self.velocity_profile,
                    self.duration_profile,
                ),
            )

            iter_step = IterateSegment(name=f"{self.name} Iteration")
            fin_step = FinalizeSegment(name=f"{self.name} Finalization")

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

        state = update(state, "frames.planet.true_course", jnp.array([self.true_course]))

        if settings.DEBUG_MODE:
            for step in self.analyze.steps:
                if isinstance(step, ProcessStep) and not isinstance(step, Process) and step.function is null_step:
                    print(f"Warning: Skipping {step.name} analysis due to missing function.")

        return super().__call__(state, system, settings)


# ----------------------------------------------------------------------------------------------------------------------
# Fixed/Unconverged Segments
# ----------------------------------------------------------------------------------------------------------------------


class FixedSegment(Segment):
    name: str = field("Fixed Segment", static=True)

    def __post_init__(self):
        # Only build the default steps if the user didn't explicitly provide custom ones
        if len(self.steps) == 0:
            # 1. Build the steps, passing the controls configuration directly into InitializeSegment
            init_step = InitializeSegment(
                name=f"{self.name} Initializations",
                active_controls=self.active_controls,
                active_residuals=self.active_residuals,
                controls_initial_guess=self.controls_initial_guess,
            )

            # Add profile initialization
            init_step = eqx.tree_at(
                lambda i: i.steps,
                init_step,
                init_step.steps
                + (
                    self.course_profile,
                    self.position_profile,
                    self.speed_profile,
                    self.velocity_profile,
                    self.duration_profile,
                ),
            )

            iter_step = AnalyzeSegment(
                name=f"{self.name} Analysis",
                steps=_default_analyses()[:-1],  # Skip Residual Calculation
            )
            fin_step = FinalizeSegment(name=f"{self.name} Finalization")

            # 2. Safely lock them into the frozen object
            object.__setattr__(self, "steps", (init_step, iter_step, fin_step))

    @property
    def analyze(self) -> AnalyzeSegment:
        return self.steps[1]


# ----------------------------------------------------------------------------------------------------------------------
# Optimal Segments
# ----------------------------------------------------------------------------------------------------------------------


# @chex.dataclass(kw_only=True)
# class OptimalSegment(Process):

#     name:                    str    = 'Optimize Segment'
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

#     def __call__(self, *args, **kwargs) -> Tuple[State, System, Settings]:

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
#         state: State,
#         system: System,
#         settings: Settings
# ):

#     energy_start    = state.energy.total_energy[0]
#     energy_end      = state.energy.total_energy[-1]
#     energy_used     = energy_end - energy_start

#     return energy_used[0]


# @chex.dataclass(kw_only=True)
# class EnergyOptimalCruise(OptimalSegment):

#     name: str = 'Energy Optimal Cruise'

#     altitude: float = 0.0
#     distance: float = 0.0

#     calculate_objective: Callable = energy_use

#     def __post_init__(self):
#         distance_check = lambda x: self.state.frames.inertial.position_vector[-1, 0]
#         self.bounds = [(-np.pi/12, np.pi/12), (0., 1.)]
#         self.constraints = [NonlinearConstraint(distance_check, lb=self.distance, ub=self.distance)]


# @chex.dataclass(kw_only=True)
# class EnergyOptimalAltitudeChange(OptimalSegment):

#     name: str = 'Energy Optimal Altitude Change'

#     altitude_start: float = 0.0
#     altitude_end:   float = 0.0

#     calculate_objective: Callable = energy_use

#     def __post_init__(self):
#         start_check = lambda x: self.state.frames.inertial.position_vector[0, 2]
#         end_check = lambda x: self.state.frames.inertial.position_vector[-1, 2]
#         self.bounds = [(-np.pi/4, np.pi/4), (0., 1.)]
#         self.constraints = [NonlinearConstraint(start_check, lb=self.altitude_start, ub=self.altitude_start),
#                             NonlinearConstraint(end_check, lb=self.altitude_end, ub=self.altitude_end)]
