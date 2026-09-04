import os
import sys

def numerical_environment():
    # 1. JAX Memory/Precision Config (Safe everywhere)
    os.environ["JAX_ENABLE_X64"] = "True"
    os.environ['OPENMDAO_REPORTS'] = '0'
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    os.environ["JAX_PERSISTENT_CACHE_DISABLE"] = "1"
    os.environ["JAX_PLATFORM_NAME"] = "gpu"
    
    # 2. NUMA / Hardware Auto-Detection
    if sys.platform == "linux":
        # A simple heuristic: if you have a massive amount of cores, 
        # it's likely the Threadripper workstation.
        cpu_count = os.cpu_count() or 1
        if cpu_count > 16:  # Adjust threshold based on your hardware
            try:
                # Bind to the first 16 cores (Node 0) to prevent cross-NUMA memory latency
                node_0_cores = set(range(16))
                os.sched_setaffinity(0, node_0_cores)
                
                # Tell OpenMP to respect this boundary
                os.environ["OMP_PROC_BIND"] = "true"
                os.environ["OMP_PLACES"] = "cores"
                print(f"Hardware Config: NUMA affinity set to Node 0 (16 cores).")
            except Exception as e:
                print(f"Hardware Config Warning: Could not set CPU affinity: {e}")

    cache_path = os.path.expanduser("~/.eden_trace/jax_cache")
    os.makedirs(cache_path, exist_ok=True)
    os.environ["JAX_COMPILATION_CACHE_DIR"] = cache_path

numerical_environment()

import json
import time
import tracemalloc
import gc
import numpy as np
import matplotlib.pyplot as plt
import openmdao.api as om
import pycycle.api as pyc
import scipy.sparse

import jax
import jax.numpy as jnp
import equinox as eqx

from tqdm import tqdm
from pathlib import Path
from dataclasses import replace

import warnings
from openmdao.utils.om_warnings import OpenMDAOWarning, SolverWarning
# Suppress the PyCycle negative root warning during Newton steps
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in sqrt')

# Suppress the OpenMDAO monolithic matrix warning which we are intentionally triggering
warnings.filterwarnings('ignore', category=OpenMDAOWarning, message='The top level group has a nonlinear solver')
warnings.filterwarnings('ignore', category=SolverWarning)

# Import OpenMDAO and FlowTangent models
from simple_turbojet import Turbojet
from turbojet_validation import system_setup as ft_turbojet

from eden_trace.utils import DataPath, configure_environment
from eden_trace.framework import State, Settings, Process
from eden_trace.framework.settings import NumericalSettings, JacobianSettings, JacobianMap
from eden_trace.framework.analyses.batched import BatchedAnalysis
from eden_trace.framework.analyses.energy.jets import build_turbojet_design, build_turbojet_performance, JetSettings
from eden_trace.framework.simulation.initialize import initialize_energy
from eden_trace.framework.simulation.update import update_freestream

from eden_trace.library import units
from eden_trace.library.atmospheres import USStandard1976
from eden_trace.library.components.energy.jets.classes import TurbojetOpPoint

pact_primal_calls = 0
pact_vjp_calls = 0
opaque_fd_primal_calls = 0
opaque_ad_primal_calls = 0
opaque_ad_jac_calls = 0

test_dir = Path(__file__).resolve().parent

# ==============================================================================
# MAUD MONOLITHIC BENCHMARK
# ==============================================================================

class MAUD_Monolithic(pyc.MPCycle):
    """
    Scalable version of PyCycle's MPTurbojet. 
    Dynamically generates N off-design points to benchmark O(N^2) scaling.
    """
    def initialize(self):
        self.options.declare('N_points', default=2, types=int)
        super().initialize()

    def setup(self):
        N = self.options['N_points']
        
        # 1. Create design instance of model (Sea-Level Static)
        self.pyc_add_pnt('DESIGN', Turbojet())

        self.set_input_defaults('DESIGN.Nmech', 8070.0, units='rpm')
        self.set_input_defaults('DESIGN.inlet.MN', 0.60)
        self.set_input_defaults('DESIGN.comp.MN', 0.020)
        self.set_input_defaults('DESIGN.burner.MN', 0.020)
        self.set_input_defaults('DESIGN.turb.MN', 0.4)

        self.pyc_add_cycle_param('burner.dPqP', 0.03)
        self.pyc_add_cycle_param('nozz.Cv', 0.99)

        # 2. Define N off-design conditions
        # We duplicate a known convergent point to ensure the automated sweep never 
        # fails on a bad initial guess during the forward pass.
        self.od_pts = [f'OD{i}' for i in range(N)]
        
        for pt in self.od_pts:
            self.pyc_add_pnt(pt, Turbojet(design=False))
            self.set_input_defaults(pt+'.fc.MN', 0.2)
            self.set_input_defaults(pt+'.fc.alt', 5000.0, units='ft')
            self.set_input_defaults(pt+'.balance.Fn_target', 8000.0, units='lbf')

        # 3. Establish the Arrowhead Matrix Coupling
        self.pyc_use_default_des_od_conns()
        self.pyc_connect_des_od('nozz.Throat:stat:area', 'balance.rhs:W')

        super().setup()

def get_jac_memory(system):
    """Recursively hunts for the AssembledJacobian and extracts its exact memory footprint."""
    if getattr(system, '_assembled_jac', None) is not None:
        matrix = system._assembled_jac.get_dr_do_matrix()
        if matrix is not None:
            # Safely check if it is a scipy sparse matrix
            if scipy.sparse.issparse(matrix):
                mem_mb = (matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes) / (1024*1024)
                return mem_mb, matrix.shape, 'sparse'
            # Otherwise, it is a dense numpy array
            else:
                return matrix.nbytes / (1024*1024), matrix.shape, 'dense'
                
    # Recurse down the model tree if not found here
    for subsys in system._subsystems_myproc:
        result = get_jac_memory(subsys)
        if result: 
            return result
            
    return None

def run_monolithic_benchmark(N_points):
    """
    Builds the PyCycle problem, converges it, and times the global adjoint solve.
    Tracks Setup, Compile (0.0s), and Execution (Forward + Adjoint) times.
    """
    gc.collect()
    tracemalloc.start()
    
    # =======================================================
    # PHASE 1: SETUP
    # =======================================================
    t_setup_start = time.perf_counter()
    
    prob = om.Problem()
    
    # Add the scalable multi-point cycle
    mp_turbojet = prob.model.add_subsystem('mp_turbojet', MAUD_Monolithic(N_points=N_points), promotes=['*'])
    mp_turbojet.options['assembled_jac_type'] = 'dense'
    
    # Add Objective: Average TSFC across all N points
    eq_str = 'avg_tsfc = (' + ' + '.join([f'tsfc_{i}' for i in range(N_points)]) + f') / {N_points}'
    prob.model.add_subsystem('objective', om.ExecComp(eq_str, units='lbm/h/lbf'), promotes_outputs=['avg_tsfc'])
    
    for i in range(N_points):
        prob.model.connect(f'OD{i}.perf.TSFC', f'objective.tsfc_{i}')
        
    prob.model.add_objective('avg_tsfc')
    prob.model.add_design_var('DESIGN.comp.PR', lower=10.0, upper=20.0)

    # Force the monolithic matrix assembly for the global adjoint
    prob.model.linear_solver = om.DirectSolver(assemble_jac=True)
    prob.model.options['assembled_jac_type'] = 'dense'
    
    prob.setup(check=False, mode='rev')

    # --- Set Initial Guesses ---
    prob.set_val('DESIGN.fc.alt', 0, units='ft')
    prob.set_val('DESIGN.fc.MN', 0.000001)
    prob.set_val('DESIGN.balance.Fn_target', 11800.0, units='lbf')
    prob.set_val('DESIGN.balance.T4_target', 2370.0, units='degR')
    prob.set_val('DESIGN.comp.PR', 13.5)
    prob.set_val('DESIGN.comp.eff', 0.83)
    prob.set_val('DESIGN.turb.eff', 0.86)

    prob['DESIGN.balance.FAR'] = 0.0175506829934
    prob['DESIGN.balance.W'] = 168.453135137
    prob['DESIGN.balance.turb_PR'] = 4.46138725662
    prob['DESIGN.fc.balance.Pt'] = 14.6955113159
    prob['DESIGN.fc.balance.Tt'] = 518.665288153

    # OFF-DESIGN Points
    for pt in mp_turbojet.od_pts:
        prob[pt+'.balance.W'] = 166.073
        prob[pt+'.balance.FAR'] = 0.01680
        prob[pt+'.balance.Nmech'] = 8197.38
        prob[pt+'.fc.balance.Pt'] = 15.703
        prob[pt+'.fc.balance.Tt'] = 558.31
        prob[pt+'.turb.PR'] = 4.6690

    prob.set_solver_print(level=-1)
    
    t_setup_end = time.perf_counter()

    # =======================================================
    # PHASE 2: COMPILATION (Eager Python = 0.0s)
    # =======================================================
    t_compile_start = time.perf_counter()
    # No AOT compilation step for OpenMDAO
    t_compile_end = time.perf_counter()

    # =======================================================
    # PHASE 3: EXECUTION (Forward Pass + Backward Adjoint)
    # =======================================================
    t_exec_start = time.perf_counter()
    
    # 1. Forward pass (Newton solvers converge the non-linear states)
    prob.run_model()
    
    # 2. Backward pass (Adjoint solves the linear system for the derivatives)
    totals = prob.compute_totals(of=['avg_tsfc'], wrt=['DESIGN.comp.PR'])
    
    t_exec_end = time.perf_counter()

    # =======================================================
    # METRICS EXTRACTION
    # =======================================================
    jac_info = get_jac_memory(prob.model)
    if jac_info:
        jac_mem_mb, shape, fmt = jac_info
    else:
        jac_mem_mb = 0.0
    
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    setup_time = t_setup_end - t_setup_start
    compile_time = t_compile_end - t_compile_start
    exec_time = t_exec_end - t_exec_start
    total_mem_mb = peak_mem / (1024 * 1024)

    mean_tsfc = prob.get_val('avg_tsfc')[0]
    gradient = totals['avg_tsfc', 'DESIGN.comp.PR'][0][0]
    
    return total_mem_mb, jac_mem_mb, setup_time, compile_time, exec_time, mean_tsfc, gradient, 1 + N_points, 1 + N_points

# ==============================================================================
# HYBRID PACT BENCHMARK
# ==============================================================================

COUPLED_VARS_DES = [
    'comp.s_PR', 'comp.s_Wc', 'comp.s_eff', 'comp.s_Nc',
    'turb.s_PR', 'turb.s_Wp', 'turb.s_eff', 'turb.s_Np',
    'inlet.Fl_O:stat:area',
    'comp.Fl_O:stat:area',
    'burner.Fl_O:stat:area',
    'turb.Fl_O:stat:area',
    'nozz.Throat:stat:area'
]

COUPLED_VARS_OD = [
    'comp.s_PR', 'comp.s_Wc', 'comp.s_eff', 'comp.s_Nc',
    'turb.s_PR', 'turb.s_Wp', 'turb.s_eff', 'turb.s_Np',
    'inlet.area',
    'comp.area',
    'burner.area',
    'turb.area',
    'balance.rhs:W'
]


# 1. NODE 1: DESIGN SIZING POINT (Runs ONCE) -----------------------------------

prob_des = om.Problem()
prob_des.model.add_subsystem('des', Turbojet(design=True), promotes=['*'])
prob_des.model.linear_solver = om.DirectSolver(assemble_jac=True)
prob_des.model.options['assembled_jac_type'] = 'dense'

prob_des.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
prob_des.model.nonlinear_solver.options['maxiter'] = 50
prob_des.model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS(bound_enforcement='scalar')
prob_des.model.nonlinear_solver.options['err_on_non_converge'] = False

prob_des.setup(check=False, mode='rev')
prob_des.set_solver_print(level=-1)

prob_des.set_val('burner.dPqP', 0.03)
prob_des.set_val('nozz.Cv', 0.99)
prob_des.set_val('Nmech', 8070.0, units='rpm')
prob_des.set_val('inlet.MN', 0.60)
prob_des.set_val('comp.MN', 0.020)
prob_des.set_val('burner.MN', 0.020)
prob_des.set_val('turb.MN', 0.4)

# Set Design Guesses
prob_des.set_val('fc.alt', 0, units='ft')
prob_des.set_val('fc.MN', 0.000001)
prob_des.set_val('balance.Fn_target', 11800.0, units='lbf')
prob_des.set_val('balance.T4_target', 2370.0, units='degR')
prob_des.set_val('comp.eff', 0.83)
prob_des.set_val('turb.eff', 0.86)

def des_primal_np(pr_arr):
    global pact_primal_calls
    pact_primal_calls += 1

    prob_des.set_val('comp.PR', pr_arr.item())

    prob_des.set_val('balance.FAR', 0.0175506829934)
    prob_des.set_val('balance.W', 168.453135137)
    prob_des.set_val('balance.turb_PR', 4.46138725662)
    prob_des.set_val('fc.balance.Pt', 14.6955113159)
    prob_des.set_val('fc.balance.Tt', 518.665288153)

    prob_des.run_model()
    # Output all 9 coupling variables
    return np.array([prob_des.get_val(var)[0] for var in COUPLED_VARS_DES], dtype=np.float64)

def des_vjp_np(pr_arr, y_bar):
    global pact_vjp_calls
    pact_vjp_calls += 1
    prob_des.set_val('comp.PR', pr_arr.item())
    
    # RESTORE STATE BEFORE COMPUTING DERIVATIVES
    prob_des.set_val('fc.alt', 0, units='ft')
    prob_des.set_val('fc.MN', 0.000001)
    prob_des.set_val('balance.Fn_target', 11800.0, units='lbf')
    prob_des.set_val('balance.T4_target', 2370.0, units='degR')
    prob_des.set_val('comp.eff', 0.83)
    prob_des.set_val('turb.eff', 0.86)
    prob_des.set_val('balance.FAR', 0.0175506829934)
    prob_des.set_val('balance.W', 168.453135137)
    prob_des.set_val('balance.turb_PR', 4.46138725662)
    prob_des.set_val('fc.balance.Pt', 14.6955113159)
    prob_des.set_val('fc.balance.Tt', 518.665288153)
    
    prob_des.run_model() 
    
    # Explicitly request nested dict format
    J_dict = prob_des.compute_totals(of=COUPLED_VARS_DES, wrt=['comp.PR'], return_format='dict')
    
    grad = 0.0
    for i, var in enumerate(COUPLED_VARS_DES):
        # Nested dictionary lookup using exact strings
        grad += J_dict[var]['comp.PR'][0][0] * y_bar[i]
        
    return np.array([grad], dtype=np.float64)

@jax.custom_vjp
def design_node(comp_PR):
    shape = jax.ShapeDtypeStruct((13,), jnp.float64) # <--- Now size 9
    return jax.pure_callback(des_primal_np, shape, comp_PR, vmap_method="sequential")

def des_fwd(comp_PR):
    return design_node(comp_PR), comp_PR

def des_bwd(res, y_bar):
    comp_PR, = res
    shape = jax.ShapeDtypeStruct((1,), jnp.float64)
    return (jax.pure_callback(des_vjp_np, shape, comp_PR, y_bar, vmap_method="sequential"),)

design_node.defvjp(des_fwd, des_bwd)

# 2. NODE 2: OFF-DESIGN OPERATING POINT (Vmapped across N conditions) ----------

prob_od = om.Problem()
prob_od.model.add_subsystem('od', Turbojet(design=False), promotes=['*'])
prob_od.model.linear_solver = om.DirectSolver(assemble_jac=True)

prob_od.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
prob_od.model.nonlinear_solver.options['maxiter'] = 50
prob_od.model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS(bound_enforcement='scalar')
prob_od.model.nonlinear_solver.options['err_on_non_converge'] = False
prob_od.model.nonlinear_solver.options['maxiter'] = 30

prob_od.model.nonlinear_solver.add_recorder(om.SqliteRecorder(test_dir / "solver_errors.sql"))
prob_od.model.nonlinear_solver.recording_options['record_abs_error'] = True
prob_od.model.nonlinear_solver.recording_options['record_rel_error'] = True
prob_od.model.nonlinear_solver.linesearch.options['iprint'] = -1

prob_od.setup(check=False, mode='rev')
prob_od.set_solver_print(level=-1)

prob_od.set_val('balance.W', 166.073)
prob_od.set_val('balance.FAR', 0.01680)
prob_od.set_val('balance.Nmech', 8197.38)
prob_od.set_val('fc.balance.Pt', 15.703)
prob_od.set_val('fc.balance.Tt', 558.31)
prob_od.set_val('turb.PR', 4.6690)

def od_primal_np(inputs, update_count=True):
    if update_count:
        global pact_primal_calls
        pact_primal_calls += 1
    prob_od.set_val('burner.dPqP', 0.03)
    prob_od.set_val('nozz.Cv', 0.99)

    coupling_vals = inputs[:13]
    alt, mn, fn_target = inputs[13:]
    
    for var, val in zip(COUPLED_VARS_OD, coupling_vals):
        prob_od.set_val(var, float(val))
        
    prob_od.set_val('fc.alt', float(alt), units='ft')
    prob_od.set_val('fc.MN', float(mn))
    prob_od.set_val('balance.Fn_target', float(fn_target), units='lbf')
    
    # Force the solver to start from a safe place every time!
    prob_od.set_val('balance.W', 166.073)
    prob_od.set_val('balance.FAR', 0.01680)
    prob_od.set_val('balance.Nmech', 8197.38)
    prob_od.set_val('fc.balance.Pt', 15.703)
    prob_od.set_val('fc.balance.Tt', 558.31)
    prob_od.set_val('turb.PR', 4.6690)
    
    prob_od.run_model()
    return np.array([prob_od.get_val('perf.TSFC')[0]], dtype=np.float64)

def od_vjp_np(inputs, y_bar):
    global pact_vjp_calls
    pact_vjp_calls += 1
    od_primal_np(inputs, update_count=False) # Instantly restore state
    
    # Explicitly request nested dict format
    J_dict = prob_od.compute_totals(of=['perf.TSFC'], wrt=COUPLED_VARS_OD, return_format='dict')
    
    grad = np.zeros(16, dtype=np.float64)
    for i, var in enumerate(COUPLED_VARS_OD):
        # Nested dictionary lookup using exact strings
        grad[i] = J_dict['perf.TSFC'][var][0][0] * y_bar[0]
        
    return grad

@jax.custom_vjp
def off_design_node(od_inputs):
    shape = jax.ShapeDtypeStruct((1,), jnp.float64)
    return jax.pure_callback(od_primal_np, shape, od_inputs, vmap_method="sequential")

def od_fwd(od_inputs):
    return off_design_node(od_inputs), (od_inputs,)

def od_bwd(res, y_bar):
    od_inputs, = res
    shape = jax.ShapeDtypeStruct((16,), jnp.float64)
    return (jax.pure_callback(od_vjp_np, shape, od_inputs, y_bar, vmap_method="sequential"),)

off_design_node.defvjp(od_fwd, od_bwd)

def run_pact_hybrid_benchmark(N_points):
    """
    Benchmarks Hybrid PACT memory and tracks Setup, Compile, and Execution times.
    """
    global pact_primal_calls, pact_vjp_calls
    pact_primal_calls = 0
    pact_vjp_calls = 0

    gc.collect()
    tracemalloc.start()
    jax.clear_caches()
    
    # PHASE 1: SETUP

    t_setup_start = time.perf_counter()
    
    flight_conditions = jnp.tile(jnp.array([5000.0, 0.2, 8000.0]), (N_points, 1))
    
    # Define it dynamically so JAX treats it as an uncompiled, fresh graph
    def total_tsfc_objective(comp_PR):
        # Returns all 9 coupling variables
        coupling_vars = design_node(comp_PR) 
        
        # Tile the 9 variables N times
        coupling_matrix = jnp.tile(coupling_vars, (N_points, 1))
        
        # Stack horizontally: (N, 9) + (N, 3) = (N, 12)
        od_input_matrix = jnp.hstack([coupling_matrix, flight_conditions])
        
        tsfc_array = jax.vmap(off_design_node)(od_input_matrix)
        return jnp.mean(tsfc_array)

    grad_fn = jax.jit(jax.grad(total_tsfc_objective))
    comp_pr_init = jnp.array([13.5])
    
    t_setup_end = time.perf_counter()
    
    # PHASE 2: JAX COMPILATION (Lowering + XLA Compile)

    t_compile_start = time.perf_counter()
    
    # Explicitly compile without running the numerical payload
    compiled_grad_fn = grad_fn.lower(comp_pr_init).compile()
    
    t_compile_end = time.perf_counter()

    
    # PHASE 3: EXECUTION (The True Compute Benchmark)
    
    t_exec_start = time.perf_counter()
    
    total_grad = compiled_grad_fn(comp_pr_init)
    
    # block_until_ready() is strictly required here; otherwise, JAX will 
    # return the timer immediately while the GPU/CPU works asynchronously!
    total_grad.block_until_ready()
    
    t_exec_end = time.perf_counter()

    # METRICS EXTRACTION
    
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    setup_time = t_setup_end - t_setup_start
    compile_time = t_compile_end - t_compile_start
    exec_time = t_exec_end - t_exec_start
    
    jac_mem_mb = 0.0 # Bypassed by VJP chaining
    total_mem_mb = peak_mem / (1024 * 1024)

    mean_tsfc = total_tsfc_objective(comp_pr_init).item()
    gradient = total_grad.item()
    
    return total_mem_mb, jac_mem_mb, setup_time, compile_time, exec_time, mean_tsfc, gradient, pact_primal_calls, pact_vjp_calls

# ==============================================================================
# PYTHON PACT BENCHMARK
# ==============================================================================

def run_pact_python_benchmark(N_points):
    global pact_primal_calls, pact_vjp_calls
    pact_primal_calls = 0
    pact_vjp_calls = 0
    
    gc.collect()
    tracemalloc.start()
    
    t_setup_start = time.perf_counter()
    comp_pr_init = np.array([13.5], dtype=np.float64)
    flight_conditions = np.tile(np.array([5000.0, 0.2, 8000.0]), (N_points, 1))
    t_setup_end = time.perf_counter()

    t_exec_start = time.perf_counter()
    
    # ==========================================
    # FORWARD PASS (Mathematical Graph Execution)
    # ==========================================
    # 1. Design Node
    coupling_vars = des_primal_np(comp_pr_init)
    
    # 2. Off-Design Nodes
    tsfc_array = np.zeros(N_points, dtype=np.float64)
    for i in range(N_points):
        # Stack coupling vars + flight conditions for this specific point
        od_inputs = np.hstack([coupling_vars, flight_conditions[i]])
        tsfc_array[i] = od_primal_np(od_inputs)[0]
        
    mean_tsfc = np.mean(tsfc_array)
    
    # ==========================================
    # REVERSE PASS (Manual VJP Chaining)
    # ==========================================
    # The gradient of mean(TSFC) wrt each individual TSFC is just 1/N
    dy_dtsfc = np.array([1.0 / N_points], dtype=np.float64)
    
    # We will accumulate the sensitivities of the 13 coupling variables here
    coupling_gradient_accumulator = np.zeros(13, dtype=np.float64)
    
    # 1. Backprop through Off-Design Nodes
    for i in range(N_points):
        od_inputs = np.hstack([coupling_vars, flight_conditions[i]])
        
        # od_vjp_np returns a 16-element array. The first 13 are the coupling vars.
        # We accumulate them because the Design node broadcasted to all N points (Chain Rule)
        point_grad = od_vjp_np(od_inputs, dy_dtsfc)
        coupling_gradient_accumulator += point_grad[:13]
        
    # 2. Backprop through Design Node
    # Pass the accumulated coupling sensitivities backward to find d(TSFC)/d(PR)
    final_gradient = des_vjp_np(comp_pr_init, coupling_gradient_accumulator)[0]
    
    t_exec_end = time.perf_counter()
    
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return (peak_mem / (1024*1024)), 0.0, (t_setup_end - t_setup_start), 0.0, (t_exec_end - t_exec_start), mean_tsfc, final_gradient, pact_primal_calls, pact_vjp_calls

# ==============================================================================
# MAUD OPAQUE AD BENCHMARK (Does not converge due to MDF constraint)
# ==============================================================================

class OpaqueDesignAD(om.ExplicitComponent):
    def setup(self):
        self.add_input('comp_PR', val=13.5)

        for var in COUPLED_VARS_DES:
            safe_name = var.replace('.', '_').replace(':', '_')
            self.add_output(safe_name, val=1.0)
            
        self.declare_partials('*', 'comp_PR')

    def compute(self, inputs, outputs):
        global opaque_ad_primal_calls
        opaque_ad_primal_calls += 1

        pr_val = inputs['comp_PR'][0]
        # print(f"\n[DEBUG] OpaqueDesignAD received comp_PR = {pr_val}")
        
        # Force the engine to crash the script instantly if it fails
        prob_des.model.nonlinear_solver.options['err_on_non_converge'] = False
        
        prob_des.set_val('comp.PR', pr_val)
        
        prob_des.run_model()
        
        for var in COUPLED_VARS_DES:
            safe_name = var.replace('.', '_').replace(':', '_')
            outputs[safe_name] = prob_des.get_val(var)[0]

    def compute_partials(self, inputs, partials):
        global opaque_ad_jac_calls
        opaque_ad_jac_calls += 1

        pr_val = inputs['comp_PR'][0]
        print(f"\n[DEBUG] OpaqueDesignAD received comp_PR = {pr_val}")
        
        # Force the engine to crash the script instantly if it fails
        prob_des.model.nonlinear_solver.options['err_on_non_converge'] = False
        
        prob_des.set_val('comp.PR', pr_val)
        
        prob_des.set_val('comp.PR', inputs['comp_PR'][0])
        
        prob_des.run_model()
        
        J_dict = prob_des.compute_totals(of=COUPLED_VARS_DES, wrt=['comp.PR'], return_format='dict')
        
        for var in COUPLED_VARS_DES:
            safe_name = var.replace('.', '_').replace(':', '_')
            partials[safe_name, 'comp_PR'] = J_dict[var]['comp.PR'][0][0]

class OpaqueOffDesignAD(om.ExplicitComponent):
    def setup(self):
        self.safe_vars = [v.replace('.', '_').replace(':', '_') for v in COUPLED_VARS_OD]

        for idx, safe_name in enumerate(self.safe_vars):
            self.add_input(safe_name, val=coupling_inits[idx])
            
        self.add_input('alt', val=5000.0)
        self.add_input('mn', val=0.2)
        self.add_input('fn_target', val=8000.0)
        self.add_output('tsfc', val=1.0)
        
        # Declare partials using the sanitized names list
        self.declare_partials('tsfc', self.safe_vars)

    def compute(self, inputs, outputs):
        global opaque_ad_primal_calls
        opaque_ad_primal_calls += 1
        
        prob_od.set_val('burner.dPqP', 0.03)
        prob_od.set_val('nozz.Cv', 0.99)
        
        prob_od.set_val('fc.alt', inputs['alt'][0], units='ft')
        prob_od.set_val('fc.MN', inputs['mn'][0])
        prob_od.set_val('balance.Fn_target', inputs['fn_target'][0], units='lbf')
        
        for var in COUPLED_VARS_OD:
            safe_name = var.replace('.', '_').replace(':', '_')
            prob_od.set_val(var, inputs[safe_name][0])
            
        # Reset the solver guesses to prevent local-minima traps from previous crashed points
        prob_od.set_val('balance.W', 166.073)
        prob_od.set_val('balance.FAR', 0.01680)
        prob_od.set_val('balance.Nmech', 8197.38)
            
        prob_od.run_model()
        outputs['tsfc'] = prob_od.get_val('perf.TSFC')[0]

    def compute_partials(self, inputs, partials):
        global opaque_ad_jac_calls
        opaque_ad_jac_calls += 1
        
        prob_od.set_val('burner.dPqP', 0.03)
        prob_od.set_val('nozz.Cv', 0.99)

        prob_od.set_val('fc.alt', inputs['alt'][0], units='ft')
        prob_od.set_val('fc.MN', inputs['mn'][0])
        prob_od.set_val('balance.Fn_target', inputs['fn_target'][0], units='lbf')
        
        for var in COUPLED_VARS_OD:
            safe_name = var.replace('.', '_').replace(':', '_')
            prob_od.set_val(var, inputs[safe_name][0])
            
        prob_od.run_model()
        
        J_dict = prob_od.compute_totals(of=['perf.TSFC'], wrt=COUPLED_VARS_OD, return_format='dict')
        
        for var in COUPLED_VARS_OD:
            safe_name = var.replace('.', '_').replace(':', '_')
            partials['tsfc', safe_name] = J_dict['perf.TSFC'][var][0][0]

def run_maud_opaque_ad_benchmark(N_points):
    global opaque_ad_primal_calls, opaque_ad_jac_calls
    opaque_ad_primal_calls = 0
    opaque_ad_jac_calls = 0
    
    gc.collect()
    tracemalloc.start()
    t_setup_start = time.perf_counter()
    
    prob = om.Problem()
    prob.model.add_subsystem('ivc', om.IndepVarComp('comp_PR', 13.5), promotes_outputs=['comp_PR'])
    prob.model.add_subsystem('design', OpaqueDesignAD(), promotes_inputs=['comp_PR'])
    
    eq_str = 'avg_tsfc = (' + ' + '.join([f'tsfc_{i}' for i in range(N_points)]) + f') / {N_points}'
    prob.model.add_subsystem('objective', om.ExecComp(eq_str), promotes_outputs=['avg_tsfc'])
    
    geom_map = {
        'inlet.Fl_O:stat:area': 'inlet.area',
        'comp.Fl_O:stat:area': 'comp.area',
        'burner.Fl_O:stat:area': 'burner.area',
        'turb.Fl_O:stat:area': 'turb.area',
        'nozz.Throat:stat:area': 'balance.rhs:W'
    }

    for i in range(N_points):
        pt = f'OD{i}'
        prob.model.add_subsystem(pt, OpaqueOffDesignAD())
        
        # Connect the 8 scalar variables (identical names)
        for var in COUPLED_VARS_DES[:8]:
            safe_name = var.replace('.', '_').replace(':', '_')
            prob.model.connect(f'design.{safe_name}', f'{pt}.{safe_name}')
            
        # Connect the 5 geometric variables (mapped names)
        for des_var, od_var in geom_map.items():
            safe_des = des_var.replace('.', '_').replace(':', '_')
            safe_od = od_var.replace('.', '_').replace(':', '_')
            prob.model.connect(f'design.{safe_des}', f'{pt}.{safe_od}')
            
        prob.model.connect(f'{pt}.tsfc', f'objective.tsfc_{i}')
        
    prob.model.add_design_var('comp_PR', lower=10.0, upper=20.0)
    prob.model.add_objective('avg_tsfc')
    
    prob.setup(check=False, mode='rev')
    prob.set_val('comp_PR', 13.5)
    t_setup_end = time.perf_counter()

    t_exec_start = time.perf_counter()
    prob.run_model()
    totals = prob.compute_totals(of=['avg_tsfc'], wrt=['comp_PR'])
    t_exec_end = time.perf_counter()

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    mean_tsfc = prob.get_val('avg_tsfc')[0]
    gradient = totals['avg_tsfc', 'comp_PR'][0][0]
    
    jac_info = get_jac_memory(prob.model)
    if jac_info:
        jac_mem_mb, shape, fmt = jac_info
    else:
        jac_mem_mb = 0.0

    return (peak_mem / (1024*1024)), jac_mem_mb, (t_setup_end - t_setup_start), 0.0, (t_exec_end - t_exec_start), mean_tsfc, gradient, opaque_ad_primal_calls, opaque_ad_jac_calls

# ==============================================================================
# MAUD OPAQUE FD BENCHMARK (Does not converge due to MDF constraint)
# ==============================================================================

class OpaqueDesignFD(om.ExplicitComponent):
    def setup(self):
        self.add_input('comp_PR', val=13.5)
        for var in COUPLED_VARS_DES:
            safe_name = var.replace('.', '_').replace(':', '_')
            self.add_output(safe_name, val=1.0)
            
        # The FD tax: OpenMDAO must perturb the input to find the gradient
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        global opaque_fd_primal_calls
        opaque_fd_primal_calls += 1
        
        prob_des.set_val('comp.PR', inputs['comp_PR'][0])
        # (Insert your exact 11 state resets here: fc.alt, balance.FAR, etc.)
        prob_des.run_model()
        
        for var in COUPLED_VARS_DES:
            safe_name = var.replace('.', '_').replace(':', '_')
            outputs[safe_name] = prob_des.get_val(var)[0]

class OpaqueOffDesignFD(om.ExplicitComponent):
    def setup(self):
        for var in COUPLED_VARS_OD:
            safe_name = var.replace('.', '_').replace(':', '_')
            self.add_input(safe_name, val=1.0)
            
        self.add_input('alt', val=5000.0)
        self.add_input('mn', val=0.2)
        self.add_input('fn_target', val=8000.0)
        self.add_output('tsfc', val=1.0)
        
        # The FD tax: OpenMDAO must perturb ALL inputs individually!
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        global opaque_fd_primal_calls
        opaque_fd_primal_calls += 1
        
        for var in COUPLED_VARS_OD:
            safe_name = var.replace('.', '_').replace(':', '_')
            prob_od.set_val(var, inputs[safe_name][0])
            
        prob_od.set_val('fc.alt', inputs['alt'][0], units='ft')
        prob_od.set_val('fc.MN', inputs['mn'][0])
        prob_od.set_val('balance.Fn_target', inputs['fn_target'][0], units='lbf')
        
        # (Insert your exact 6 off-design state resets here: balance.W, etc.)
        prob_od.run_model()
        
        outputs['tsfc'] = prob_od.get_val('perf.TSFC')[0]

def run_maud_opaque_fd_benchmark(N_points):
    global opaque_fd_primal_calls
    opaque_fd_primal_calls = 0
    gc.collect()
    tracemalloc.start()
    
    t_setup_start = time.perf_counter()
    
    prob = om.Problem()
    prob.model.add_subsystem('ivc', om.IndepVarComp('comp_PR', 13.5), promotes_outputs=['comp_PR'])
    prob.model.add_subsystem('design', OpaqueDesignFD(), promotes_inputs=['comp_PR'])
    
    eq_str = 'avg_tsfc = (' + ' + '.join([f'tsfc_{i}' for i in range(N_points)]) + f') / {N_points}'
    prob.model.add_subsystem('objective', om.ExecComp(eq_str), promotes_outputs=['avg_tsfc'])
    
    # Mapping dict for the geometric variables that change names
    geom_map = {
        'inlet.Fl_O:stat:area': 'inlet.area',
        'comp.Fl_O:stat:area': 'comp.area',
        'burner.Fl_O:stat:area': 'burner.area',
        'turb.Fl_O:stat:area': 'turb.area',
        'nozz.Throat:stat:area': 'balance.rhs:W'
    }

    for i in range(N_points):
        pt = f'OD{i}'
        prob.model.add_subsystem(pt, OpaqueOffDesignFD())
        
        # Connect the 8 scalar variables (identical names)
        for var in COUPLED_VARS_DES[:8]:
            safe_name = var.replace('.', '_').replace(':', '_')
            prob.model.connect(f'design.{safe_name}', f'{pt}.{safe_name}')
            
        # Connect the 5 geometric variables (mapped names)
        for des_var, od_var in geom_map.items():
            safe_des = des_var.replace('.', '_').replace(':', '_')
            safe_od = od_var.replace('.', '_').replace(':', '_')
            prob.model.connect(f'design.{safe_des}', f'{pt}.{safe_od}')
            
        prob.model.connect(f'{pt}.tsfc', f'objective.tsfc_{i}')
        
    prob.model.add_design_var('comp_PR', lower=10.0, upper=20.0)
    prob.model.add_objective('avg_tsfc')
    
    prob.setup(check=False, mode='fwd')
    prob.set_val('comp_PR', 13.5)
    t_setup_end = time.perf_counter()

    t_exec_start = time.perf_counter()
    prob.run_model()
    totals = prob.compute_totals(of=['avg_tsfc'], wrt=['comp_PR'])
    t_exec_end = time.perf_counter()

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    jac_info = get_jac_memory(prob.model)
    if jac_info:
        jac_mem_mb, shape, fmt = jac_info
    else:
        jac_mem_mb = 0.0
    
    return (peak_mem / (1024*1024)), jac_mem_mb, (t_setup_end - t_setup_start), 0.0, (t_exec_end - t_exec_start), opaque_fd_primal_calls

# ==============================================================================
# FLOWTANGENT BENCHMARK
# ==============================================================================

def run_flowtangent_benchmark(N_points):
    primal_calls = 1 + N_points
    jac_calls = 1 + N_points

    jax.clear_caches()
    gc.collect()
    tracemalloc.start()

    #---------------------------------------------------------------------------
    # Setup: Data Structures and Settings
    #---------------------------------------------------------------------------
    t_setup_start = time.perf_counter()

    state = State()
    system = ft_turbojet()
    settings = eqx.tree_at(
        lambda s: (s.analysis.energy, s.numerical),
        Settings(DEBUG_MODE=True),
        (
            JetSettings(design_mode=True, statics=False),
            NumericalSettings(
                batch_size=N_points,
                jacobian=JacobianSettings(
                    calculate=True,
                    couple_time=False,
                    mapping=JacobianMap(
                        system_inputs=(DataPath((
                            "energy",
                            "nodes",
                            "network.line.engine.compressor",
                            "design_parameters",
                            "pressure_ratio")),),
                        state_outputs=(DataPath((
                            "energy",
                            "nodes",
                            "network.line.engine",
                            "fuel",
                            "TSFC"
                        )),))
                )),
        )
    )
    configure_environment(settings)

    # Design Point Setup -----------------------------------
    des_state, des_system, des_settings, design_node = build_turbojet_design(
        state,
        system,
        settings
    )

    if settings.DEBUG_MODE:
        debug_des = design_node.run(des_state, des_system, des_settings)

    # Off-Design Point Setup -------------------------------

    od = TurbojetOpPoint(
        altitude=5_000 * units.ft,
        mach_number=0.2,
        thrust=8_000 * units.lbf,
        mass_flow_rate=168.453135137 * units.parse('lbm/s'),
        rotation_speed=8197.38 * units.rpm,
        turbine_PR=4.669,
        FAR=0.0168
    )

    od_state = od.update_state(des_state)

    od_state, _, _ = update_freestream(od_state, des_system, settings)
    od_base_analysis = build_turbojet_performance(des_system.energy, od)
    od_node = BatchedAnalysis(tag="Off-Design Analysis", analyze=od_base_analysis)

    def design_handover(swap_state, swap_system, swap_settings):
    
            updated_settings = eqx.tree_at(
                lambda s: s.analysis.energy,
                swap_settings,
                replace(swap_settings.analysis.energy, design_mode=False)
            )

            new_state, new_system, new_settings = od_node.initialize(od_state, swap_system, updated_settings)
            
            return new_state, new_system, new_settings

    pact_process = Process(
        tag='PACT Benchmark',
        steps=(design_node, design_handover, od_node),
        initialize=design_node.initialize_controls
        )

    def cycle_objective(comp_pr):
        f_st, f_sys, f_set = pact_process.run(des_state, des_system, des_settings)
        return f_st.process_jacobian

    t_setup_end = time.perf_counter()

    if des_settings.DEBUG_MODE:
        debug_tsfc = cycle_objective(1000.0)

    #---------------------------------------------------------------------------
    # Compilation
    #---------------------------------------------------------------------------

    t_comp_start = time.perf_counter()
    grad_func = jax.jit(pact_process.__call__)
    _ = grad_func.lower(des_state, des_system, des_settings).compile()
    t_comp_end = time.perf_counter()

    #---------------------------------------------------------------------------
    # Execution
    #---------------------------------------------------------------------------

    t_exec_start = time.perf_counter()
    f_st, f_sys, f_set = pact_process.run(des_state, des_system, des_settings)
    grad = f_st.process_jacobian / units.parse('lbm/(hr*lbf)')
    mean_tsfc = f_st.energy.nodes['network.line.engine'].fuel.TSFC
    mean_tsfc.block_until_ready()
    t_exec_end = time.perf_counter()

    #---------------------------------------------------------------------------
    # Metrics
    #---------------------------------------------------------------------------

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    peak_mem_mb = peak_mem / (1024 * 1024)
    jac_mem_mb = 0.0  # The graph architecture never builds a Jacobian matrix
    
    t_setup = t_setup_end - t_setup_start
    t_comp = t_comp_end - t_comp_start
    t_exec = t_exec_end - t_exec_start
    
    # Cast JAX arrays back to standard Python floats for the summary table
    return (
        peak_mem_mb, 
        jac_mem_mb, 
        t_setup, 
        t_comp, 
        t_exec, 
        float(mean_tsfc), 
        float(grad[0] if grad.ndim > 0 else grad), 
        primal_calls, 
        jac_calls
    )

#===============================================================================
# HELPER FUNCTIONS
#===============================================================================

def save_results(filepath: Path | str, architecture: str, metrics: dict):
    """Updates a specific architecture's results in the JSON cache."""
    filepath = Path(filepath)
    data = json.load(open(filepath, 'r')) if filepath.exists() else {}
    
    data[architecture] = metrics
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Saved {architecture} results to {filepath}")

def load_results(filepath: Path | str, architecture: str) -> dict | None:
    """Retrieves the metric dictionary for plotting. Returns None if not found."""
    filepath = Path(filepath)
    if not filepath.exists():
        return None
        
    data = json.load(open(filepath, 'r'))
    return data.get(architecture)

def execute_benchmark(name: str, func, N_array: list, cache_file: Path) -> dict:
    metrics = load_results(cache_file, name)
    
    if metrics:
        print(f"\n{'='*130}\n {name.upper()} BENCHMARK LOADED FROM CACHE\n{'-'*130}")
    else:
        print(f"\n{'='*130}\n EXECUTING {name.upper()} BENCHMARK\n{'-'*130}")
        
        # Initialize empty arrays
        metrics = {k: [] for k in ['N_array', 'total_mem', 'jac_mem', 'setup_time', 'comp_time', 'exec_time', 'tsfc', 'grad', 'func_calls', 'jac_calls']}
        metrics['N_array'] = N_array
        
        _ = func(1) # Eat the cold-start penalty
        
        with tqdm(N_array, leave=False) as pbar:
            for N in pbar:
                pbar.set_description(f"{name}; N={N}")
                res = func(N)
                metrics['total_mem'].append(res[0])
                metrics['jac_mem'].append(res[1])
                metrics['setup_time'].append(res[2])
                metrics['comp_time'].append(res[3])
                metrics['exec_time'].append(res[4])
                metrics['tsfc'].append(res[5])
                metrics['grad'].append(res[6])
                metrics['func_calls'].append(res[7])
                metrics['jac_calls'].append(res[8])

    # Print Formatted Output
    print(f"{'N Points':<10} | {'Mem (MB)':<10} | {'J.Mem (MB)':<10} | {'Setup (s)':<10} | {'Comp (s)':<10} | {'Exec (s)':<10} | {'TSFC':<10} | {'Grad':<15} | {'Primal':<10} | {'Jacobian ':<10}")
    print("-" * 130)
    for i in range(len(metrics['N_array'])):
        print(f"{metrics['N_array'][i]:<10} | {metrics['total_mem'][i]:<10.1f} | {metrics['jac_mem'][i]:<10.1f} | {metrics['setup_time'][i]:<10.2f} | {metrics['comp_time'][i]:<10.2f} | {metrics['exec_time'][i]:<10.2f} | {metrics['tsfc'][i]:<10.2f} | {metrics['grad'][i]:<15.6f} | {metrics['func_calls'][i]:<10d}  | {metrics['jac_calls'][i]:<10d}")

    save_results(cache_file, name, metrics)
    
    return metrics

def plot_error():
    cr = om.CaseReader(test_dir / "solver_errors.sql")
    case_keys = cr.list_cases("root.nonlinear_solver", out_stream=None)

    abs_error_history = [cr.get_case(cid).abs_err for cid in case_keys]
    rel_error_history = [cr.get_case(cid).rel_err for cid in case_keys]
    print(f"Rel. Error: {rel_error_history}")
    print(f"Abs. Error: {abs_error_history}")

    plt.figure(figsize=(8, 5))
    
    # The naive guess will thrash and hit max_iter without dropping the residual
    plt.plot(rel_error_history, 'r-x', linewidth=2, label='Rel. Error')
    
    # The good guess should drop to 1e-6 in 0-3 iterations
    plt.plot(abs_error_history, 'b-o', linewidth=2, label='Abs. Error')
    
    plt.yscale('log')
    plt.title('PyCycle Internal Newton Solver Convergence (Opaque MDF)')
    plt.xlabel('Newton Iteration')
    plt.ylabel('Absolute Residual Norm')
    plt.axhline(1e-6, color='k', linestyle='--', label='Convergence Tolerance')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(test_dir/'error_history.png', dpi=300)


def Compare_Architectures(N_array: list[int], fig_filename: str | Path):
    cache_file = test_dir / "benchmark_cache.json"
    
    # Easily toggle architectures by commenting them out
    architectures = [
        ("MAUD Monolithic", run_monolithic_benchmark, 'r-o'),
        ("Hybrid PACT", run_pact_hybrid_benchmark, 'b-o'),
        ("Python PACT", run_pact_python_benchmark, 'm-o'),
        ("FlowTangent GPU", run_flowtangent_benchmark, 'g-o')
        # ("MAUD Opaque AD", run_maud_opaque_ad_benchmark, 'g-o'),
        # ("MAUD Opaque FD", run_maud_opaque_fd_benchmark, 'm-o'),
    ]
    
    results = {}
    for name, func, style in architectures:
        results[name] = execute_benchmark(name, func, N_array, cache_file)
        results[name]['style'] = style

    # Generate Plots
    fig, axes = plt.subplots(2, 3, figsize=(24, 10))
    
    plot_configs = [
        (axes[0, 0], 'jac_mem', 'Adjoint Matrix Memory Scaling', 'Peak Memory Allocated (MB)'),
        (axes[0, 1], 'total_mem', 'Total Process Memory Scaling', 'Peak Memory Allocated (MB)'),
        (axes[0, 2], 'total_calls', 'Total Engine Sub-Problem Evaluations', 'Number of Evaluations'),
        (axes[1, 0], 'setup_time', 'Problem Setup Time', 'Wall-clock Time (s)'),
        (axes[1, 1], 'comp_time', 'Problem Compilation Time', 'Wall-clock Time (s)'),
        (axes[1, 2], 'exec_time', 'Global Execution Time', 'Wall-clock Time (s)'),
    ]
    
    for ax, key, title, ylabel in plot_configs:
        for name, res in results.items():
            if key == 'total_calls':
                # Dynamically calculate the total calls for the plot
                y_data = [p + a for p, a in zip(res['func_calls'], res['jac_calls'])]
            else:
                y_data = res[key]
                
            ax.plot(res['N_array'], y_data, res['style'], linewidth=2, label=name)
        
        ax.set_title(title)
        ax.set_xlabel('Number of Off-Design Points (N)')
        ax.set_ylabel(ylabel)
        
        # Use a logarithmic scale for the call counts since FD will be exponentially higher
        if key == 'total_calls':
            ax.set_yscale('log')
            
        ax.grid(True)
        ax.legend()
        
    plt.tight_layout()
    plt.savefig(fig_filename, dpi=300)
    print(f"\nBenchmark complete. Saved plots to {fig_filename}")


if __name__ == "__main__":
    N_array = [1,
            #    2, 5, 10, 20, 30, 40, 50
               ]
    fig_fn = test_dir / 'architecture_scaling_benchmark.png'
    Compare_Architectures(N_array, fig_fn)