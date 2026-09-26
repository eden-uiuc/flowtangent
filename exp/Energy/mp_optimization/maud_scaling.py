import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import flowtangent as ft

import json
import time
import tracemalloc
import gc
import numpy as np
import matplotlib.pyplot as plt
import openmdao.api as om
import pycycle.api as pyc
import scipy.sparse
import pynvml
import multiprocessing as mp

import jax
import jax.numpy as jnp
import equinox as eqx

from functools import partial
from tqdm import tqdm
from pathlib import Path
from dataclasses import replace
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

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

from flowtangent import State, Settings, Process
from flowtangent.utils import TreePath, configure_environment, update
from flowtangent.solve import NumericalSettings, JacobianSettings, JacobianMap, BatchedAnalysis
from flowtangent.solve.energy.jets import build_turbojet_design, build_turbojet_performance, JetSettings
from flowtangent.sim.update import update_freestream

from flowtangent.data import units
from flowtangent.components.energy.jets import TurbojetOpPoint

from flowtangent.core._processes import array_barrier

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

def run_maud_benchmark(N_points, dense: bool):
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
    mp_turbojet.options['assembled_jac_type'] = 'dense' if dense else 'csc'
    
    # Add Objective: Average TSFC across all N points
    eq_str = 'avg_tsfc = (' + ' + '.join([f'tsfc_{i}' for i in range(N_points)]) + f') / {N_points}'
    prob.model.add_subsystem('objective', om.ExecComp(eq_str, units='lbm/h/lbf'), promotes_outputs=['avg_tsfc'])
    
    for i in range(N_points):
        prob.model.connect(f'OD{i}.perf.TSFC', f'objective.tsfc_{i}')
        
    prob.model.add_objective('avg_tsfc')
    prob.model.add_design_var('DESIGN.comp.PR', lower=10.0, upper=20.0)

    # Force the monolithic matrix assembly for the global adjoint
    prob.model.linear_solver = om.DirectSolver(assemble_jac=True)
    prob.model.options['assembled_jac_type'] = 'dense' if dense else 'csc'
    
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
    # EXECUTION (Forward Pass + Backward Adjoint)
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
    exec_time = t_exec_end - t_exec_start
    total_mem_mb = peak_mem / (1024 * 1024)

    mean_tsfc = prob.get_val('avg_tsfc')[0]
    gradient = totals['avg_tsfc', 'DESIGN.comp.PR'][0][0]
    
    return total_mem_mb, jac_mem_mb, setup_time, 0.0, exec_time, setup_time + exec_time, mean_tsfc, gradient

# ==============================================================================
# PACT BENCHMARKS
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
prob_od.model.options['assembled_jac_type'] = 'dense'

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

def od_primal_np(inputs):
    prob_od.set_val('burner.dPqP', 0.03)
    prob_od.set_val('nozz.Cv', 0.99)

    coupling_vals = inputs[:13]
    alt, mn, fn_target = inputs[13:]
    
    for var, val in zip(COUPLED_VARS_OD, coupling_vals):
        prob_od.set_val(var, float(val))
        
    prob_od.set_val('fc.alt', float(alt), units='ft')
    prob_od.set_val('fc.MN', float(mn))
    prob_od.set_val('balance.Fn_target', float(fn_target), units='lbf')
    
    # Force the solver to start from a safe place every time
    prob_od.set_val('balance.W', 166.073)
    prob_od.set_val('balance.FAR', 0.01680)
    prob_od.set_val('balance.Nmech', 8197.38)
    prob_od.set_val('fc.balance.Pt', 15.703)
    prob_od.set_val('fc.balance.Tt', 558.31)
    prob_od.set_val('turb.PR', 4.6690)
    
    prob_od.run_model()
    return np.array([prob_od.get_val('perf.TSFC')[0]], dtype=np.float64)

def od_vjp_np(inputs, y_bar):
    od_primal_np(inputs) # Instantly restore state
    
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

# ==============================================================================
# PACT-AD BENCHMARK
# ==============================================================================

def run_pact_ad_benchmark(N_points):
    """
    Benchmarks Hybrid PACT memory and tracks Setup, Compile, and Execution times.
    """

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

    mem_analysis = compiled_grad_fn.memory_analysis()
    peak_algo_ram = mem_analysis.temp_size_in_bytes
    jac_mem_mb = peak_algo_ram / (1024 * 1024)
    
    # PHASE 3: EXECUTION (The True Compute Benchmark)
    
    t_exec_start = time.perf_counter()
    
    total_grad = compiled_grad_fn(comp_pr_init)
    
    # block_until_ready() is strictly required here; otherwise, JAX will 
    # return the timer immediately while the GPU/CPU works asynchronously
    total_grad.block_until_ready()
    
    t_exec_end = time.perf_counter()

    # METRICS EXTRACTION
    
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    setup_time = t_setup_end - t_setup_start
    compile_time = t_compile_end - t_compile_start
    exec_time = t_exec_end - t_exec_start
    
    total_mem_mb = peak_mem / (1024 * 1024)

    mean_tsfc = total_tsfc_objective(comp_pr_init).item()
    gradient = total_grad.item()
    
    return total_mem_mb, jac_mem_mb, setup_time, compile_time, exec_time, setup_time + compile_time + exec_time, mean_tsfc, gradient

# ==============================================================================
# PYTHON PACT BENCHMARK
# ==============================================================================

def run_pact_python_benchmark(N_points):
    
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

    setup_time = t_setup_end - t_setup_start
    exec_time = t_exec_end - t_exec_start
    
    return (peak_mem / (1024*1024)), 0.0, setup_time, 0.0, exec_time, setup_time + exec_time, mean_tsfc, final_gradient

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
        
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        
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

    jax.clear_caches()
    gc.collect()
    # tracemalloc.start()

    #---------------------------------------------------------------------------
    # Setup: Data Structures and Settings
    #---------------------------------------------------------------------------
    t_setup_start = time.perf_counter()

    state = State()
    system = ft_turbojet()
    settings = eqx.tree_at(
        lambda s: (s.analysis.energy, s.numerical),
        Settings(DEBUG_MODE=False),
        (
            JetSettings(design_mode=True, statics=False),
            NumericalSettings(
                batch_size=N_points,
                jacobian=JacobianSettings(
                    calculate=True,
                    couple_time=False,
                    mapping=JacobianMap(
                        system_inputs=(TreePath((
                            "energy",
                            "nodes",
                            "network.line.engine.compressor",
                            "design_parameters",
                            "pressure_ratio")),),
                        state_outputs=(TreePath((
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
        name="Off Design",
        mach_number=0.2,
        altitude=5_000 * units.ft,
        thrust=8_000 * units.lbf,
        compressor_Rline = 2.0,
        turbine_PR = 4.669,
        rotation_speed = 8197.38 * units.rpm,
        mass_flow_rate = 168.45 * units.parse('lbm/s'),
        FAR = 0.0168
    )

    od_state = od.update_state(des_state)

    od_state, _, _ = update_freestream(od_state, des_system, settings)
    od_base_analysis = build_turbojet_performance(des_system.energy, od)
    od_node = BatchedAnalysis(name="Off-Design Analysis", analyze=od_base_analysis)

    def design_handover(swap_state, swap_system, swap_settings):
    
        updated_settings = update(
            swap_settings,
            "analysis.energy",
            replace(swap_settings.analysis.energy, design_mode=False)
        )
        
        return swap_state, swap_system, updated_settings

    def batch_average_TSFC(batch_state, batch_system, batch_settings):
        avg_TSFC = jnp.atleast_3d(jnp.mean(batch_state.energy.nodes['network.line.engine'].fuel.TSFC))

        avg_state = update(
            batch_state,
            lambda b: b.energy.nodes['network.line.engine'].fuel.TSFC,
            avg_TSFC
        )

        return avg_state, batch_system, batch_settings


    pact_process = Process(
        name='PACT Benchmark',
        steps=(
                design_node.initialize_variables,
                design_node,
                design_handover,
                od_node,
                batch_average_TSFC,
            ),
        )

    des_state, des_system, des_settings = array_barrier(des_state, des_system, des_settings)

    t_setup_end = time.perf_counter()

    if settings.DEBUG_MODE:
        full_debug = pact_process(des_state, des_system, des_settings)

    #---------------------------------------------------------------------------
    # Compilation
    #---------------------------------------------------------------------------

    t_comp_start = time.perf_counter()
    grad_func = jax.jit(pact_process.__call__)
    compiled_func = grad_func.lower(des_state, des_system, des_settings).compile()
    t_comp_end = time.perf_counter()

    mem_analysis = compiled_func.memory_analysis()
    peak_algo_vram = (
        mem_analysis.argument_size_in_bytes +
        mem_analysis.output_size_in_bytes + 
        mem_analysis.temp_size_in_bytes -
        mem_analysis.alias_size_in_bytes
    )
    jac_mem_mb = peak_algo_vram / (1024 * 1024)

    #---------------------------------------------------------------------------
    # Execution
    #---------------------------------------------------------------------------

    t_exec_start = time.perf_counter()
    f_st, f_sys, f_set = grad_func(des_state, des_system, des_settings)
    grad = f_st.process_jacobian / units.parse('lbm/(hr*lbf)')
    mean_tsfc = f_st.energy.nodes['network.line.engine'].fuel.TSFC / units.parse('lbm/(hr*lbf)')
    mean_tsfc.block_until_ready()
    t_exec_end = time.perf_counter()

    #---------------------------------------------------------------------------
    # Metrics
    #---------------------------------------------------------------------------

    # current_mem, peak_mem = tracemalloc.get_traced_memory()
    # tracemalloc.stop()
    # info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    # peak_mem = info.used / (1024 * 1024)
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    pid = os.getpid()
    peam_mem = 0

    for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
        if proc.pid == pid:
            peak_mem = proc.usedGpuMemory / (1024 * 1024)
            break

    pynvml.nvmlShutdown()

    # device = jax.local_devices()[0]
    # stats = device.memory_stats()
    # peak_mem = stats.get('peak_bytes_in_use', 0) / (1024 * 1024)
    
    t_setup = t_setup_end - t_setup_start
    t_comp = t_comp_end - t_comp_start
    t_exec = t_exec_end - t_exec_start
    
    # Cast JAX arrays back to standard Python floats for the summary table
    return (
        peak_mem, 
        jac_mem_mb, 
        t_setup, 
        t_comp, 
        t_exec,
        t_setup + t_comp + t_exec,
        float(mean_tsfc.item()), 
        float(grad.item() if grad.ndim > 0 else grad),
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
        print(f"{'N Points':<10} | {'Mem (MB)':<10} | {'J.Mem (MB)':<10} | {'Setup (s)':<10} | {'Comp (s)':<10} | {'Exec (s)':<10} | {'Total (s)':<10} | {'TSFC':<10} | {'Grad':<10}")
        print("-" * 130)
        for i in range(len(N_array)):
            print(f"{metrics['N_array'][i]:<10} | {metrics['total_mem'][i]:<10.1f} | {metrics['jac_mem'][i]:<10.1f} | {metrics['setup_time'][i]:<10.2f} | {metrics['comp_time'][i]:<10.2f} | {metrics['exec_time'][i]:<10.2f} | {metrics['total_time'][i]:<10.2f} | {metrics['tsfc'][i]:<10.4f} | {metrics['grad'][i]:<10.4f}")
    else:       

        print(f"Running {name} warmup pass...")
        warmup_res = func(1)
        del warmup_res
        gc.collect()
        jax.clear_caches()
        print(f"{name} warmup pass complete.")

         # Initialize empty arrays
        metrics = {k: [] for k in ['N_array', 'total_mem', 'jac_mem', 'setup_time', 'comp_time', 'exec_time', 'total_time', 'tsfc', 'grad']}
        metrics['N_array'] = N_array

        print(f"\n{'='*145}\n EXECUTING {name.upper()} BENCHMARK\n{'-'*145}")
        print(f"{'Time':<10} | {'N Points':<10} | {'Mem (MB)':<10} | {'J.Mem (MB)':<10} | {'Setup (s)':<10} | {'Comp (s)':<10} | {'Exec (s)':<10} | {'Total (s)':<10} | {'TSFC':<10} | {'Grad':<10}")
        print("-" * 145)
        
        for N in N_array:
            ctx = mp.get_context('spawn')
            with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as executor:
                future = executor.submit(func, N)
                res = future.result()
            metrics['total_mem'].append(res[0])
            metrics['jac_mem'].append(res[1])
            metrics['setup_time'].append(res[2])
            metrics['comp_time'].append(res[3])
            metrics['exec_time'].append(res[4])
            metrics['total_time'].append(res[5])
            metrics['tsfc'].append(res[6])
            metrics['grad'].append(res[7])

            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"{timestamp:<10} | {N:<10} | {res[0]:<10.1f} | {res[1]:<10.1f} | {res[2]:<10.2f} | {res[3]:<10.2f} | {res[4]:<10.2f} | {res[5]:<10.2f} | {res[6]:<10.4f} | {res[7]:<10.4f}")        

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
        ("MAUD-Dense", partial(run_maud_benchmark, dense=True), 'r-o', 7),
        ("MAUD-Sparse", partial(run_maud_benchmark, dense=False), 'm-x', 10),
        ("PACT-Python", run_pact_python_benchmark, 'b-o', 10),
        ("PACT-AD", run_pact_ad_benchmark, 'c-x', 10),
        ("FlowTangent CPU", run_flowtangent_benchmark, 'g-o', 14),
        ("FlowTangent GPU", run_flowtangent_benchmark, 'k-x', 13)
    ]
    
    results = {}
    for name, func, style, N_max in architectures:
        results[name] = execute_benchmark(name, func, N_array[:N_max], cache_file)
        results[name]['style'] = style

    # 1. Pre-process: Combine Setup and Compile times into Initialization Time
    for name in results:
        results[name]['init_time'] = [
            s + c for s, c in zip(results[name]['setup_time'], results[name]['comp_time'])
        ]

    results['FlowTangent CPU']['total_mem'] = [r + results['FlowTangent CPU']['jac_mem'][i] for i, r in enumerate(results['FlowTangent CPU']['total_mem'])]
    results['PACT-AD']['total_mem'] = [r + results['PACT-AD']['jac_mem'][i] for i, r in enumerate(results['PACT-AD']['total_mem'])]

    # Generate Plots (2x3 grid, we will hide the 6th plot)
    fig, axes = plt.subplots(2, 3, figsize=(24, 10))

    plot_configs = [
        (axes[0, 0], 'jac_mem', 'Adjoint Memory Scaling', 'Peak Memory Allocated (MB)'),
        (axes[0, 1], 'total_mem', 'Total Process Memory Scaling', 'Peak Memory Allocated (MB)'),
        (axes[0, 2], 'total_time', 'Total Program Runtime', 'Wall-clock Time (s)'),
        (axes[1, 0], 'init_time', 'Problem Initialization (Setup + Compile)', 'Wall-clock Time (s)'),
        (axes[1, 1], 'exec_time', 'Global Execution Time', 'Wall-clock Time (s)'),
    ]
    
    axes[1, 2].axis('off') # Hide the unused 6th subplot

    # Create a master array for extrapolation out to N=50,000
    N_extrap = np.logspace(0, np.log10(50000), 100)

    def get_fit_config(name, key):
        """Returns (degree, min_N) based on expected analytical scaling laws."""
        degree, min_N = 1, 1 # Default: linear fit from the start
        
        if 'MAUD-Dense' in name:
            if 'mem' in key:
                degree = 2
            elif 'time' in key and 'init' not in key:
                degree = 3
                
        elif 'PACT' in name:
            if key == "total_mem" or key == "init_time":
                degree = 0
            elif key == "jac_mem":
                degree = 1
            min_N = 50
                
        elif 'FlowTangent' in name:
            if key == 'init_time':
                degree = 0
                min_N = 10 # JIT Compilation time is roughly constant
            elif key == 'exec_time' and 'GPU' in name:
                degree = 1
                min_N = 5000 # Wait for SM thread saturation to see the true O(N) execution slope
            elif key == 'total_mem' and 'GPU' in name:
                degree = 1
                min_N = 10000 # Wait for cuSOLVER 3.6GB workspace to plateau
                
        return degree, min_N

    # Dictionary to store the raw regression stats for LaTeX generation
    fit_stats = {cfg[1]: [] for cfg in plot_configs}

    for ax, key, title, ylabel in plot_configs:
        for name, res in results.items():
            N_data = np.array(res['N_array'])
            y_data = np.array(res[key])
            
            if np.max(y_data) <= 1e-8:
                continue
                
            base_line, = ax.plot(N_data, y_data, res['style'], linewidth=2, label=name)
            degree, min_N = get_fit_config(name, key)
            
            fit_mask = N_data >= min_N
            N_fit = N_data[fit_mask]
            y_fit = y_data[fit_mask]
                
            if len(N_fit) >= max(degree + 1, 1):
                N_proj = N_extrap[N_extrap > np.max(N_data)]
                
                if degree == 0:
                    plateau_val = np.mean(y_fit)
                    y_pred = np.full_like(y_fit, plateau_val)
                    y_proj = np.full_like(N_proj, plateau_val)
                    
                    std_dev = np.std(y_fit)
                    cv = (std_dev / plateau_val) * 100 if plateau_val > 0 else 0
                    wmape = (np.sum(np.abs(y_fit - y_pred)) / np.sum(y_fit)) * 100
                    
                    fit_stats[key].append({
                        'name': name, 'degree': 0, 'coeffs': [plateau_val],
                        'error_val': cv, 'mape': wmape, 'min_N': min_N
                    })
                    
                else:
                    coeffs = np.polyfit(N_fit, y_fit, degree)
                    poly = np.poly1d(coeffs)
                    y_pred = poly(N_fit)
                    y_proj = poly(N_proj)
                    
                    ss_res = np.sum((y_fit - y_pred) ** 2)
                    ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
                    wmape = (np.sum(np.abs(y_fit - y_pred)) / np.sum(y_fit)) * 100
                    
                    fit_stats[key].append({
                        'name': name, 'degree': degree, 'coeffs': coeffs,
                        'error_val': 1 - r_squared, 'mape': wmape, 'min_N': min_N  # Added min_N
                    })
                
                if len(N_proj) > 0:
                    valid = y_proj > 0
                    ax.plot(N_proj[valid], y_proj[valid], color=base_line.get_color(), 
                            linestyle='--', linewidth=1.5, alpha=0.7)
            
        ax.set_title(title)
        ax.set_xlabel('Number of Off-Design Points (N)')
        ax.set_ylabel(ylabel)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend()

    plt.tight_layout()
    plt.savefig(fig_filename, dpi=300)
    
    # =========================================================================
    # LaTeX TABLE GENERATION
    # =========================================================================
    
    metric_name_dict = {
        'jac_mem': 'Adjoint Memory',
        'total_mem': 'Total Process Memory',
        'init_time': 'Problem Initialization',
        'exec_time': 'Global Execution Time',
        'total_time': 'Total Program Runtime'
    }

    print(f"\n{'='*90}\n REGRESSION SUMMARY (LaTeX)\n{'-'*90}")

    for key, title in metric_name_dict.items():
        if key not in fit_stats or not fit_stats[key]: continue
        
        # Dynamically set columns based on the max degree fitted
        max_deg = max([stat['degree'] for stat in fit_stats[key]])
        num_coeff_cols = max_deg + 1
        total_cols = num_coeff_cols + 4  # Name, N_min, Coeffs, and 2 Metrics
        
        # 'l' for Name, 'r' for N_min, 'r's for coeffs, '|rr' for metrics
        col_spec = "lr" + "r" * num_coeff_cols + "|rr"
        
        coeff_headers = " & ".join([f"$C_{{{i}}}$" for i in range(max_deg, 0, -1)])
        coeff_headers += " & $C_0$ / $\\mu$" if max_deg > 0 else "$C_0$ / $\\mu$"
            
        tex = f"\\begin{{table}}[hbt!]\\label{{tab:{key}_regression}}\n"
        tex += f"\\caption{{{title} Regression Models}}\n"
        tex += "\\centering\n"
        tex += f"\\begin{{tabular}}{{{col_spec}}}\n\\hline\n"
        
        # Grouped Multicolumn Headers (shifting \cline to start at column 3)
        tex += f" & & \\multicolumn{{{num_coeff_cols}}}{{c|}}{{Regression Coefficients}} & \\multicolumn{{2}}{{c}}{{Quality Metrics}} \\\\\\cline{{3-{total_cols}}}\n"
        tex += f"Architecture & $N_{{min}}$ & {coeff_headers} & $1 - R^2$ / CV & wMAPE (\\%) \\\\\\hline\n"
        
        for stat in fit_stats[key]:
            name = stat['name'].replace('_', '\\_')
            deg = stat['degree']
            coeffs = stat['coeffs']
            min_N_val = stat['min_N']
            
            # Pad with missing columns if this arch has a lower degree
            padded_coeffs = ["--"] * (max_deg - deg) + [f"{c:.3e}" for c in coeffs]
            coeff_str = " & ".join(padded_coeffs)
            
            # -------------------------------------------------------------
            # Handle microscopic errors for CV, 1-R^2, and wMAPE
            # -------------------------------------------------------------
            err = stat['error_val']
            mape_val = stat['mape']
            
            if deg == 0:
                err_str = "$< 10^{-4}$" if err < 1e-4 else f"{err:.2f}"
            else:
                err_str = "$< 10^{-6}$" if err < 1e-6 else f"{err:.2e}"
                
            mape_str = "$< 10^{-4}$" if mape_val < 1e-4 else f"{mape_val:.2f}"
            # -------------------------------------------------------------
                
            tex += f"{name} & {min_N_val} & {coeff_str} & {err_str} & {mape_str} \\\\\n"
            
        tex += "\\hline\n\\end{tabular}\n\\end{table}\n"
        print(tex)


    print(f"\n{'='*90}\n BENCHMARK RESULTS (LaTeX)\n{'-'*90}")

    for name, res in results.items():
        table_label = name.lower().replace('-', '_')
        tex = f"\\begin{{table}}[hbt!]\\label{{tab:{table_label}_benchmark}}\n"
        tex += f"\\caption{{{name} Benchmark Scaling Data}}\n"
        tex += "\\centering\n"
        tex += "\\begin{tabular}{l|rrrrr}\n\\hline\n"
        tex += "N & Peak (MB) & Grad. (MB) & Init. (s) & Exec. (s) & Total (s) \\\\\\hline\n"
        
        for i, N in enumerate(res['N_array']):
            tex += f"{N} & {res['total_mem'][i]:.1f} & {res['jac_mem'][i]:.1f} & "
            tex += f"{res['init_time'][i]:.2f} & {res['exec_time'][i]:.2f} & {res['total_time'][i]:.2f} \\\\\n"
            
        tex += "\\hline\n\\end{tabular}\n\\end{table}\n"
        print(tex)


if __name__ == "__main__":
    N_array = [
        1, # Testing
        2, 5, 10, 25, 50, 100, # MAUD-Dense
        250, 500, 1000, # MAUD-Sparse, PACT-Python, PACT-AD,
        5000, 10000, 25000, # FT-GPU
        50000 # FT-CPU
    ]

    fig_fn = test_dir / 'architecture_scaling_benchmark.png'
    Compare_Architectures(N_array, fig_fn)