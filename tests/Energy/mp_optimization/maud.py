import time
import tracemalloc
import gc
import numpy as np
import matplotlib.pyplot as plt
import openmdao.api as om
import pycycle.api as pyc
import scipy.sparse

from pathlib import Path

import warnings
from openmdao.utils.om_warnings import OpenMDAOWarning
# Suppress the PyCycle negative root warning during Newton steps
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in sqrt')

# Suppress the OpenMDAO monolithic matrix warning which we are intentionally triggering
warnings.filterwarnings('ignore', category=OpenMDAOWarning, message='The top level group has a nonlinear solver')

# Assuming Turbojet is defined in your local pycycle install/scripts
from simple_turbojet import Turbojet

class BenchmarkMPTurbojet(pyc.MPCycle):
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

def run_openmdao_benchmark(N_points):
    """
    Builds the PyCycle problem, converges it, and times the global adjoint solve.
    """
    gc.collect()
    tracemalloc.start()
    
    prob = om.Problem()
    
    # Add the scalable multi-point cycle
    mp_turbojet = prob.model.add_subsystem('mp_turbojet', BenchmarkMPTurbojet(N_points=N_points), promotes=['*'])
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
    # DESIGN Point
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
    
    # 1. Run the forward pass (Newton solvers)
    prob.run_model()

    # print(f"Jacobian Shape: {jac_shape} | Format: {prob.model.options['assembled_jac_type']} | Matrix Memory: {jac_mem_mb:.4f} MB")
    
    # 2. Benchmark the Adjoint (Derivatives)
    start_time = time.perf_counter()
    
    totals = prob.compute_totals(of=['avg_tsfc'], wrt=['DESIGN.comp.PR'])
    
    end_time = time.perf_counter()

    jac_info = get_jac_memory(prob.model)
    if jac_info:
        mem_mb, shape, fmt = jac_info
        # print(f"Jacobian Shape: {shape} | Format: {fmt} | Matrix Mem: {mem_mb:.4f} MB")
    else:
        mem_mb = 0.0
        shape = (0,0)
    
    # 3. Extract Memory
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return peak_mem / (1024 * 1024), shape, mem_mb, end_time - start_time

def MAUD_Scaling(N_array: list[int], fig_filename: str | Path):
        total_mem_results = []
        jac_mem_results = []
        time_results = []
        
        print(f"{'N Points':<10} | {'Total Memory (MB)':<20} | {'Jac. Shape':<20} | {'Jac. Memory (MB)':<20} | {'Gradient Time (s)':<20}")
        print("-" * 110)
        
        for N in N_array:
            total_mem, shape, jac_mem, t = run_openmdao_benchmark(N)
            total_mem_results.append(total_mem)
            jac_mem_results.append(jac_mem)
            time_results.append(t)
            print(f"{N:<10} | {total_mem:<20.2f} | {str(shape):<20} | {jac_mem:<20.2f} | {t:<20.4f}")
    
        # Generate the Money Shot plots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        # Figure A: The O(N^2) Memory Explosion
        ax1.plot(N_array, jac_mem_results, 'r-o', linewidth=2, label='OpenMDAO/MAUD')
        ax1.set_title('Adjoint Memory Scaling')
        ax1.set_xlabel('Number of Nested Solvers (N)')
        ax1.set_ylabel('Peak Memory Allocated (MB)')
        ax1.grid(True)
        ax1.legend()
    
        ax2.plot(N_array, total_mem_results, 'g-o', linewidth=2, label='OpenMDAO/MAUD')
        ax2.set_title('Total Memory Scaling')
        ax2.set_xlabel('Number of Nested Solvers (N)')
        ax2.set_ylabel('Peak Memory Allocated (MB)')
        ax2.grid(True)
        ax2.legend()
        
        # Figure B: The Execution Time Bottleneck
        ax3.plot(N_array, time_results, 'b-o', linewidth=2, label='OpenMDAO/MAUD')
        ax3.set_title('Adjoint Extraction Time')
        ax3.set_xlabel('Number of Nested Solvers (N)')
        ax3.set_ylabel('Wall-clock Time (s)')
        ax3.grid(True)
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig(fig_filename, dpi=300)
        print("\nBenchmark complete. Saved plots to maud_scaling_benchmark.png")


if __name__ == "__main__":
    test_dir = Path("./tests/Energy/mp_optimization")

    N_array = [
        1, 2, 5, 10,
        20, 30, 40, 50
        ]

    MAUD_fig_fn = test_dir / 'maud_scaling_benchmark.png'

    MAUD_Scaling(N_array, MAUD_fig_fn)