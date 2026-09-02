import jax
import jax.numpy as jnp
import numpy as np
import openmdao.api as om
import gc
import tracemalloc
import time

from pathlib import Path

from simple_turbojet import Turbojet

# ==============================================================================
# 1. NODE 1: DESIGN SIZING POINT (Runs ONCE)
# ==============================================================================
prob_des = om.Problem()
prob_des.model.add_subsystem('des', Turbojet(design=True), promotes=['*'])
prob_des.model.linear_solver = om.DirectSolver(assemble_jac=True)
prob_des.setup(check=False, mode='rev')

# Set Design Guesses
prob_des.set_val('fc.alt', 0, units='ft')
prob_des.set_val('fc.MN', 0.000001)
prob_des.set_val('balance.Fn_target', 11800.0, units='lbf')
prob_des.set_val('balance.T4_target', 2370.0, units='degR')
prob_des.set_val('comp.eff', 0.83)
prob_des.set_val('turb.eff', 0.86)

def des_primal_np(pr_arr):
    prob_des.set_val('comp.PR', float(pr_arr[0]))
    prob_des.run_model()
    throat_area = prob_des.get_val('nozz.Throat:stat:area')[0]
    return np.array([throat_area], dtype=np.float64)

def des_vjp_np(pr_arr, y_bar):
    J_dict = prob_des.compute_totals(of=['nozz.Throat:stat:area'], wrt=['comp.PR'])
    J = J_dict['nozz.Throat:stat:area']['comp.PR'][0][0]
    return np.array([J * y_bar[0]], dtype=np.float64)

@jax.custom_vjp
def design_node(comp_PR):
    shape = jax.ShapeDtypeStruct((1,), jnp.float64)
    return jax.pure_callback(des_primal_np, shape, comp_PR)

def des_fwd(comp_PR):
    return design_node(comp_PR), comp_PR

def des_bwd(res, y_bar):
    comp_PR, = res
    shape = jax.ShapeDtypeStruct((1,), jnp.float64)
    return (jax.pure_callback(des_vjp_np, shape, comp_PR, y_bar),)

design_node.defvjp(des_fwd, des_bwd)


# ==============================================================================
# 2. NODE 2: OFF-DESIGN OPERATING POINT (Vmapped across N conditions)
# ==============================================================================
prob_od = om.Problem()
prob_od.model.add_subsystem('od', Turbojet(design=False), promotes=['*'])
prob_od.model.linear_solver = om.DirectSolver(assemble_jac=True)
prob_od.setup(check=False, mode='rev')

def od_primal_np(inputs):
    # inputs = [throat_area, alt, MN, Fn_target]
    throat_area, alt, mn, fn_target = inputs
    
    prob_od.set_val('balance.rhs:W', float(throat_area))
    prob_od.set_val('fc.alt', float(alt), units='ft')
    prob_od.set_val('fc.MN', float(mn))
    prob_od.set_val('balance.Fn_target', float(fn_target), units='lbf')
    
    prob_od.run_model()
    tsfc = prob_od.get_val('perf.TSFC')[0]
    return np.array([tsfc], dtype=np.float64)

def od_vjp_np(inputs, y_bar):
    # We only need the gradient wrt throat_area for backprop to Node 1
    J_dict = prob_od.compute_totals(of=['perf.TSFC'], wrt=['balance.rhs:W'])
    J_area = J_dict['perf.TSFC']['balance.rhs:W'][0][0]
    
    # Return gradient wrt inputs (throat_area receives J_area * y_bar, others set to 0.0)
    return np.array([J_area * y_bar[0], 0.0, 0.0, 0.0], dtype=np.float64)

@jax.custom_vjp
def off_design_node(od_inputs):
    shape = jax.ShapeDtypeStruct((1,), jnp.float64)
    return jax.pure_callback(od_primal_np, shape, od_inputs)

def od_fwd(od_inputs):
    return off_design_node(od_inputs), od_inputs

def od_bwd(res, y_bar):
    od_inputs, = res
    shape = jax.ShapeDtypeStruct((4,), jnp.float64)
    return (jax.pure_callback(od_vjp_np, shape, od_inputs, y_bar),)

off_design_node.defvjp(od_fwd, od_bwd)


# ==============================================================================
# 3. PACT NETWORK ORCHESTRATION & BENCHMARK
# ==============================================================================
def run_pact_two_node_benchmark(N_points):
    gc.collect()
    tracemalloc.start()

    # Define N flight conditions: [alt, MN, Fn_target]
    flight_conditions = jnp.tile(
        jnp.array([5000.0, 0.2, 8000.0]), (N_points, 1)
    )

    def total_tsfc_objective(comp_PR):
        # Node 1: Run Design Sizing ONCE
        throat_area = design_node(comp_PR)

        # Build batched inputs for Node 2: stack throat_area with each flight condition
        # Shape: (N_points, 4) -> [throat_area, alt, MN, Fn_target]
        throat_column = jnp.tile(throat_area, (N_points, 1))
        od_input_matrix = jnp.hstack([throat_column, flight_conditions])

        # Node 2: Vmap Off-Design across N points
        tsfc_array = jax.vmap(off_design_node)(od_input_matrix)

        # Average TSFC
        return jnp.mean(tsfc_array)

    start_time = time.perf_counter()

    # Compute global gradient through the 2-node PACT graph
    grad_fn = jax.jit(jax.grad(total_tsfc_objective))
    comp_pr_init = jnp.array([13.5])
    total_grad = grad_fn(comp_pr_init).block_until_ready()

    end_time = time.perf_counter()

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return peak_mem / (1024 * 1024), f"Node1 + vmap({N_points})", 0.0, end_time - start_time

if __name__ == "__main__":
    test_dir = Path(__file__).resolve().parent

    N_array = [
        1, 2, 5, 10,
        20, 30, 40, 50
        ]

    MAUD_fig_fn = test_dir / 'pact_scaling_benchmark.png'

    MAUD_Scaling(N_array, MAUD_fig_fn)