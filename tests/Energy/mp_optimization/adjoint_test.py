import jax
import jax.numpy as jnp

# A global counter to track how many times the adjoint solver is executed
adjoint_call_counter = 0

# 1. Define the interface for our "External Solver"
@jax.custom_vjp
def mock_cfd_solver(x):
    # Imagine this calls an external C++ binary
    # It takes 2 inputs and outputs 3 coupling variables (N_c = 3)
    out_1 = x[0]**2          # Lift
    out_2 = x[1]**3          # Drag
    out_3 = x[0] * x[1]      # Moment
    return jnp.array([out_1, out_2, out_3])

# 2. Define the Forward Pass (Primal)
def cfd_fwd(x):
    # Return the outputs, and save whatever state we need for the reverse pass
    y = mock_cfd_solver(x)
    return y, (x,)  # (x,) is the saved state/residuals

# 3. Define the Backward Pass (Adjoint)
def cfd_bwd(saved_state, y_bar):
    global adjoint_call_counter
    adjoint_call_counter += 1
    x, = saved_state
    
    print("\n--- EXTERNAL ADJOINT SOLVER TRIGGERED ---")
    print(f"Incoming cotangent vector (y_bar): {y_bar}")
    print(f"Size of y_bar: {len(y_bar)} (This matches N_c = 3)")
    
    # Calculate the Vector-Jacobian Product (J^T * y_bar)
    # The solver uses y_bar directly to weight the outputs in a SINGLE pass
    grad_x0 = (2 * x[0] * y_bar[0]) + (0 * y_bar[1]) + (x[1] * y_bar[2])
    grad_x1 = (0 * y_bar[0]) + (3 * x[1]**2 * y_bar[1]) + (x[0] * y_bar[2])
    
    return (jnp.array([grad_x0, grad_x1]),)

# Tie the custom rules to the function
mock_cfd_solver.defvjp(cfd_fwd, cfd_bwd)


# 4. Build the MDO Problem (The PACT Network)
def mdo_objective(x):
    # Call the solver (returns 3 coupling variables)
    coupling_vars = mock_cfd_solver(x)
    
    # Do some downstream analysis (mocking an engine cycle, mission solver, etc.)
    # We reduce the 3 variables down to 1 scalar objective (e.g., Fuel Burn)
    fuel_burn = jnp.sum(coupling_vars ** 2) + 10.0 * coupling_vars[0]
    return fuel_burn


# 5. Run the experiment
if __name__ == "__main__":
    # Initial design variables
    x_in = jnp.array([2.0, 3.0])
    
    print("Executing jax.grad(mdo_objective)...")
    # Ask JAX for the gradient of the entire MDO problem
    total_gradient = jax.grad(mdo_objective)(x_in)
    
    print("\n--- FINAL RESULTS ---")
    print(f"Total Gradient: {total_gradient}")
    print(f"Total Adjoint Executions: {adjoint_call_counter}")