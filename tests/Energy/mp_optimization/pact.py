import jax
import jax.numpy as jnp
import numpy as np
import openmdao.api as om

# ==============================================================================
# 1. THE MOCK EXTERNAL AD-CFD SOLVER (e.g., written in C++/Fortran)
# ==============================================================================
class MockAdjointCFD:
    """
    Simulates a high-fidelity reverse-mode AD CFD solver.
    Inputs: [alpha, mach]
    Outputs: [Fx, Fy, Fz, Mx, My, Mz] (6 coupling variables)
    """
    def __init__(self):
        self.primal_calls = 0
        self.adjoint_calls = 0

    def solve_primal(self, x):
        self.primal_calls += 1
        a, m = x[0], x[1]
        # Arbitrary non-linear aerodynamics
        return np.array([
            a**2 * m,      # Fx
            a * m**2,      # Fy
            a + m,         # Fz
            a**3,          # Mx
            m**3,          # My
            a * m          # Mz
        ])

    def solve_adjoint(self, x, y_bar):
        """
        Reverse-mode AD only computes Vector-Jacobian Products (J^T * y_bar).
        y_bar is the downstream sensitivity (the cotangent).
        """
        self.adjoint_calls += 1
        a, m = x[0], x[1]
        
        # Analytic local Jacobians (internal to the CFD solver)
        dFx = np.array([2*a*m, a**2])
        dFy = np.array([m**2, 2*a*m])
        dFz = np.array([1.0, 1.0])
        dMx = np.array([3*a**2, 0.0])
        dMy = np.array([0.0, 3*m**2])
        dMz = np.array([m, a])
        
        # Internal dense Jacobian (6x2)
        J = np.vstack([dFx, dFy, dFz, dMx, dMy, dMz]) 
        
        # Return the VJP (Shape: 2)
        return J.T @ y_bar


# ==============================================================================
# 2. MAUD / OPENMDAO IMPLEMENTATION
# ==============================================================================
class OM_ExternalCFD(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('cfd_tool')

    def setup(self):
        self.add_input('x', val=np.ones(2))
        self.add_output('y', val=np.ones(6))
        self.declare_partials('y', 'x')

    def compute(self, inputs, outputs):
        outputs['y'] = self.options['cfd_tool'].solve_primal(inputs['x'])

    def compute_partials(self, inputs, partials):
        cfd = self.options['cfd_tool']
        x = inputs['x']
        
        J = np.zeros((6, 2))
        # ----------------------------------------------------------------------
        # MAUD FATAL FLAW: To populate the 6x2 partials matrix for MAUD, 
        # we must seed the AD-CFD tool with 6 orthogonal unit vectors.
        # This requires 6 completely separate adjoint passes through the CFD!
        # ----------------------------------------------------------------------
        for i in range(6):
            seed = np.zeros(6)
            seed[i] = 1.0
            J[i, :] = cfd.solve_adjoint(x, seed)
            
        partials['y', 'x'] = J


class OM_Objective(om.ExplicitComponent):
    def setup(self):
        self.add_input('y', val=np.ones(6))
        self.add_output('obj', val=0.0)
        self.declare_partials('obj', 'y')

    def compute(self, inputs, outputs):
        y = inputs['y']
        outputs['obj'] = np.sum(y**2) + 10.0 * y[0]

    def compute_partials(self, inputs, partials):
        y = inputs['y']
        # 1. Calculate the 1D gradient safely
        dy = 2 * y
        dy[0] += 10.0 
        
        # 2. Reshape it to strictly match OpenMDAO's (1, 6) matrix requirement
        partials['obj', 'y'] = dy.reshape(1, 6)


def run_maud_benchmark():
    cfd_tool_maud = MockAdjointCFD()
    
    prob = om.Problem()
    prob.model.add_subsystem('cfd', OM_ExternalCFD(cfd_tool=cfd_tool_maud), promotes=['*'])
    prob.model.add_subsystem('calc_obj', OM_Objective(), promotes=['*'])
    
    prob.model.add_design_var('x')
    prob.model.add_objective('obj')
    prob.setup(mode='rev')
    
    prob.set_val('x', np.array([2.0, 3.0]))
    
    print("\n" + "="*70)
    print(" MAUD / OPENMDAO ARCHITECTURE")
    print("="*70)
    prob.run_model()
    print(f"Primal CFD Evaluations: {cfd_tool_maud.primal_calls}")
    
    total_derivs = prob.compute_totals(of=['obj'], wrt=['x'])
    print(f"Adjoint CFD Evaluations (The Matrix Tax): {cfd_tool_maud.adjoint_calls}")
    print(f"Total Gradient: {total_derivs['obj', 'x'][0]}")


# ==============================================================================
# 3. PACT / FLOWTANGENT IMPLEMENTATION
# ==============================================================================
cfd_tool_pact = MockAdjointCFD()

@jax.custom_vjp
def pact_cfd_wrapper(x):
    # JAX must operate on JAX arrays, but the solver might be external
    x_np = np.asarray(x)
    return jnp.asarray(cfd_tool_pact.solve_primal(x_np))

def pact_fwd(x):
    return pact_cfd_wrapper(x), x

def pact_bwd(res, y_bar):
    x = res
    x_np = np.asarray(x)
    y_bar_np = np.asarray(y_bar)
    
    # ----------------------------------------------------------------------
    # THE PACT ADVANTAGE: JAX automatically assembled the downstream 
    # sensitivities into a single 6-dimensional cotangent vector (y_bar).
    # We pass it into the CFD adjoint EXACTLY ONCE.
    # ----------------------------------------------------------------------
    print(f"  [PACT] Incoming downstream cotangent (y_bar): {y_bar_np}")
    grad_np = cfd_tool_pact.solve_adjoint(x_np, y_bar_np)
    
    return (jnp.asarray(grad_np),)

pact_cfd_wrapper.defvjp(pact_fwd, pact_bwd)

def pact_objective(x):
    # Run the CFD wrapper
    y = pact_cfd_wrapper(x)
    # Synthetic objective (e.g. aero magnitude penalty)
    return jnp.sum(y**2) + 10.0 * y[0]


def run_pact_benchmark():
    print("\n" + "="*70)
    print(" PACT / FLOWTANGENT ARCHITECTURE")
    print("="*70)
    
    x_in = jnp.array([2.0, 3.0])
    
    # Primal execution
    obj_val = pact_objective(x_in)
    print(f"Primal CFD Evaluations: {cfd_tool_pact.primal_calls}")
    
    # Adjoint execution
    grad_fn = jax.grad(pact_objective)
    total_grad = grad_fn(x_in)
    
    print(f"Adjoint CFD Evaluations (VJP Chaining): {cfd_tool_pact.adjoint_calls}")
    print(f"Total Gradient: {total_grad}")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_maud_benchmark()
    run_pact_benchmark()