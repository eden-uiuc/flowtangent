import jax.numpy as jnp
import numpy as np

def test_evaluate_requirements_regression(ndarrays_regression):
    """
    Snapshots the exact numeric outputs of a JAX process.
    If the underlying math changes in the future, this test will fail.
    """
    
    # 1. Setup and run your complex JAX process (mocked here)
    # process = ftp.VORJAX(...)
    # x = jnp.array([1.0, 2.0])
    # results = process.evaluate_requirements(x)
    
    # Mocking the results dictionary for the example
    results = {
        "lift_req": jnp.array([1.234, 5.678]),
        "drag_req": jnp.array([0.012, 0.034])
    }
    
    # 2. Cast JAX DeviceArrays to standard NumPy arrays
    # pytest-regressions requires standard numpy arrays to save to disk
    snapshot_data = {}
    for name, array_val in results.items():
        snapshot_data[name] = np.asarray(array_val)
        
    # 3. Check against the snapshot
    ndarrays_regression.check(snapshot_data)