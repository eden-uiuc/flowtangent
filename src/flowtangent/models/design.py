from ..utils import Module
from ..utils.typing import _

from equinox import AbstractClassVar

class BaseDesignSpace(Module):
    """
    The deterministic foundation. Handles dataset translation, manifold projection, 
    and standard gradient-based optimization.
    """
    dataset: AbstractClassVar = _
    manifold: AbstractClassVar = _   # The encoder/decoder
    surrogate: Module = _  # The performance predictor (e.g., MLP, Polynomial, GP)

    # --- 1. The Facade Methods ---
    def project(self, physical_inputs):
        """Pre-processes, scales, and calls manifold.encode()"""
        pass
        
    def reconstruct(self, continuous_inputs):
        """Calls manifold.decode(), un-scales, and enforces physical bounds."""
        pass

    # --- 2. The Dynamic Query Engine ---
    def evaluate(self, inputs: dict, quantities: list):
        """
        Passes inputs through the surrogate. 
        Returns point estimates (means) for the requested quantities.
        """
        pass

    # --- 3. The Universal Solver ---
    def optimize(self, objective_fn, free_variables: list, fixed_variables: dict):
        """
        Standard deterministic optimization (e.g., L-BFGS, Adam).
        By defining what is 'fixed', this seamlessly handles both:
        - Shape Optimization (Free: Latent, Fixed: Mach/Re)
        - Trajectory Optimization (Free: Mach/Alpha, Fixed: Latent)
        """
        pass

    def synthesize(self, oracle_function, new_designs):
        """Sends data to the Oracle, updates the Dataset, and retrains."""
        pass


class ProbabilisticDesignSpace(BaseDesignSpace):
    """
    The advanced tier. Requires a probabilistic surrogate. 
    Unlocks uncertainty, Active Learning, and MCMC.
    """
    
    def evaluate_probabilistic(self, inputs: dict, quantities: list):
        """Returns distributions (Mean + Variance) instead of just point estimates."""
        pass

    def domain_trust(self, inputs: dict):
        """
        Calculates epistemic uncertainty to prevent the optimizer 
        from hallucinating outside the known physics manifold.
        """
        pass

    def explore(self, target_quantity, free_variables, fixed_variables, method="UCB"):
        """
        Active Learning acquisition. Finds the point in the 'free' space 
        that maximizes surrogate uncertainty or expected improvement.
        """
        pass

    def sample(self, requirements: dict, free_variables: list, fixed_variables: dict):
        """
        MCMC Inverse Design. 
        Wanders the 'free' space to return a distribution of valid inputs 
        that satisfy all operational requirements.
        """
        pass