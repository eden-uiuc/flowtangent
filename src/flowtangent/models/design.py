import jax
import jax.numpy as jnp
import equinox as eqx
import optimistix as optx
from typing import Dict, List, Callable, Any, Tuple, Optional

from ..utils import Module
from ..utils.typing import _
from .manifolds import Manifold
from .surrogates import Surrogate, PredictionState

class ParameterSpace(Module):
    """
    The universal translator between JAX array vectors and physical Oracle keywords.
    """
    names: tuple = eqx.field(static=True, default=_)
    
    def __init__(self, names: List[str]):
        self.names = tuple(names)
        
    def to_kwargs(self, array: jax.Array) -> Dict[str, jax.Array]:
        """Unpacks a stacked JAX array into a named dictionary for external solvers."""
        # Assuming the last dimension of the array matches the number of parameters
        splits = jnp.split(array, len(self.names), axis=-1)
        return {name: split.squeeze(-1) for name, split in zip(self.names, splits)}
        
    def from_kwargs(self, kwargs: Dict[str, jax.Array]) -> jax.Array:
        """Packs a named dictionary back into a stacked JAX array."""
        arrays = [jnp.atleast_1d(kwargs[name]) for name in self.names]
        return jnp.stack(arrays, axis=-1)


class BaseDesignSpace(Module):
    """
    The deterministic foundation. Handles dataset translation, manifold projection, 
    and standard gradient-based optimization.
    """
    dataset: Any = eqx.field(static=True, default=_)
    
    input_space: ParameterSpace = _
    output_space: ParameterSpace = _
    
    manifold: Manifold = _
    surrogate: Surrogate = _

    def project(self, physical_inputs: jax.Array) -> jax.Array:
        return self.manifold.encode(physical_inputs)
        
    def reconstruct(self, latent_z: jax.Array) -> jax.Array:
        return self.manifold.decode(latent_z)

    def evaluate(self, latent_z: jax.Array, context: jax.Array, quantities: Optional[List[str]] = None) -> Dict[str, jax.Array]:
        preds = self.surrogate.predict(latent_z, context)
        out_dict = self.output_space.to_kwargs(preds.means)
        
        if quantities is not None:
            return {q: out_dict[q] for q in quantities}
        return out_dict

    def optimize(self, objective_fn: Callable, initial_state: Dict[str, jax.Array], free_mask: Any):
        """
        Standard deterministic optimization using Equinox PyTree partitioning.
        
        Args:
            objective_fn: Callable taking (preds: dict, state: dict) returning a scalar loss.
            initial_state: Dict like {"latent_z": z_array, "context": c_array}
            free_mask: PyTree matching initial_state with booleans indicating what to optimize.
                       e.g., {"latent_z": True, "context": jnp.array([True, False, False])}
        """
        # Equinox magic: separates the dict into JAX-traced arrays (free) and static context (fixed)
        free_params, fixed_params = eqx.partition(initial_state, free_mask)

        def loss_fn(free, args):
            # Recombine the free and fixed variables back into the full state dict
            state = eqx.combine(free, fixed_params)
            
            # Evaluate surrogate
            preds = self.evaluate(state["latent_z"], state["context"])
            return objective_fn(preds, state)

        # Use BFGS for deterministic unconstrained continuous optimization
        solver = optx.BFGS(rtol=1e-5, atol=1e-5)
        sol = optx.minimise(loss_fn, solver, y0=free_params, throw=False)
        
        # Return the fully updated state dictionary
        return eqx.combine(sol.value, fixed_params)

    def synthesize(self, oracle_function: Callable, new_latents: jax.Array, context: jax.Array, fine_tune_epochs: int = 50):
        """
        The Active Learning feedback loop: Hitting the Oracle and fine-tuning the surrogates.
        """
        # 1. Translate and call the Oracle
        physical_inputs = self.reconstruct(new_latents)
        total_inputs = jnp.concatenate([physical_inputs, context], axis=-1)
        
        oracle_kwargs = self.input_space.to_kwargs(total_inputs)
        ground_truth_results = oracle_function(**oracle_kwargs)
        
        # Pack oracle results back into JAX array
        new_targets = self.output_space.from_kwargs(ground_truth_results)
        
        # 2. Append to Dataset (Assumes dataset has an in-place append method like Zarr)
        self.dataset.append(physical_inputs, context, new_targets)
        
        # 3. Fine-Tune Surrogates (Automatically warm-starts from current weights)
        # We pass self.dataset.X (etc.) to train on the newly updated dataset
        new_manifold = self.manifold.fit(self.dataset.X, epochs=fine_tune_epochs)
        new_surrogate = self.surrogate.fit(
            self.dataset.Z, self.dataset.C, self.dataset.Y, epochs=fine_tune_epochs
        )
        
        # Return a functionally updated DesignSpace
        return eqx.tree_at(
            lambda s: (s.manifold, s.surrogate), 
            self, 
            (new_manifold, new_surrogate)
        )

    def make_subspace(self, z_bounds: Tuple[jax.Array, jax.Array], refit: bool = False):
        """
        Creates a constrained design space. 
        If refit=False, uses zero-copy memory wrapping.
        If refit=True, filters the dataset and trains new local models.
        """
        if not refit:
            return DesignSubspace(base_space=self, z_bounds=z_bounds)
            
        # Hard refit logic: Create a specialized local surrogate
        # 1. Dataset needs a filter method returning a new proxy view of the data
        sliced_dataset = self.dataset.filter_bounds(bounds=z_bounds)
        
        # 2. Train completely new models (or warm start, depending on preference)
        # Using more epochs since this is a new local topological mapping
        new_manifold = self.manifold.fit(sliced_dataset.X, epochs=1000)
        new_surrogate = self.surrogate.fit(
            sliced_dataset.Z, sliced_dataset.C, sliced_dataset.Y, epochs=1000
        )
        
        # Return a new BaseDesignSpace instance focused entirely on this corner of physics
        return eqx.tree_at(
            lambda s: (s.dataset, s.manifold, s.surrogate), 
            self, 
            (sliced_dataset, new_manifold, new_surrogate)
        )


class ProbabilisticDesignSpace(BaseDesignSpace):
    """
    The advanced tier. Requires a probabilistic surrogate. 
    Unlocks uncertainty, Active Learning, and MCMC.
    """
    
    def evaluate_probabilistic(self, latent_z: jax.Array, context: jax.Array) -> PredictionState:
        return self.surrogate.predict(latent_z, context)

    def domain_trust(self, latent_z: jax.Array, context: jax.Array) -> jax.Array:
        preds = self.evaluate_probabilistic(latent_z, context)
        # TODO: Aggregate variances into a scalar domain trust score
        pass

    def explore(self, target_quantity: str, free_variables: List[str], fixed_variables: Dict[str, jax.Array], method: str = "UCB"):
        pass

    def sample(self, requirements: Dict[str, tuple], free_variables: List[str], fixed_variables: Dict[str, jax.Array]):
        """MCMC Inverse Design. Returns valid distributions."""
        pass
        
    def mcmc_log_likelihood(self, latent_z: jax.Array, context: jax.Array, requirements: Dict[str, tuple]) -> jax.Array:
        # Base prior: Standard normal N(0, 1) over the latent space
        log_prior = jnp.sum(jax.scipy.stats.norm.logpdf(latent_z, loc=0.0, scale=1.0))
        # TODO: Add requirement penalties here
        return log_prior


class DesignSubspace(Module):
    """
    Zero-copy memory wrapper that strictly enforces subspace bounds on the Z-vector.
    Duck-types as a BaseDesignSpace or ProbabilisticDesignSpace.
    """
    base_space: Any = _  # Can hold BaseDesignSpace or ProbabilisticDesignSpace
    z_bounds: Tuple[jax.Array, jax.Array] = _ # (lower_bound_array, upper_bound_array)

    # --- Property Forwarding ---
    @property
    def input_space(self) -> ParameterSpace: return self.base_space.input_space
    @property
    def output_space(self) -> ParameterSpace: return self.base_space.output_space
    @property
    def dataset(self) -> Any: return self.base_space.dataset
    @property
    def manifold(self) -> Manifold: return self.base_space.manifold
    @property
    def surrogate(self) -> Surrogate: return self.base_space.surrogate

    def _clip_z(self, z: jax.Array) -> jax.Array:
        return jnp.clip(z, self.z_bounds[0], self.z_bounds[1])

    def project(self, physical_inputs: jax.Array) -> jax.Array:
        z = self.base_space.project(physical_inputs)
        return self._clip_z(z)

    def reconstruct(self, latent_z: jax.Array) -> jax.Array:
        return self.base_space.reconstruct(self._clip_z(latent_z))

    def evaluate(self, latent_z: jax.Array, context: jax.Array, quantities: Optional[List[str]] = None):
        return self.base_space.evaluate(self._clip_z(latent_z), context, quantities)

    def optimize(self, objective_fn: Callable, free_variables: List[str], fixed_variables: Dict[str, jax.Array]):
        """
        Injects a log-barrier penalty into the objective function so 
        gradient-based solvers naturally avoid the subspace boundaries.
        """
        def constrained_objective(z, *args):
            dist_lower = z - self.z_bounds[0]
            dist_upper = self.z_bounds[1] - z
            
            # Log-barrier penalty (pushes value to +inf as z approaches bounds)
            barrier = -jnp.sum(jnp.log(jnp.maximum(dist_lower, 1e-8))) \
                      -jnp.sum(jnp.log(jnp.maximum(dist_upper, 1e-8)))
                      
            # Scale barrier weight as needed, typically annealed during optimization
            return objective_fn(z, *args) + 1e-3 * barrier

        return self.base_space.optimize(constrained_objective, free_variables, fixed_variables)

    def mcmc_log_likelihood(self, latent_z: jax.Array, context: jax.Array, requirements: Dict[str, tuple]) -> jax.Array:
        """Injects a hard uniform boundary prior for BlackJAX samplers."""
        in_bounds = jnp.all((latent_z >= self.z_bounds[0]) & (latent_z <= self.z_bounds[1]))
        boundary_penalty = jnp.where(in_bounds, 0.0, -jnp.inf)
        
        base_ll = self.base_space.mcmc_log_likelihood(latent_z, context, requirements)
        return base_ll + boundary_penalty

    # Forward other methods transparently
    def synthesize(self, *args, **kwargs): return self.base_space.synthesize(*args, **kwargs)
    def evaluate_probabilistic(self, *args, **kwargs): return self.base_space.evaluate_probabilistic(*args, **kwargs)
    def domain_trust(self, *args, **kwargs): return self.base_space.domain_trust(*args, **kwargs)
    def explore(self, *args, **kwargs): return self.base_space.explore(*args, **kwargs)
    def sample(self, *args, **kwargs): return self.base_space.sample(*args, **kwargs)