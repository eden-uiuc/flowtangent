import warnings
from typing import Any, Callable, Optional

import blackjax
import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from blackjax.mcmc.hmc import HMCState
from jax.flatten_util import ravel_pytree
from jax.scipy.stats import norm

from .. import Process
from .. import utils as ftu
from ..opt import _funcs as opt_funcs
from ..utils import Module, TreePath, TreePathLike, method_field, static_field
from ..utils.data import DataLoader, Dataset, LatentDataLoader, RandomSampler, numpy_collate
from ..utils.typing import _
from .manifolds import Manifold
from .surrogates import Surrogate, SurrogateEvaluation


class Parameter(Module):
    name: str = static_field("Parameter")

    path: Optional[TreePathLike] = None

    extract_func: Optional[Callable] = method_field(None)
    update_func: Optional[Callable] = method_field(None)

    is_output: bool = False

    def __post_init__(self):

        has_path = self.path is not None
        has_extract = self.extract_func is not None
        has_update = self.extract_func is not None

        if has_path:
            tree_path = TreePath.cast(self.path)
            object.__setattr__(self, "path", tree_path)

            if self.name == "Parameter" and tree_path.name is not None:
                object.__setattr__(self, "name", tree_path.name)

            if has_extract and has_update:
                warnings.warn(
                    f"Parameter {self.name} has a path specified along with extract and update functions. "
                    "This is not the intended usage and may lead to unexpected behavior."
                )

            if not self.extract_func:
                object.__setattr__(self, "extract_func", lambda tree: ftu.get_target(tree, tree_path))
            if not self.update_func:
                object.__setattr__(
                    self, "update_func", lambda tree, value: ftu.update(tree, ftu.update(tree_path, "value", value))
                )
        else:
            if not self.extract_func:
                raise ValueError(f"Parameter {self.name} has no path and is missing an extraction function.")
            if not self.update_func:
                raise ValueError(f"Parameter {self.name} has no path and is missing an update function.")
            object.__setattr__(self, "path", TreePath(path="", name=self.name))

        if self.name == "Parameter":
            raise ValueError("Parameters must be given a name matching a value in the design space dataset.")

    @property
    def value(self):
        if self.path is not None:
            return TreePath.cast(self.path).value
        else:
            return jnp.empty(0)


class RequirementEvaluation(Module):
    """Payload returned by a Requirement's evaluate method."""

    met: jax.Array = _
    residual: jax.Array = _  # Positive = margin of safety, Negative = violation
    probability: Optional[jax.Array] = None
    log_likelihood: Optional[jax.Array] = None
    derivatives: Optional[jax.Array] = None


class Requirement(Parameter):
    """
    Evaluates physical constraints.
    Can map directly to a variable name or use a custom derived extract_func.
    """

    name: str = "Parameter"
    eq_bound: Optional[jax.Array] = None
    lower_bound: Optional[jax.Array] = None
    upper_bound: Optional[jax.Array] = None
    tolerance: float = 1e-5

    # Generalizes the probabilistic evaluation (default is Gaussian CDF)
    # Signature: (x, loc, scale) -> cumulative probability
    error_cdf: Callable = eqx.field(default=norm.cdf, static=True)

    def __post_init__(self):
        # Handle equality constraints
        if self.eq_bound is not None:
            object.__setattr__(self, "lower_bound", self.eq_bound)
            object.__setattr__(self, "upper_bound", self.eq_bound)

        # Apply tolerance to expand the acceptable bounds
        if self.lower_bound is not None:
            object.__setattr__(self, "lower_bound", self.lower_bound - self.tolerance)
        if self.upper_bound is not None:
            object.__setattr__(self, "upper_bound", self.upper_bound + self.tolerance)

        super().__post_init__()

    def evaluate(
        self,
        vals_dict: dict[str, jax.Array],
        vars_dict: Optional[dict[str, jax.Array]] = None,
    ) -> RequirementEvaluation:

        # 1. Evaluate the Mean Value safely
        if self.name in vals_dict:
            val = vals_dict[self.name]
        elif self.extract_func is not None:
            val = self.extract_func(vals_dict)
        else:
            raise ValueError(f"Requirement '{self.name}' not found in dictionary and lacks extract_func.")

        # 2. Evaluate Residual (Margin to nearest boundary)
        if self.lower_bound is not None and self.upper_bound is not None:
            residual = jnp.minimum(val - self.lower_bound, self.upper_bound - val)
        elif self.lower_bound is not None:
            residual = val - self.lower_bound
        elif self.upper_bound is not None:
            residual = self.upper_bound - val
        else:
            residual = jnp.array(jnp.inf)

        # Bounds were already expanded by tolerance in __post_init__
        met = residual >= 0.0

        # 3. Probabilistic Evaluation (If variance is provided)
        prob, log_ll = None, None
        if vars_dict is not None:
            if self.extract_func is None:
                val_var = vars_dict.get(str(self.name), jnp.array(0.0))
            else:
                # Delta Method: propagate variance through the custom derived function
                grads = jax.grad(self.extract_func)(vals_dict)
                val_var = sum((grads[k] ** 2) * vars_dict[k] for k in vars_dict.keys() if k in grads)

            sigma = jnp.sqrt(jnp.maximum(val_var, 1e-12))

            if self.lower_bound is not None and self.upper_bound is not None:
                prob = self.error_cdf(self.upper_bound, loc=val, scale=sigma) - self.error_cdf(
                    self.lower_bound, loc=val, scale=sigma
                )
            elif self.lower_bound is not None:
                prob = 1.0 - self.error_cdf(self.lower_bound, loc=val, scale=sigma)
            elif self.upper_bound is not None:
                prob = self.error_cdf(self.upper_bound, loc=val, scale=sigma)
            else:
                prob = jnp.array(1.0)

            prob = jnp.clip(prob, 1e-12, 1.0)
            log_ll = jnp.log(prob)

        return RequirementEvaluation(met=met, residual=residual, probability=prob, log_likelihood=log_ll)


class DesignSpace(Module):
    """
    The deterministic foundation. Handles dataset translation, manifold projection,
    and standard gradient-based optimization.
    """

    dataset: Dataset = static_field(_)
    strict_dataset: bool = static_field(True)

    parameters: tuple[Parameter, ...] = _
    requirements: tuple[Requirement, ...] = _

    manifold: Manifold = _
    surrogate: Surrogate = _

    solver: Any = optx.LevenbergMarquardt

    augment: Process = _

    _unravel: Callable = method_field(_)

    @property
    def _param_map(self):
        return {str(p.name): p for p in self.parameters}

    @property
    def inputs(self):
        return tuple(p for p in self.parameters if not p.is_output)

    @property
    def outputs(self):
        return tuple(p for p in self.parameters if p.is_output)

    @property
    def X(self) -> jax.Array:
        """Extracts all physical inputs from the dataset as a packed [N, D] array."""
        # Because self.inputs is sorted, this guarantees the exact same
        # packing order as ravel_pytree!
        cols = [jnp.atleast_1d(getattr(self.dataset, str(p.name))) for p in self.inputs]
        return jnp.column_stack(cols)

    @property
    def Y(self) -> jax.Array:
        """Extracts all targets from the dataset as a packed [N, D_out] array."""
        cols = [jnp.atleast_1d(getattr(self.dataset, str(p.name))) for p in self.outputs]
        return jnp.column_stack(cols)

    def __getattr__(self, name):
        if name in self._param_map:
            return self._param_map[name]
        return object.__getattribute__(self, name)

    def __post_init__(self):
        if len(self.dataset) == 0:
            raise ValueError("Dataset must have at least one row to infer Parameter shapes.")

        sample_row = self.dataset[0]
        initialized_params = []

        for p in self.parameters:
            in_dataset = str(p.name) in self.dataset.full_vars

            # CASE 1: User did NOT provide a value
            if p.value.size == 0 or p.value is None:
                if not in_dataset:
                    raise ValueError(
                        f"Parameter '{p.name}' is not in the dataset and has no default value. "
                        "Shape cannot be inferred."
                    )
                dataset_val = jnp.atleast_1d(sample_row[p.name])
                new_p = eqx.tree_at(lambda x: x._tree_path.value, p, dataset_val)
                initialized_params.append(new_p)

            # CASE 2: User DID provide a value
            else:
                user_val = jnp.atleast_1d(p.value)

                if in_dataset:
                    dataset_val = jnp.atleast_1d(sample_row[p.name])
                    if user_val.shape != dataset_val.shape:
                        raise ValueError(
                            f"Shape mismatch for Parameter '{p.name}'. User: {user_val.shape}, Dataset: {dataset_val.shape}."
                        )
                    if user_val.dtype != dataset_val.dtype:
                        user_val = user_val.astype(dataset_val.dtype)
                else:
                    # New derived variable! Pass the strict_dataset flag as the quiet toggle
                    nan_overlay = jnp.full((len(self.dataset), *user_val.shape), jnp.nan, dtype=user_val.dtype)
                    object.__setattr__(
                        self,
                        "dataset",
                        self.dataset.add_variable(str(p.name), nan_overlay, quiet=not self.strict_dataset),
                    )

                new_p = eqx.tree_at(lambda x: x._tree_path.value, p, user_val)
                initialized_params.append(new_p)

        ordered_defaults = tuple(p.value for p in self.inputs)
        _, unravel_fn = ravel_pytree(ordered_defaults)
        object.__setattr__(self, "_unravel_inputs", unravel_fn)

    def get_dataloader(self, batch_size: int = 256, shuffle: bool = True, **kwargs) -> DataLoader:
        """
        Creates a NumPy-native DataLoader that streams [X_batch, Y_batch] arrays on the fly.
        """
        in_names = [str(p.name) for p in self.inputs]
        out_names = [str(p.name) for p in self.outputs]

        def xy_collate_fn(batch):
            # 1. Use the standalone utility to handle the raw dictionary batching safely
            batched_dict = numpy_collate(batch)

            # 2. Extract and stack into the [Batch, D] matrices required by the 1D contract
            X_batch = jnp.column_stack([batched_dict[name] for name in in_names])
            Y_batch = jnp.column_stack([batched_dict[name] for name in out_names])

            return X_batch, Y_batch

        # Instantiate the FT facade wrapper
        return DataLoader(
            dataset=self.dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=xy_collate_fn, **kwargs
        )

    def project(self, physical_inputs: dict) -> jax.Array:
        """
        Maps a dictionary of physical variables to the latent space.
        Guarantees topological ordering and flattens multi-dimensional parameters.
        """
        try:
            # Extract into a Tuple (strictly ordered by self.inputs)
            ordered_vals = tuple(physical_inputs[p.name] for p in self.inputs)
        except KeyError as e:
            raise KeyError(f"Missing required input parameter: {e}")

        # Ravel the tuple into a flat 1D array
        flat_x, _ = ravel_pytree(ordered_vals)

        return self.manifold.encode(flat_x)

    def reconstruct(self, z: jax.Array) -> dict[str, jax.Array]:
        """
        Maps a latent vector back to a dictionary of physical variables.
        """
        # Decode returns the flat 1D array
        flat_x = self.manifold.decode(z)

        # Unravel exactly back into the structured Tuple
        structured_tuple = self._unravel(flat_x)

        # Zip it back together with the parameter names
        return {str(p.name): val for p, val in zip(self.inputs, structured_tuple)}

    def evaluate(self, x: Optional[jax.Array] = None, compute_derivatives: bool = False) -> SurrogateEvaluation:
        """
        Evaluates the surrogate.
        If 'x' is omitted, projects the current Parameter values into the latent space.
        """
        if x is None:
            # Grab current state directly from the Parameter objects
            current_inputs = {p.name: p.value for p in self.inputs}
            x = self.project(current_inputs)

        return self.surrogate.predict(x, compute_derivatives)

    def evaluate_requirements(self, x: jax.Array, compute_derivatives: bool = False) -> dict[str, Any]:
        """
        Evaluates all registered requirements against a latent vector x.
        If compute_derivatives=True, JAX calculates the Jacobian of the residuals w.r.t x.
        """

        def _get_residuals(x_in):
            preds = self.evaluate(x_in)

            # Combine physical inputs and predicted outputs into a single dictionary
            full_dict = self.reconstruct(x_in)
            for i, p in enumerate(self.outputs):
                full_dict[p.name] = preds.means[i]

            residuals = {}
            for req in self.requirements:
                req_eval = req.evaluate(vals_dict=full_dict)
                residuals[req.name] = req_eval.residual
            return residuals

        # Execute standard evaluation
        preds = self.evaluate(x)
        full_dict = self.reconstruct(x)
        for i, p in enumerate(self.outputs):
            full_dict[p.name] = preds.means[i]

        jacobians = jax.jacrev(_get_residuals)(x) if compute_derivatives else {}

        results = {}
        for req in self.requirements:
            req_eval = req.evaluate(vals_dict=full_dict)
            if compute_derivatives:
                req_eval = eqx.tree_at(lambda e: e.derivatives, req_eval, jacobians[req.name])
            results[req.name] = req_eval

        return results

    def sample_feasible(
        self,
        num_samples: int = 1000,
        domain: str = "latent",
        sampler: Optional[Any] = None,
        target_requirements: Optional[list[str]] = None,
        batch_size: int = 256,
    ) -> jax.Array:
        """
        Filters the dataset to find points that satisfy engineering requirements.
        Uses RandomSampler by default if none is provided.
        """
        req_names = target_requirements or [r.name for r in self.requirements]

        if sampler is None:
            # Default to scanning the dataset in random order
            sampler = RandomSampler(self.dataset)

        dataloader = self.get_dataloader(batch_size=batch_size, sampler=sampler, shuffle=False)

        feasible_points = []
        collected = 0

        # Create a vmap-able function to check feasibility across a batch
        def check_batch_feasible(x_batch_point):
            req_evals = self.evaluate_requirements(x_batch_point)
            return jnp.all(jnp.array([req_evals[name].met for name in req_names]))

        check_batch_vmap = jax.vmap(check_batch_feasible)

        for x_batch, _ in dataloader:
            if domain == "latent":
                eval_batch = jax.vmap(self.manifold.encode)(x_batch)
            elif domain == "primal":
                eval_batch = x_batch
            else:
                raise ValueError(f"Unknown domain: {domain}")

            # Mask out the points that violate physics/requirements
            mask = check_batch_vmap(eval_batch)
            valid_points = eval_batch[mask]

            if len(valid_points) > 0:
                feasible_points.append(valid_points)
                collected += len(valid_points)

            if collected >= num_samples:
                break

        if not feasible_points:
            raise ValueError("No feasible points found in the sampled dataset.")

        return jnp.concatenate(feasible_points, axis=0)[:num_samples]

    def estimate_feasible_bounds(self, padding: float = 0.05, **kwargs) -> tuple[jax.Array, jax.Array]:
        """Calculates a tight bounding box around the feasible points."""
        valid_points = self.sample_feasible(**kwargs)
        lower_bound = jnp.min(valid_points, axis=0) - padding
        upper_bound = jnp.max(valid_points, axis=0) + padding
        return lower_bound, upper_bound

    def optimize(
        self,
        objective_fn: Callable,
        opt_vars: list[str],
        initial_values: Optional[dict[str, jax.Array]] = None,
        opt_kwargs: Optional[dict[str, Any]] = None,
    ):

        # 1. Gather all baseline parameter values
        base_values = {p.name: p.value for p in self.inputs}

        # 2. Selectively override with any provided initial values
        if initial_values is not None:
            base_values.update(initial_values)

        if opt_kwargs is None:
            opt_kwargs = {"rtol": 1e-5, "atol": 1e-5}

        solver = self.solver(**opt_kwargs)

        # 3. Automatically infer fixed variables
        fixed_vars = [var for var in base_values.keys() if var not in opt_vars]

        # 4. Partition the initial state dicts
        opt_initial = {var: base_values[var] for var in opt_vars}
        fixed_initial = {var: base_values[var] for var in fixed_vars}

        # 5. Ravel the dictionaries into flat arrays for the solver
        opt_vals, opt_unravel = ravel_pytree(opt_initial)
        fixed_vals, fixed_unravel = ravel_pytree(fixed_initial)

        # 6. Define the optimistix-compatible loss function
        def loss_fn(opt_v, fixed_v):
            # A. Unpack the flat arrays back into dictionaries
            opt_dict = opt_unravel(opt_v)
            fixed_dict = fixed_unravel(fixed_v)

            # B. Merge into a single physical dictionary
            full_dict = {**fixed_dict, **opt_dict}

            # C. Project to latent space
            x = self.project(full_dict)

            # D. Evaluate the surrogate
            preds = self.evaluate(x)

            return objective_fn(preds, x, full_dict)

        # 7. Execute the continuous optimization loop
        sol = optx.minimise(fn=loss_fn, solver=solver, y0=opt_vals, args=fixed_vals, throw=False)

        # Return the completely recombined, optimized physical dictionary
        optimized_opt_dict = opt_unravel(sol.value)
        return {**fixed_initial, **optimized_opt_dict}

    def augment_data(
        self,
        new_latents: jax.Array,
        augment_fn: Callable,
        fixed_values: Optional[dict[str, jax.Array]] = None,
    ) -> "DesignSpace":
        """
        Gathers ground truth data from the Oracle and appends it to the dataset's virtual overlay.
        Does NOT refit the models automatically.
        """
        # 1. Reconstruct physical geometry
        opt_dict = self.reconstruct(new_latents)

        # 2. Gather base variables and apply fixed overrides
        full_input_dict = {p.name: p.value for p in self.inputs}
        if fixed_values is not None:
            full_input_dict.update(fixed_values)

        full_input_dict.update(opt_dict)

        # 3. Query the augmentation function
        ground_truth_dict = augment_fn(full_input_dict)

        # 4. Append to the Dataset's virtual overlay
        new_row = {**full_input_dict, **ground_truth_dict}
        new_dataset = self.dataset.append(new_row)

        # 5. Return functionally updated DesignSpace
        return eqx.tree_at(lambda s: s.dataset, self, new_dataset)

    def refit_models(
        self, batch_size: int = 256, manifold_epochs: int = 100, surrogate_epochs: int = 50, **dataloader_kwargs
    ) -> "DesignSpace":
        """
        Streams the current dataset to fine-tune the models.
        """
        dataloader = self.get_dataloader(batch_size=batch_size, **dataloader_kwargs)

        # 1. Fine-Tune Manifold on physical data
        new_manifold = self.manifold.fit(dataloader, epochs=manifold_epochs)

        # 2. Wrap the dataloader so the Surrogate only sees latent vectors
        latent_dataloader = LatentDataLoader(dataloader, new_manifold)

        # 3. Fine-Tune Surrogate (blissfully unaware of the Manifold's existence)
        new_surrogate = self.surrogate.fit(latent_dataloader, epochs=surrogate_epochs)

        return eqx.tree_at(lambda s: (s.manifold, s.surrogate), self, (new_manifold, new_surrogate))

    def make_subspace(
        self,
        latent_bounds: Optional[tuple[jax.Array, jax.Array]] = None,
        primal_bounds: Optional[dict[str, tuple[Optional[float], Optional[float]]]] = None,
        refit: bool = False,
        **refit_kwargs,
    ):
        """
        Creates a constrained design space filtered by primal and/or latent bounds.

        Args:
            latent_bounds: (lower_bound_array, upper_bound_array)
            primal_bounds: Dict of {"var_name": (lower_bound, upper_bound)}. Use None for open bounds.
                           e.g., {"mach": (0.8, 0.9), "L_over_D": (15.0, None)}
        """
        # 1. Compute Primal Mask (Extremely fast, evaluated directly on the columns)
        master_mask = jnp.ones(len(self.dataset), dtype=jnp.bool_)

        if primal_bounds is not None:
            for var_name, (low, high) in primal_bounds.items():
                # Note: This loads the specific column into memory, which is usually safe
                # even for large datasets (e.g., 10M rows of float32 = 40MB).
                col_data = jnp.asarray(getattr(self.dataset, var_name))

                if low is not None:
                    master_mask = master_mask & (col_data >= low)
                if high is not None:
                    master_mask = master_mask & (col_data <= high)

        # 2. Compute Latent Mask (Out-of-core safe via DataLoader streaming)
        if latent_bounds is not None:
            latent_mask_chunks = []

            # Use shuffle=False so the batch order perfectly matches the dataset row order
            for x_batch, _ in self.get_dataloader(shuffle=False):
                z_batch = jax.vmap(self.manifold.encode)(x_batch)

                in_bounds = jnp.all((z_batch >= latent_bounds[0]) & (z_batch <= latent_bounds[1]), axis=-1)
                latent_mask_chunks.append(in_bounds)

            latent_mask = jnp.concatenate(latent_mask_chunks, axis=0)
            master_mask = master_mask & latent_mask

        if not jnp.any(master_mask):
            raise ValueError("The requested bounds filtered out every row in the dataset!")

        # 3. Create the zero-copy sliced dataset
        sliced_dataset = self.dataset.filter(master_mask)

        # 4. Return either a zero-copy wrapper or a hard refit instance
        if not refit:
            return DesignSubspace(base_space=self, latent_bounds=latent_bounds, primal_bounds=primal_bounds)

        # Hard refit logic: Swap out the dataset and stream the sliced rows to the fitters
        temp_space = eqx.tree_at(lambda s: s.dataset, self, sliced_dataset)
        return temp_space.refit_models(
            batch_size=refit_kwargs.get("batch_size", 256),
            manifold_epochs=refit_kwargs.get("manifold_epochs", 100),
            surrogate_epochs=refit_kwargs.get("surrogate_epochs", 50),
        )


class DesignSubspace(Module):
    base_space: Any = _
    latent_bounds: Optional[tuple[jax.Array, jax.Array]] = _
    primal_bounds: Optional[dict[str, tuple[Optional[float], Optional[float]]]] = _

    def __getattr__(self, name: str):
        """
        Dynamically forwards properties (dataset, parameters, surrogate, etc.)
        and flat methods (get_dataloader, augment_data, refit_models).
        """
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        return getattr(self.base_space, name)

    def _clip_x(self, x: jax.Array) -> jax.Array:
        if self.latent_bounds is None:
            return x
        return jnp.clip(x, self.latent_bounds[0], self.latent_bounds[1])

    # --- Computational Overrides (Must intercept to enforce bounds) ---

    def project(self, physical_inputs: dict) -> jax.Array:
        x = self.base_space.project(physical_inputs)
        return self._clip_x(x)

    def reconstruct(self, x: jax.Array) -> dict:
        return self.base_space.reconstruct(self._clip_x(x))

    def evaluate(self, x: Optional[jax.Array] = None, compute_derivatives: bool = False):
        if x is not None:
            x = self._clip_x(x)
        return self.base_space.evaluate(x, compute_derivatives)

    def domain_trust(self, x: Optional[jax.Array] = None, **kwargs):
        if x is not None:
            x = self._clip_x(x)
        return self.base_space.domain_trust(x, **kwargs)

    def optimize(
        self,
        objective_fn: Callable,
        opt_vars: list[str],
        initial_values: Optional[dict[str, jax.Array]] = None,
        opt_kwargs: Optional[dict[str, Any]] = None,
        penalty_type: str | Callable = "quadratic_penalty",
        penalty_kwargs: Optional[dict[str, Any]] = None,
    ):
        if penalty_kwargs is None:
            penalty_kwargs = {"weight": 1e-3}
        penalty_weight = penalty_kwargs.pop("weight", 1e-3)

        penalty_fn = getattr(opt_funcs, penalty_type) if isinstance(penalty_type, str) else penalty_type

        def constrained_objective(preds, x, full_dict):
            base_obj = objective_fn(preds, x, full_dict)
            penalty = 0.0

            if self.latent_bounds is not None:
                penalty += penalty_fn(x, self.latent_bounds, **penalty_kwargs)

            if self.primal_bounds is not None:
                for var_name, (low, high) in self.primal_bounds.items():
                    if var_name in full_dict:
                        val = full_dict[var_name]
                        lb = low if low is not None else -jnp.inf
                        ub = high if high is not None else jnp.inf
                        penalty += penalty_fn(val, (lb, ub), **penalty_kwargs)

            return base_obj + penalty_weight * penalty

        return self.base_space.optimize(constrained_objective, opt_vars, initial_values, opt_kwargs)

    def explore(
        self,
        target_quantity: str,
        opt_vars: list[str],
        initial_values=None,
        method="ei",
        maximize=True,
        method_kwargs=None,
        opt_kwargs=None,
    ):
        """Borrows the objective from the base space, but runs it through our penalized optimizer!"""
        if method_kwargs is None:
            method_kwargs = {}

        # Get the bare acquisition objective from the probabilistic base
        obj_fn = self.base_space._get_explore_objective(target_quantity, method, maximize, method_kwargs)

        # Pass it to the SUBSPACE's optimize method to inject the boundary penalties
        return self.optimize(obj_fn, opt_vars, initial_values, opt_kwargs)

    def mcmc_log_likelihood(self, x: jax.Array, requirements: dict[str, tuple]) -> jax.Array:
        """Injects a hard uniform boundary prior for MCMC samplers."""
        base_ll = self.base_space.mcmc_log_likelihood(x, requirements)

        if self.latent_bounds is not None:
            in_bounds = jnp.all((x >= self.latent_bounds[0]) & (x <= self.latent_bounds[1]))
            base_ll += jnp.where(in_bounds, 0.0, -jnp.inf)

        return base_ll


class ActiveSpace(DesignSpace):
    """
    The advanced tier. Requires a probabilistic surrogate (e.g., SVGP, Deep Ensemble).
    Unlocks uncertainty quantification, Active Learning, and MCMC.
    """

    def evaluate(self, x: Optional[jax.Array] = None, compute_derivatives: bool = False) -> SurrogateEvaluation:
        preds = super().evaluate(x, compute_derivatives)
        if not preds.is_probabilistic:
            raise TypeError(f"Surrogate {type(self.surrogate).__name__} does not return variances.")
        return preds

    def domain_trust(self, x: Optional[jax.Array] = None, method: str = "exponential_trust", **kwargs) -> jax.Array:
        preds = self.evaluate(x)
        trust_fn = getattr(opt_funcs, method)
        return trust_fn(preds.variances, **kwargs)

    def evaluate_requirements(
        self, x: jax.Array, compute_derivatives: bool = False
    ) -> dict[str, RequirementEvaluation]:

        # 1. The traceable function that JAX will differentiate
        def _get_tracked_values(x_in):
            preds = self.evaluate(x_in, compute_derivatives=False)

            # Reconstruct primal dictionary
            full_means = self.reconstruct(x_in)
            full_vars = {}
            for i, p in enumerate(self.outputs):
                full_means[p.name] = preds.means[i]
                full_vars[p.name] = preds.variances[i]

            # Evaluate all requirements
            res_dict = {}
            ll_dict = {}
            for req in self.requirements:
                req_eval = req.evaluate(vals_dict=full_means, vars_dict=full_vars)
                res_dict[req.name] = req_eval.residual
                ll_dict[req.name] = req_eval.log_likelihood

            return res_dict, ll_dict

        # 2. Compute the actual base values
        res_vals, ll_vals = _get_tracked_values(x)

        # 3. Compute the Jacobians using JAX reverse-mode autodiff
        if compute_derivatives:
            # jacrev perfectly handles functions returning tuples of dictionaries
            jacobians_res, jacobians_ll = jax.jacrev(_get_tracked_values)(x)
        else:
            jacobians_res, jacobians_ll = {}, {}

        # 4. Pack everything neatly into the Return objects
        preds = self.evaluate(x, compute_derivatives=compute_derivatives)
        full_means = self.reconstruct(x)
        full_vars = {p.name: preds.variances[i] for i, p in enumerate(self.outputs)}
        for i, p in enumerate(self.outputs):
            full_means[p.name] = preds.means[i]

        results = {}
        for req in self.requirements:
            req_eval = req.evaluate(vals_dict=full_means, vars_dict=full_vars)

            if compute_derivatives:
                # Add the residual derivative
                req_eval = eqx.tree_at(lambda e: e.derivatives, req_eval, jacobians_res[req.name])

                # Optional: We can add the log_likelihood derivative as well if we extend RequirementEvaluation
                # req_eval = eqx.tree_at(lambda e: e.ll_derivatives, req_eval, jacobians_ll[req.name])

            results[req.name] = req_eval

        return results

    def sample_feasible(
        self,
        num_samples: int = 1000,
        method: str = "dataset",
        key: Optional[jax.Array] = None,
        num_warmup: int = 500,
        **kwargs,
    ) -> jax.Array:

        if method != "mcmc":
            return super().sample_feasible(num_samples=num_samples, **kwargs)

        if key is None:
            raise ValueError("An explicit PRNG key is required for MCMC sampling.")

        target_reqs = kwargs.get("target_requirements", [r.name for r in self.requirements])

        def logprob_fn(x):
            log_prior = jnp.sum(norm.logpdf(x, loc=0.0, scale=1.0))
            req_evals = self.evaluate_requirements(x)
            log_likelihood = jnp.sum(jnp.array([req_evals[name].log_likelihood for name in target_reqs]))
            return log_prior + log_likelihood

        warmup_key, sample_key = jax.random.split(key)
        initial_position = jnp.zeros(self.manifold.latent_dim)

        # 1. Adapt phase (pass num_warmup as a positional argument)
        adapt = blackjax.window_adaptation(blackjax.nuts, logprob_fn)
        (last_state, parameters), _ = adapt.run(warmup_key, initial_position, num_steps=num_warmup)  # type: ignore

        # 2. Kernel setup
        kernel = blackjax.nuts(logprob_fn, **parameters).step

        @jax.jit
        def inference_loop(rng_key, state: HMCState, total_steps: int):
            def step_fn(carry, _):
                key_step, curr_state = carry
                key_step, subkey = jax.random.split(key_step)

                # Unpack the step return. Explicitly hint that next_state is an HMCState
                next_state, info = kernel(subkey, curr_state)  # type: ignore
                next_state: HMCState = next_state

                return (key_step, next_state), next_state.position

            _, positions = jax.lax.scan(step_fn, (rng_key, state), jnp.arange(total_steps))
            return positions

        return inference_loop(sample_key, last_state, num_samples)

    def _get_explore_objective(self, target_quantity: str, method: str, maximize: bool, method_kwargs: dict):
        """Helper to construct the Active Learning objective without executing it."""
        acq_fn = getattr(opt_funcs, method)
        out_names = [p.name for p in self.outputs]
        target_idx = out_names.index(target_quantity)

        y_known = self.Y[:, target_idx]
        y_best = float(jnp.max(y_known) if maximize else jnp.min(y_known))

        def acq_objective(preds, x, full_dict):
            mu = preds.means[target_idx]
            sigma = jnp.sqrt(jnp.maximum(preds.variances[target_idx], 0.0))
            acq_value = acq_fn(mu, sigma, y_best, maximize, **method_kwargs)
            return -acq_value  # Minimizer assumes negative acquisition

        return acq_objective

    def explore(
        self,
        target_quantity: str,
        opt_vars: list[str],
        initial_values: Optional[dict[str, jax.Array]] = None,
        method: str = "ei",
        maximize: bool = True,
        method_kwargs: Optional[dict[str, Any]] = None,
        opt_kwargs: Optional[dict[str, Any]] = None,
    ) -> dict[str, jax.Array]:
        """Active Learning execution via standard optimize loop."""
        if method_kwargs is None:
            method_kwargs = {}

        # Build the objective, then route it right back through our own optimizer!
        obj_fn = self._get_explore_objective(target_quantity, method, maximize, method_kwargs)
        return self.optimize(obj_fn, opt_vars, initial_values, opt_kwargs)

    def explore_boundaries(
        self,
        target_requirement: str,
        opt_vars: list[str],
        initial_values: Optional[dict[str, jax.Array]] = None,
        method: str = "binary_entropy",
        method_kwargs: Optional[dict] = None,
        opt_kwargs: Optional[dict] = None,
    ) -> dict[str, jax.Array]:
        """
        Active Learning for Constraint Mapping.
        Searches for specific probability contours (like maximum confusion at P=0.5).
        """
        if method_kwargs is None:
            method_kwargs = {}
        if opt_kwargs is None:
            opt_kwargs = {"rtol": 1e-5, "atol": 1e-5}

        # Dynamically fetch from the new registry
        acq_fn = getattr(opt_funcs, method)

        req_names = [r.name for r in self.requirements]
        if target_requirement not in req_names:
            raise ValueError(f"Requirement '{target_requirement}' not found. Available: {req_names}")

        def boundary_objective(preds, x, full_dict):
            req_evals = self.evaluate_requirements(x)
            p = req_evals[target_requirement].probability

            acq_value = acq_fn(p, **method_kwargs)

            # optimistix minimizes, so we return negative acquisition to maximize it
            return -acq_value

        return self.optimize(boundary_objective, opt_vars, initial_values, opt_kwargs)

    def augment_data(
        self,
        augment_fn: Callable,
        new_latents: Optional[jax.Array] = None,
        target_quantity: Optional[str] = None,
        opt_vars: Optional[list[str]] = None,
        method: str = "ei",
        maximize: bool = True,
        fixed_values: Optional[dict[str, jax.Array]] = None,
        refit: bool = True,
        explore_kwargs: Optional[dict] = None,
        refit_kwargs: Optional[dict] = None,
    ) -> "ActiveSpace":
        """
        Closed-loop Active Learning step.
        If new_latents is None, automatically explores to find the optimal next point.
        """
        if new_latents is None:
            if target_quantity is None or opt_vars is None:
                raise ValueError("Must provide target_quantity and opt_vars if new_latents is None.")

            explore_kwargs = explore_kwargs or {}

            # 1. Run the Active Learning acquisition optimizer
            proposed_dict = self.explore(
                target_quantity=target_quantity,
                opt_vars=opt_vars,
                initial_values=fixed_values,
                method=method,
                maximize=maximize,
                **explore_kwargs,
            )

            # 2. Project the optimal physical parameters back into the latent space
            full_dict = {p.name: p.value for p in self.inputs}
            if fixed_values is not None:
                full_dict.update(fixed_values)
            full_dict.update(proposed_dict)
            new_latents = self.project(full_dict)

        # 3. Route through the base class for Oracle evaluation and dataset appending
        new_space = super().augment_data(new_latents, augment_fn, fixed_values)

        # 4. Automatically close the loop by refitting the models on the newly appended data
        if refit:
            refit_kwargs = refit_kwargs or {}
            return new_space.refit_models(**refit_kwargs)  # type: ignore

        return new_space  # type: ignore
