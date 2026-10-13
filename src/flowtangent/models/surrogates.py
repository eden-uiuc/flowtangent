import abc
from typing import List, Optional, TypeAlias, Union

import equinox as eqx
import jax
import jax.numpy as jnp

from ..utils import Module
from ..utils.data import DataLoader, LatentDataLoader
from ..utils.typing import _
from .kernels import Kernel
from .nn import TransformerBlock, fit_neural_model

_loader: TypeAlias = DataLoader | LatentDataLoader


# --- DataLoader Compatibility Helper ---
def _get_full_XY(data: Union[tuple[jax.Array, jax.Array], _loader]) -> tuple[jax.Array, jax.Array]:
    r"""Collect feature and target arrays from paired training data.

    A loader is consumed for one iteration and its batches are concatenated in
    yielded order. Directly supplied pairs retain their original arrays.

    ### Mathematical Formulation
    $X = concat_0(X_1, \ldots, X_b)$ and $Y = concat_0(Y_1, \ldots, Y_b)$.

    ### Process Architecture
    Collect each batch's features and targets, then concatenate each along axis zero.

    ### Usage Example
    ```python
    import jax.numpy as jnp
    from flowtangent.models.surrogates import _get_full_XY

    x, y = _get_full_XY((jnp.ones((2, 3)), jnp.zeros((2, 1))))
    ```

    Args:
        data (tuple[jax.Array, jax.Array] | DataLoader | LatentDataLoader):
            An `(X, Y)` pair, normally with 2D arrays `(N, D)` and `(N, O)`, or
            a loader yielding pairs with compatible non-batch dimensions.

    Returns:
        tuple[jax.Array, jax.Array]: Features and targets. Loader data is
            materialized in memory; direct pairs are returned without conversion.

    Raises:
        IndexError: If data or a loader batch lacks a target item.
        ValueError: If the loader yields no batches.
        TypeError: If batch array shapes are incompatible for concatenation.
    """
    if hasattr(data, "dataset") and hasattr(data, "__iter__"):
        X_list, Y_list = [], []
        for batch in data:
            X_list.append(batch[0])
            Y_list.append(batch[1])
        return jnp.concatenate(X_list, axis=0), jnp.concatenate(Y_list, axis=0)
    return data[0], data[1]


# --- Base State and Classes ---
class SurrogateEvaluation(Module):
    """The universal, strongly-typed return payload for all surrogates."""

    means: jax.Array = _
    variances: Optional[jax.Array] = None
    covariance: Optional[jax.Array] = None
    samples: Optional[jax.Array] = None
    derivatives: Optional[jax.Array] = None

    @property
    def is_probabilistic(self) -> bool:
        return self.variances is not None or self.samples is not None


class Surrogate(Module):
    @abc.abstractmethod
    def predict(self, x: jax.Array, compute_derivatives: bool = False) -> SurrogateEvaluation:
        pass

    @abc.abstractmethod
    def fit(self, data: Union[tuple[jax.Array, jax.Array], _loader], **kwargs) -> "Surrogate":
        r"""Define the fitting interface for surrogate implementations.

        Subclasses provide the training rule; this abstract body performs no fitting.

        ### Mathematical Formulation
        Concrete implementations determine the map $(M, X, Y) \mapsto M'$.

        ### Usage Example
        ```python
        import jax.numpy as jnp
        from flowtangent.models.surrogates import (
            InverseDistanceWeighting,
        )

        model = InverseDistanceWeighting(1, 1)
        fitted = model.fit((jnp.zeros((2, 1)), jnp.ones((2, 1))))
        ```

        Args:
            data (tuple[jax.Array, jax.Array] | DataLoader | LatentDataLoader):
                Feature and target arrays, conventionally 2D `(N, D)` and `(N, O)`,
                or a loader yielding corresponding batches.
            **kwargs (Any): Training options interpreted by the concrete implementation.

        Returns:
            Surrogate: The fitted model supplied by the concrete implementation.
        """
        pass


# ==========================================
# 1. Classical & Statistical Surrogates
# ==========================================


class PolynomialChaosSurrogate(Surrogate):
    weights: eqx.nn.Linear = _
    degree: int = _

    def __init__(self, key, input_dim: int, output_dim: int, degree: int = 3):
        self.degree = degree
        poly_features = input_dim * degree
        self.weights = eqx.nn.Linear(poly_features, output_dim, use_bias=True, key=key)

    def _polynomial_basis(self, x):
        bases = [x**d for d in range(1, self.degree + 1)]
        return jnp.concatenate(bases, axis=-1)

    def _internal_predict(self, x):
        poly_x = self._polynomial_basis(x)
        return self.weights(poly_x)

    def predict(self, x, compute_derivatives=False):
        means = self._internal_predict(x)
        preds = SurrogateEvaluation(means=means)
        if compute_derivatives:
            preds = eqx.tree_at(
                lambda p: p.derivatives, preds, jax.jacrev(self._internal_predict)(x), is_leaf=lambda x: x is None
            )
        return preds

    def fit(
        self,
        data: Union[tuple[jax.Array, jax.Array], _loader],
        learning_rate: float = 1e-2,
        epochs: int = 500,
        **kwargs,
    ) -> "PolynomialChaosSurrogate":
        r"""Fit polynomial-basis coefficients by minimizing mean squared error.

        The current parameters provide the starting point for Adam training.
        Training returns a fitted model without modifying this instance.

        ### Mathematical Formulation
        $L = mean((Y-f_\theta(X))^2)$, where the basis uses elementwise powers
        from one through the configured degree.

        ### Process Architecture
        Apply Adam updates once per loader batch per epoch, or once per epoch for array pairs.

        ### Usage Example
        ```python
        import jax
        import jax.numpy as jnp
        from flowtangent.models.surrogates import (
            PolynomialChaosSurrogate,
        )

        model = PolynomialChaosSurrogate(
            jax.random.PRNGKey(0), 1, 1
        )
        fitted = model.fit(
            (jnp.zeros((2, 1)), jnp.ones((2, 1))), epochs=1
        )
        ```

        Args:
            data (tuple[jax.Array, jax.Array] | DataLoader | LatentDataLoader):
                2D feature and target arrays `(N, D)` and `(N, O)`, or a loader yielding
                such batches.
            learning_rate (float): Adam step size. Defaults to `1e-2`.
            epochs (int): Number of passes over the data. Defaults to `500`;
                a nonpositive count performs no updates.
            **kwargs (Any): Ignored.

        Returns:
            PolynomialChaosSurrogate: Model after training; returns the original instance if
                no updates run. Loader and training errors propagate.

        Raises:
            TypeError: If target and prediction shapes cannot broadcast in the loss.
            jaxtyping.TypeCheckError: If arguments violate the method annotations.
        """

        def loss_fn(model, batch):
            x_batch, y_batch = batch

            preds = jax.vmap(model._internal_predict)(x_batch)
            return jnp.mean((y_batch - preds) ** 2)

        return fit_neural_model(self, data, loss_fn, learning_rate, epochs)


class InverseDistanceWeighting(Surrogate):
    X_fit: jax.Array = _
    Y_fit: jax.Array = _
    power: float = _

    def __init__(self, input_dim: int, output_dim: int, power: float = 2.0):
        self.X_fit = jnp.zeros((1, input_dim))
        self.Y_fit = jnp.zeros((1, output_dim))
        self.power = power

    def _internal_predict(self, x):
        distances = jnp.linalg.norm(self.X_fit - x, axis=-1)
        weights = 1.0 / (distances**self.power + 1e-8)
        weights /= jnp.sum(weights)
        return jnp.sum(weights[:, None] * self.Y_fit, axis=0)

    def predict(self, x, compute_derivatives=False):
        means = self._internal_predict(x)
        preds = SurrogateEvaluation(means=means)
        if compute_derivatives:
            preds = eqx.tree_at(
                lambda p: p.derivatives, preds, jax.jacrev(self._internal_predict)(x), is_leaf=lambda x: x is None
            )
        return preds

    def fit(self, data: Union[tuple[jax.Array, jax.Array], _loader], **kwargs) -> "InverseDistanceWeighting":
        """Store training samples for inverse-distance interpolation.

        Fitting replaces the reference features and targets without optimizing parameters.
        Loaders are materialized in their yielded order.

        ### Mathematical Formulation
        $X_{fit} = X$ and $Y_{fit} = Y$; the distance-weighting power is unchanged.

        ### Process Architecture
        Collect paired arrays and replace the reference data in a new model.

        ### Usage Example
        ```python
        import jax.numpy as jnp
        from flowtangent.models.surrogates import (
            InverseDistanceWeighting,
        )

        fitted = InverseDistanceWeighting(1, 1).fit(
            (jnp.zeros((2, 1)), jnp.ones((2, 1)))
        )
        ```

        Args:
            data (tuple[jax.Array, jax.Array] | DataLoader | LatentDataLoader):
                2D features `(N, D)` and targets `(N, O)`, or a loader yielding pairs.
            **kwargs (Any): Ignored.

        Returns:
            InverseDistanceWeighting: A new model containing the training arrays.

        Raises:
            jaxtyping.TypeCheckError: If arguments violate the method annotations.
            IndexError: If a loader batch lacks a target item.
            ValueError: If the loader yields no batches.
            TypeError: If loader batch shapes are incompatible for concatenation.
        """
        X_full, Y_full = _get_full_XY(data)

        return eqx.tree_at(lambda m: (m.X_fit, m.Y_fit), self, (X_full, Y_full))


class RadialBasisFunction(Surrogate):
    centers: jax.Array = _
    weights: jax.Array = _
    lengthscale: float = _

    def __init__(self, input_dim: int, output_dim: int, lengthscale: float = 1.0):
        self.centers = jnp.zeros((1, input_dim))
        self.weights = jnp.zeros((1, output_dim))
        self.lengthscale = lengthscale

    def _internal_predict(self, x):
        r = jnp.linalg.norm(self.centers - x, axis=-1)
        phi = jnp.exp(-0.5 * (r / self.lengthscale) ** 2)
        return self.weights.T @ phi

    def predict(self, x, compute_derivatives=False):
        means = self._internal_predict(x)
        preds = SurrogateEvaluation(means=means)
        if compute_derivatives:
            preds = eqx.tree_at(
                lambda p: p.derivatives, preds, jax.jacrev(self._internal_predict)(x), is_leaf=lambda x: x is None
            )
        return preds

    def fit(self, data: Union[tuple[jax.Array, jax.Array], _loader], **kwargs) -> "RadialBasisFunction":
        r"""Fit Gaussian radial-basis weights to paired training data.

        Each training feature becomes a center. A fixed diagonal ridge regularizes
        the interpolation system, and the original model is unchanged.

        ### Mathematical Formulation
        $\Phi_{ij} = \exp(-\|X_i-X_j\|^2/(2\ell^2))$ and
        $W = (\Phi + 10^{-6}I)^{-1}Y$, using the current lengthscale $\ell$.

        ### Process Architecture
        Materialize pairs, form the regularized kernel matrix, and solve for weights.

        ### Usage Example
        ```python
        import jax.numpy as jnp
        from flowtangent.models.surrogates import (
            RadialBasisFunction,
        )

        fitted = RadialBasisFunction(1, 1).fit(
            (jnp.array([[0.0], [1.0]]), jnp.ones((2, 1)))
        )
        ```

        Args:
            data (tuple[jax.Array, jax.Array] | DataLoader | LatentDataLoader):
                2D features `(N, D)` and targets `(N, O)`, or a loader yielding pairs.
            **kwargs (Any): Ignored.

        Returns:
            RadialBasisFunction: A new model with training centers and solved weights.

        Raises:
            jaxtyping.TypeCheckError: If arguments violate the method annotations.
            IndexError: If a loader batch lacks a target item.
            ValueError: If the loader yields no batches or targets have an incompatible
                leading dimension for the linear solve.
            TypeError: If loader batch shapes are incompatible for concatenation.
        """
        X_full, Y_full = _get_full_XY(data)

        def phi_fn(x1, x2):
            return jnp.exp(-0.5 * jnp.sum((x1 - x2) ** 2) / self.lengthscale**2)

        Phi = jax.vmap(lambda x1: jax.vmap(lambda x2: phi_fn(x1, x2))(X_full))(X_full)
        Phi_ridge = Phi + 1e-6 * jnp.eye(X_full.shape[0])
        W = jnp.linalg.solve(Phi_ridge, Y_full)

        return eqx.tree_at(lambda m: (m.centers, m.weights), self, (X_full, W))


class GaussianProcessSurrogate(Surrogate):
    inducing_points: jax.Array = _
    kernel: Kernel = _
    variational_mean: jax.Array = _
    noise_variance: jax.Array = _

    def __init__(self, key, input_dim: int, output_dim: int, kernel: Kernel, num_inducing: int = 128):
        k1, k2 = jax.random.split(key)
        self.kernel = kernel
        self.inducing_points = jax.random.uniform(k1, (num_inducing, input_dim))
        self.variational_mean = jax.random.normal(k2, (num_inducing, output_dim)) * 1e-2
        self.noise_variance = jnp.zeros(output_dim)

    def predict(self, x, compute_derivatives=False):
        # 1. Cross-covariance
        K_xz = jax.vmap(lambda z: self.kernel(x, z))(self.inducing_points)
        means = K_xz @ self.variational_mean

        # --- NEW VARIANCE CALCULATION ---
        # 2. Inducing point covariance matrix
        K_zz = jax.vmap(lambda z1: jax.vmap(lambda z2: self.kernel(z1, z2))(self.inducing_points))(self.inducing_points)

        # Add numerical jitter to the diagonal for Cholesky stability
        K_zz += 1e-5 * jnp.eye(self.inducing_points.shape[0])

        # 3. Cholesky decomposition (lower triangular)
        L = jax.scipy.linalg.cholesky(K_zz, lower=True)

        # 4. Solve L * v = K_zx for v
        v = jax.scipy.linalg.solve_triangular(L, K_xz, lower=True)

        # 5. Correct variance calculation
        k_xx = self.kernel(x, x)
        var = k_xx - jnp.sum(v**2) + jnp.exp(self.noise_variance)

        preds = SurrogateEvaluation(means=means, variances=var)

        if compute_derivatives:
            mean_fn = lambda x_in: (
                jax.vmap(lambda z: self.kernel(x_in, z))(self.inducing_points) @ self.variational_mean
            )
            preds = eqx.tree_at(lambda p: p.derivatives, preds, jax.jacrev(mean_fn)(x), is_leaf=lambda x: x is None)

        return preds

    def fit(
        self,
        data: Union[tuple[jax.Array, jax.Array], "_loader"],
        learning_rate: float = 1e-2,
        epochs: int = 500,
        **kwargs,
    ) -> "GaussianProcessSurrogate":
        r"""Fit inducing-point predictions with a Gaussian likelihood objective.

        The current parameters provide the starting point for Adam training.
        Training returns a fitted model without modifying this instance.

        ### Mathematical Formulation
        $L = \frac12\sum\log V + \frac12\sum (Y-\mu)^2/V
        + \frac12\sum A^2$, where $A$ is the variational mean and $V$ is the
        predictive variance.

        ### Process Architecture
        Apply Adam updates once per loader batch per epoch, or once per epoch for array pairs.

        ### Usage Example
        ```python
        import jax
        import jax.numpy as jnp
        from flowtangent.models.kernels import RBFKernel
        from flowtangent.models.surrogates import (
            GaussianProcessSurrogate,
        )

        model = GaussianProcessSurrogate(
            jax.random.PRNGKey(0),
            1,
            1,
            RBFKernel(1),
            num_inducing=2,
        )
        fitted = model.fit(
            (jnp.zeros((2, 1)), jnp.ones((2, 1))), epochs=1
        )
        ```

        Args:
            data (tuple[jax.Array, jax.Array] | DataLoader | LatentDataLoader):
                2D feature and target arrays `(N, D)` and `(N, O)`, or a loader yielding
                such batches.
            learning_rate (float): Adam step size. Defaults to `1e-2`.
            epochs (int): Number of passes over the data. Defaults to `500`;
                a nonpositive count performs no updates.
            **kwargs (Any): Ignored.

        Returns:
            GaussianProcessSurrogate: Model after training; returns the original instance if
                no updates run. Loader and training errors propagate.

        Raises:
            TypeError: If target and prediction shapes cannot broadcast in the loss.
            jaxtyping.TypeCheckError: If arguments violate the method annotations.
        """

        def loss_fn(model, batch):
            x_train, y_train = batch

            # 1. Use the model's own predict method via vmap to ensure
            #    the inducing points and variational mean are actively trained.
            #    XLA is smart enough to compile this efficiently.
            preds = jax.vmap(model.predict, in_axes=(0, None))(x_train, False)
            means = preds.means  # Shape: (N, D)
            variances = preds.variances  # Shape: (N, D)

            # 2. Compute the Negative Log Likelihood (NLL)
            #    This correctly broadcasts the variances across the output dimensions.
            diff = y_train - means
            data_fit = 0.5 * jnp.sum(jnp.log(variances)) + 0.5 * jnp.sum((diff**2) / variances)

            # 3. Add a complexity penalty (KL divergence proxy) for the variational mean.
            #    This prevents the sparse weights from exploding during SGD.
            complexity_penalty = 0.5 * jnp.sum(model.variational_mean**2)

            return data_fit + complexity_penalty

        return fit_neural_model(self, data, loss_fn, learning_rate, epochs)


# ==========================================
# 2. Deep Learning Surrogates
# ==========================================


class MLPSurrogate(Surrogate):
    network: eqx.nn.MLP = _

    def __init__(self, key, input_dim: int, output_dim: int, width: int = 256, depth: int = 4):
        self.network = eqx.nn.MLP(input_dim, output_dim, width, depth, activation=jax.nn.silu, key=key)

    def _internal_predict(self, x):
        return self.network(x)

    def predict(self, x, compute_derivatives=False):
        means = self._internal_predict(x)
        preds = SurrogateEvaluation(means=means)
        if compute_derivatives:
            preds = eqx.tree_at(
                lambda p: p.derivatives, preds, jax.jacrev(self._internal_predict)(x), is_leaf=lambda x: x is None
            )
        return preds

    def fit(
        self,
        data: Union[tuple[jax.Array, jax.Array], _loader],
        learning_rate: float = 1e-3,
        epochs: int = 1000,
        **kwargs,
    ) -> "MLPSurrogate":
        r"""Fit the neural network by minimizing mean squared error.

        The current parameters provide the starting point for Adam training.
        Training returns a fitted model without modifying this instance.

        ### Mathematical Formulation
        $L = mean((Y-f_\theta(X))^2)$.

        ### Process Architecture
        Apply Adam updates once per loader batch per epoch, or once per epoch for array pairs.

        ### Usage Example
        ```python
        import jax
        import jax.numpy as jnp
        from flowtangent.models.surrogates import MLPSurrogate

        model = MLPSurrogate(
            jax.random.PRNGKey(0), 1, 1, width=4, depth=1
        )
        fitted = model.fit(
            (jnp.zeros((2, 1)), jnp.ones((2, 1))), epochs=1
        )
        ```

        Args:
            data (tuple[jax.Array, jax.Array] | DataLoader | LatentDataLoader):
                2D feature and target arrays `(N, D)` and `(N, O)`, or a loader yielding
                such batches.
            learning_rate (float): Adam step size. Defaults to `1e-3`.
            epochs (int): Number of passes over the data. Defaults to `1000`;
                a nonpositive count performs no updates.
            **kwargs (Any): Ignored.

        Returns:
            MLPSurrogate: Model after training; returns the original instance if
                no updates run. Loader and training errors propagate.

        Raises:
            TypeError: If target and prediction shapes cannot broadcast in the loss.
            jaxtyping.TypeCheckError: If arguments violate the method annotations.
        """

        def loss_fn(model, batch):
            x_batch, y_batch = batch
            preds = jax.vmap(model._internal_predict)(x_batch)
            return jnp.mean((y_batch - preds) ** 2)

        return fit_neural_model(self, data, loss_fn, learning_rate, epochs)


class TransformerSurrogate(Surrogate):
    latent_dim: int = _
    latent_embed: eqx.nn.Linear = _
    context_embed: eqx.nn.Linear = _
    blocks: List[TransformerBlock] = _
    head: eqx.nn.Linear = _

    def __init__(self, key, latent_dim: int, context_dim: int, output_dim: int, hidden_size: int = 128):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.latent_dim = latent_dim
        self.latent_embed = eqx.nn.Linear(latent_dim, hidden_size, key=k1)
        self.context_embed = eqx.nn.Linear(context_dim, hidden_size, key=k2)

        block_keys = jax.random.split(k3, 4)
        self.blocks = [TransformerBlock(hidden_size, 4, bk) for bk in block_keys]
        self.head = eqx.nn.Linear(hidden_size, output_dim, key=k4)

    def _internal_predict(self, x):
        z, c = jnp.split(x, [self.latent_dim], axis=-1)
        z_tok = self.latent_embed(z)
        c_tok = self.context_embed(c)
        seq = jnp.stack([z_tok, c_tok], axis=0)

        for block in self.blocks:
            seq = block(seq)

        pooled = jnp.mean(seq, axis=0)
        return self.head(pooled)

    def predict(self, x, compute_derivatives=False):
        means = self._internal_predict(x)
        preds = SurrogateEvaluation(means=means)
        if compute_derivatives:
            preds = eqx.tree_at(
                lambda p: p.derivatives, preds, jax.jacrev(self._internal_predict)(x), is_leaf=lambda x: x is None
            )
        return preds

    def fit(
        self,
        data: Union[tuple[jax.Array, jax.Array], _loader],
        learning_rate: float = 1e-3,
        epochs: int = 500,
        **kwargs,
    ) -> "TransformerSurrogate":
        r"""Fit latent-and-context predictions by minimizing mean squared error.

        The current parameters provide the starting point for Adam training.
        Training returns a fitted model without modifying this instance.

        ### Mathematical Formulation
        $L = mean((Y-f_\theta(X))^2)$; each feature row concatenates latent
        coordinates followed by context coordinates.

        ### Process Architecture
        Apply Adam updates once per loader batch per epoch, or once per epoch for array pairs.

        ### Usage Example
        ```python
        import jax
        import jax.numpy as jnp
        from flowtangent.models.surrogates import (
            TransformerSurrogate,
        )

        model = TransformerSurrogate(
            jax.random.PRNGKey(0), 1, 1, 1, hidden_size=4
        )
        fitted = model.fit(
            (jnp.zeros((2, 2)), jnp.ones((2, 1))), epochs=1
        )
        ```

        Args:
            data (tuple[jax.Array, jax.Array] | DataLoader | LatentDataLoader):
                2D feature and target arrays `(N, D)` and `(N, O)`, or a loader yielding
                such batches. Feature rows concatenate latent and context coordinates.
            learning_rate (float): Adam step size. Defaults to `1e-3`.
            epochs (int): Number of passes over the data. Defaults to `500`;
                a nonpositive count performs no updates.
            **kwargs (Any): Ignored.

        Returns:
            TransformerSurrogate: Model after training; returns the original instance if
                no updates run. Loader and training errors propagate.

        Raises:
            TypeError: If target and prediction shapes cannot broadcast in the loss.
            jaxtyping.TypeCheckError: If arguments violate the method annotations.
        """

        def loss_fn(model, batch):
            x_batch, y_batch = batch

            preds = jax.vmap(model._internal_predict)(x_batch)
            return jnp.mean((y_batch - preds) ** 2)

        return fit_neural_model(self, data, loss_fn, learning_rate, epochs)


class DeepEnsembleSurrogate(Surrogate):
    models: eqx.nn.MLP = _

    def __init__(self, key, num_models: int, input_dim: int, output_dim: int):
        keys = jax.random.split(key, num_models)
        self.models = eqx.filter_vmap(lambda k: eqx.nn.MLP(input_dim, output_dim, 128, 3, key=k))(keys)

    def _internal_predict(self, x):
        return eqx.filter_vmap(lambda m: m(x))(self.models)

    def predict(self, x, compute_derivatives=False):
        samples = self._internal_predict(x)
        means = jnp.mean(samples, axis=0)
        variances = jnp.var(samples, axis=0)

        preds = SurrogateEvaluation(means=means, variances=variances, samples=samples)
        if compute_derivatives:
            mean_fn = lambda x_in: jnp.mean(self._internal_predict(x_in), axis=0)
            preds = eqx.tree_at(lambda p: p.derivatives, preds, jax.jacrev(mean_fn)(x), is_leaf=lambda x: x is None)
        return preds

    def fit(
        self,
        data: Union[tuple[jax.Array, jax.Array], _loader],
        learning_rate: float = 1e-3,
        epochs: int = 500,
        **kwargs,
    ) -> "DeepEnsembleSurrogate":
        r"""Fit all ensemble members against the same target batches.

        The current parameters provide the starting point for Adam training.
        Training returns a fitted model without modifying this instance.

        ### Mathematical Formulation
        $L = mean_{n,m,o}((Y_{no}-f_{\theta_m}(X_n)_o)^2)$; every member
        receives the same samples rather than a separate bootstrap sample.

        ### Process Architecture
        Apply Adam updates once per loader batch per epoch, or once per epoch for array pairs.

        ### Usage Example
        ```python
        import jax
        import jax.numpy as jnp
        from flowtangent.models.surrogates import (
            DeepEnsembleSurrogate,
        )

        model = DeepEnsembleSurrogate(
            jax.random.PRNGKey(0), 2, 1, 1
        )
        fitted = model.fit(
            (jnp.zeros((2, 1)), jnp.ones((2, 1))), epochs=1
        )
        ```

        Args:
            data (tuple[jax.Array, jax.Array] | DataLoader | LatentDataLoader):
                2D feature and target arrays `(N, D)` and `(N, O)`, or a loader yielding
                such batches.
            learning_rate (float): Adam step size. Defaults to `1e-3`.
            epochs (int): Number of passes over the data. Defaults to `500`;
                a nonpositive count performs no updates.
            **kwargs (Any): Ignored.

        Returns:
            DeepEnsembleSurrogate: Model after training; returns the original instance if
                no updates run. Loader and training errors propagate.

        Raises:
            TypeError: If target and prediction shapes cannot broadcast in the loss.
            jaxtyping.TypeCheckError: If arguments violate the method annotations.
        """

        def loss_fn(model, batch):
            x_batch, y_batch = batch

            all_preds = jax.vmap(model._internal_predict)(x_batch)
            return jnp.mean((jnp.expand_dims(y_batch, 1) - all_preds) ** 2)

        return fit_neural_model(self, data, loss_fn, learning_rate, epochs)


class MixtureOfExperts(Surrogate):
    gating_network: eqx.nn.Linear = _
    experts: eqx.nn.MLP = _

    def __init__(self, key, num_experts: int, input_dim: int, output_dim: int):
        k1, k2 = jax.random.split(key)
        self.gating_network = eqx.nn.Linear(input_dim, num_experts, key=k1)
        keys = jax.random.split(k2, num_experts)
        self.experts = eqx.filter_vmap(lambda k: eqx.nn.MLP(input_dim, output_dim, 64, 2, key=k))(keys)

    def _internal_predict(self, x):
        gate_logits = self.gating_network(x)
        routing_weights = jax.nn.softmax(gate_logits)
        expert_outputs = eqx.filter_vmap(lambda m: m(x))(self.experts)
        return jnp.sum(routing_weights[:, None] * expert_outputs, axis=0)

    def predict(self, x, compute_derivatives=False):
        means = self._internal_predict(x)
        preds = SurrogateEvaluation(means=means)
        if compute_derivatives:
            preds = eqx.tree_at(
                lambda p: p.derivatives, preds, jax.jacrev(self._internal_predict)(x), is_leaf=lambda x: x is None
            )
        return preds

    def fit(
        self,
        data: Union[tuple[jax.Array, jax.Array], _loader],
        learning_rate: float = 1e-3,
        epochs: int = 500,
        **kwargs,
    ) -> "MixtureOfExperts":
        r"""Fit the gating network and experts using mean squared error.

        The current parameters provide the starting point for Adam training.
        Training returns a fitted model without modifying this instance.

        ### Mathematical Formulation
        $L = mean((Y-\sum_m g_m(X)f_m(X))^2)$, where $g$ is the
        softmax routing distribution.

        ### Process Architecture
        Apply Adam updates once per loader batch per epoch, or once per epoch for array pairs.

        ### Usage Example
        ```python
        import jax
        import jax.numpy as jnp
        from flowtangent.models.surrogates import MixtureOfExperts

        model = MixtureOfExperts(jax.random.PRNGKey(0), 2, 1, 1)
        fitted = model.fit(
            (jnp.zeros((2, 1)), jnp.ones((2, 1))), epochs=1
        )
        ```

        Args:
            data (tuple[jax.Array, jax.Array] | DataLoader | LatentDataLoader):
                2D feature and target arrays `(N, D)` and `(N, O)`, or a loader yielding
                such batches.
            learning_rate (float): Adam step size. Defaults to `1e-3`.
            epochs (int): Number of passes over the data. Defaults to `500`;
                a nonpositive count performs no updates.
            **kwargs (Any): Ignored.

        Returns:
            MixtureOfExperts: Model after training; returns the original instance if
                no updates run. Loader and training errors propagate.

        Raises:
            TypeError: If target and prediction shapes cannot broadcast in the loss.
            jaxtyping.TypeCheckError: If arguments violate the method annotations.
        """

        def loss_fn(model, batch):
            x_batch, y_batch = batch

            preds = jax.vmap(model._internal_predict)(x_batch)
            return jnp.mean((y_batch - preds) ** 2)

        return fit_neural_model(self, data, loss_fn, learning_rate, epochs)


class SpectralConv1d(Module):
    weights_real: jax.Array = _
    weights_imag: jax.Array = _
    modes: int = _

    def __init__(self, key, in_channels, out_channels, modes):
        k1, k2 = jax.random.split(key)
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weights_real = jax.random.normal(k1, (in_channels, out_channels, modes)) * scale
        self.weights_imag = jax.random.normal(k2, (in_channels, out_channels, modes)) * scale

    def __call__(self, x):
        x_ft = jnp.fft.rfft(x)
        complex_weights = self.weights_real + 1j * self.weights_imag
        out_ft = jnp.zeros((complex_weights.shape[1], x_ft.shape[-1]), dtype=jnp.complex64)
        out_ft_modes = jnp.einsum("im,iom->om", x_ft[:, : self.modes], complex_weights)
        out_ft = out_ft.at[:, : self.modes].set(out_ft_modes)
        return jnp.fft.irfft(out_ft, n=x.shape[-1])


class FourierNeuralOperator(Surrogate):
    lifting: eqx.nn.Linear
    spectral_conv: SpectralConv1d
    projection: eqx.nn.Linear
    grid: int
    width: int

    def __init__(self, key, input_dim: int, output_dim: int, modes: int = 16, width: int = 64, grid: int = 64):
        # Enforce the Nyquist limit: modes cannot exceed half the grid size plus one
        if modes > grid // 2 + 1:
            raise ValueError(f"modes ({modes}) must be <= grid // 2 + 1 ({grid // 2 + 1})")

        k1, k2, k3 = jax.random.split(key, 3)
        self.grid = grid
        self.width = width

        # Lift the input into a full 2D space: (channels * grid_points)
        self.lifting = eqx.nn.Linear(input_dim, width * grid, key=k1)

        # The Spectral Convolution now correctly processes `width` channels over a spatial `grid`
        self.spectral_conv = SpectralConv1d(k2, width, width, modes)

        # Project the flattened 2D space back into the output dimension
        self.projection = eqx.nn.Linear(width * grid, output_dim, key=k3)

    def _internal_predict(self, x):
        # 1. Lift and unflatten into (channels, grid_points)
        lifted = self.lifting(x).reshape(self.width, self.grid)

        # 2. Convolve in the frequency domain, returning (channels, grid_points)
        conv_out = jax.nn.gelu(self.spectral_conv(lifted))

        # 3. Flatten back to 1D and project down to the exact output_dim
        return self.projection(conv_out.flatten())

    def predict(self, x, compute_derivatives=False):
        means = self._internal_predict(x)
        preds = SurrogateEvaluation(means=means)
        if compute_derivatives:
            preds = eqx.tree_at(
                lambda p: p.derivatives, preds, jax.jacrev(self._internal_predict)(x), is_leaf=lambda n: n is None
            )
        return preds

    def fit(
        self,
        data: Union[tuple[jax.Array, jax.Array], "_loader"],
        learning_rate: float = 1e-3,
        epochs: int = 500,
        **kwargs,
    ) -> "FourierNeuralOperator":
        r"""Fit spectral predictions by minimizing mean squared error.

        The current parameters provide the starting point for Adam training.
        Training returns a fitted model without modifying this instance.

        ### Mathematical Formulation
        $L = mean((Y-f_\theta(X))^2)$.

        ### Process Architecture
        Apply Adam updates once per loader batch per epoch, or once per epoch for array pairs.

        ### Usage Example
        ```python
        import jax
        import jax.numpy as jnp
        from flowtangent.models.surrogates import (
            FourierNeuralOperator,
        )

        model = FourierNeuralOperator(
            jax.random.PRNGKey(0), 1, 1, modes=2, width=2, grid=4
        )
        fitted = model.fit(
            (jnp.zeros((2, 1)), jnp.ones((2, 1))), epochs=1
        )
        ```

        Args:
            data (tuple[jax.Array, jax.Array] | DataLoader | LatentDataLoader):
                2D feature and target arrays `(N, D)` and `(N, O)`, or a loader yielding
                such batches.
            learning_rate (float): Adam step size. Defaults to `1e-3`.
            epochs (int): Number of passes over the data. Defaults to `500`;
                a nonpositive count performs no updates.
            **kwargs (Any): Ignored.

        Returns:
            FourierNeuralOperator: Model after training; returns the original instance if
                no updates run. Loader and training errors propagate.

        Raises:
            TypeError: If target and prediction shapes cannot broadcast in the loss.
            jaxtyping.TypeCheckError: If arguments violate the method annotations.
        """

        def loss_fn(model, batch):
            x_batch, y_batch = batch
            preds = jax.vmap(model._internal_predict)(x_batch)
            return jnp.mean((y_batch - preds) ** 2)

        return fit_neural_model(self, data, loss_fn, learning_rate, epochs)


__all__ = [
    "Surrogate",
    "SurrogateEvaluation",
    "PolynomialChaosSurrogate",
    "InverseDistanceWeighting",
    "RadialBasisFunction",
    "GaussianProcessSurrogate",
    "MLPSurrogate",
    "TransformerSurrogate",
    "DeepEnsembleSurrogate",
    "MixtureOfExperts",
    "SpectralConv1d",
    "FourierNeuralOperator",
]
