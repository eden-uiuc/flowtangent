import string
import sys
from functools import lru_cache
from pathlib import Path

import jax.numpy as jnp

from flowtangent.utils.io import _ft_root

from ._classes import Airfoil

# ----------------------------------------------------------------------------------------------------------------------
#  Airfoil Directory
# ----------------------------------------------------------------------------------------------------------------------

_AF_DIR = _ft_root() / "data/airfoils"
_AF_REGISTRY = {}
STUB_FILE = Path(__file__).resolve().parent / "_data.pyi"


def _sanitize_for_python(stem: str) -> str:
    """Converts a messy filename stem into a valid Python identifier."""
    # Replace both hyphens and periods with underscores
    safe_name = stem.replace('-', '_').replace('.', '_')

    # If it starts with a number, prepend 'af_'
    if safe_name and safe_name[0] in string.digits:
        safe_name = f"af_{safe_name}"

    return safe_name

# 1. Build the registry ONCE at module initialization
if _AF_DIR.exists():
    for file_path in _AF_DIR.glob("*.dat"):
        attr_name = _sanitize_for_python(file_path.stem)
        # Store the exact Path object, completely eliminating the need to reverse the string
        _AF_REGISTRY[attr_name] = file_path

    # Optional: Catch .txt files as well
    for file_path in _AF_DIR.glob("*.txt"):
        attr_name = _sanitize_for_python(file_path.stem)
        if attr_name not in _AF_REGISTRY:
            _AF_REGISTRY[attr_name] = file_path

# 2. Simplified Loading Logic
@lru_cache(maxsize=None)
def load_foil(name: str, interpolate: bool = True, n_pts: int = 128):
    """Hidden helper that does the disk I/O, safely cached."""
    if name not in _AF_REGISTRY:
        raise AttributeError(f"Airfoil '{name}' not found in library.")

    return Airfoil.from_file(_AF_REGISTRY[name], interpolate=interpolate, n_pts=n_pts)

def __getattr__(name: str):
    """Intercepts module-level attribute access."""
    if name.startswith("_"):
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    return load_foil(name)

def __dir__():
    """Allows IDEs and the `dir()` command to see the available airfoils."""
    return list(_AF_REGISTRY.keys()) + ['load_foil']

def generate_stub():
    """Generates the .pyi stub file for IDE autocomplete."""
    lines = [
        "from typing import Any",
        "from ._classes import Airfoil",
        "",
        "def load_foil(name: str, interpolate: bool = True, n_pts: int = 128) -> Airfoil: ...",
        "",
    ]

    for attr_name in sorted(_AF_REGISTRY.keys()):
        lines.append(f"{attr_name}: Airfoil")

    STUB_FILE.write_text("\n".join(lines))
    print(f"Generated {STUB_FILE.name} with {len(_AF_REGISTRY)} airfoils.")


def validate_library(plotting: bool = False, k=32):
    """
    Validates the airfoil dataset via hard geometric constraints and latent eigenspace norms.
    Displays a histogram of distances, then launches an interactive matplotlib session
    to manually inspect airfoils with a Mahalanobis distance >= 0.95.
    """
    print(f"Starting validation of {len(_AF_REGISTRY)} airfoils...")

    valid_airfoils = {}
    inspection_queue = [] # Stores tuples of (name, reason)

    from tqdm import tqdm

    # ---------------------------------------------------------
    # PASS 1: Hard Geometric & Topological Constraints
    # ---------------------------------------------------------
    for name in tqdm(_AF_REGISTRY.keys(), desc="Inspecting Airfoils"):


        if "30p" in name or "ua79sff" in name or "r1145" in name:
            continue

        try:
            af = load_foil(name)
        except Exception as e:
            inspection_queue.append((name, f"Load crash: {str(e)}"))
            continue

        thickness = af.y_upper - af.y_lower

        if jnp.any(thickness < -1e-5):
            inspection_queue.append((name, "Self-intersecting surfaces"))
        elif af.max_thickness < 0.01:
            inspection_queue.append((name, f"Collapsed thickness ({af.max_thickness:.4f})"))
        elif thickness[-1] > 0.05:
            inspection_queue.append((name, f"Large TE gap ({thickness[-1]:.4f})"))
        else:
            feature_vector = jnp.concatenate([af.y_upper, af.y_lower])
            valid_airfoils[name] = feature_vector

    print(f"Found {len(inspection_queue)} invalid airfoils.")

    from matplotlib import pyplot as plt

    for name, reason in inspection_queue:

        af = load_foil(name)
        file_af = load_foil(name, interpolate=False)
        if plotting:
            plt.plot(af.x_upper, af.y_upper, color='#FC6255', label="interpolated")
            plt.plot(af.x_lower, af.y_lower, color='#FC6255', label="interpolated")
            plt.plot(file_af.x_upper, file_af.y_upper, color="#0B078A", label="raw")
            plt.plot(file_af.x_lower, file_af.y_lower, color="#0B078A", label="raw")
            plt.title(f"{name} Raw vs Interpolated: {reason}")
            plt.axis('equal')
            plt.show()


    # ---------------------------------------------------------
    # PASS 2: Eigenspace Norms (Absolute Threshold >= 0.95)
    # ---------------------------------------------------------
    names = list(valid_airfoils.keys())
    X = jnp.stack(list(valid_airfoils.values()))

    mu = jnp.mean(X, axis=0)
    X_centered = X - mu
    U, S, Vt = jnp.linalg.svd(X_centered, full_matrices=False)

    # Mahalanobis Distance: Latent projection normalized by singular values
    Z = U * S
    distances = jnp.linalg.norm(Z / (S + 1e-8), axis=1)

    # Use absolute raw distance threshold
    threshold = 0.95
    outlier_indices = jnp.where(distances >= threshold)[0]
    safe_indices = jnp.where(distances < threshold)[0]

    # Sort outliers so you see the worst ones first
    sorted_outliers = sorted(outlier_indices, key=lambda idx: distances[idx], reverse=True)

    for idx in sorted_outliers:
        inspection_queue.append((names[idx], f"Mahalanobis Dist: {distances[idx]:.2f} (>= 0.95)"))

    # ---------------------------------------------------------
    # PASS 3: Histogram Plot
    # ---------------------------------------------------------

    import matplotlib.pyplot as plt
    if plotting:

        print(f"\nFound {len(outlier_indices)} statistical outliers (Dist >= 0.95).")
        print("Displaying distance histogram. Close the plot to begin manual inspection...")

        plt.figure(figsize=(10, 6))
        plt.hist(distances[safe_indices], bins=50, alpha=0.7, color='blue', label='Normal Airfoils (< 0.95)')

        if len(outlier_indices) > 0:
            plt.hist(distances[outlier_indices], bins=10, alpha=0.9, color='red', label='Outliers (>= 0.95)')

        plt.axvline(x=threshold, color='black', linestyle='--', linewidth=1.5, label='Threshold (0.95)')
        plt.title("Airfoil Eigenspace Norms (Mahalanobis Distance)")
        plt.xlabel("Distance from Dataset Mean")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show() # Pauses here until user closes histogram

        # ---------------------------------------------------------
        # PASS 4: Interactive Visual Inspection
        # ---------------------------------------------------------
        if not inspection_queue:
            print("\nNo airfoils flagged for inspection. Validation complete.")
            return

        print(f"\nStarting manual review of {len(inspection_queue)} flagged airfoils.")
        print("Close the plot window to advance to the next airfoil.")

        for i, (name, reason) in enumerate(inspection_queue, 1):
            # print(f"Inspecting {i}/{len(inspection_queue)}: {name}...")

            try:
                print(f"{name}\nFlagged for: {reason}")
            except Exception as e:
                print(f"  -> Could not plot {name}: {str(e)}")

        import numpy as np

        def get_curvature(x, y):

            """Calculates geometric curvature using standard NumPy gradients."""
            x, y = np.array(x), np.array(y)

            # Calculate differentials (adding epsilon to prevent div-by-zero at the LE)
            dx = np.gradient(x) + 1e-12
            dy = np.gradient(y)

            # First derivative (y') and second derivative (y'')
            yp = dy / dx
            ypp = np.gradient(yp) / dx

            # True curvature magnitude
            kappa = np.abs(ypp) / (1.0 + yp**2)**1.5
            return kappa

        print(f"Launching diagnostic suite for {len(sorted_outliers)} outliers...")

        # Reconstruct using only the top-k "smooth" components
        X_smooth = jnp.dot(X_centered, jnp.dot(Vt[:k].T, Vt[:k])) + mu

        # 3. Interactive Plotting Loop
        for idx in sorted_outliers:
            name = names[idx]
            dist = distances[idx]

            af = getattr(sys.modules[__name__], name)
            file_af = load_foil(name, interpolate=False)
            x = np.array(af.x_upper)

            y_up_raw = np.array(af.y_upper)
            y_lo_raw = np.array(af.y_lower)

            # Extract the smoothed reconstruction
            y_up_smooth = np.array(X_smooth[idx, :128])
            y_lo_smooth = np.array(X_smooth[idx, 128:])

            # Calculate curvatures
            kappa_up = get_curvature(x, y_up_raw)
            kappa_lo = get_curvature(x, y_lo_raw)

            # Create 3-panel diagnostic plot
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
            fig.suptitle(f"Diagnostic: {name} (M-Dist: {dist:.2f})", fontsize=14, fontweight='bold')

            # Panel 1: Raw Geometry (Look for blunt trailing edges)
            ax1.plot(x, y_up_raw, 'b-', label='Interp. Upper')
            ax1.plot(x, y_lo_raw, 'r-', label='Interp. Lower')
            ax1.scatter(file_af.x_upper, file_af.y_upper, c='c', label='File Upper')
            ax1.scatter(file_af.x_lower, file_af.y_lower, c='g', label='File Lower')
            ax1.set_aspect('equal')
            ax1.set_title("1. Raw Interpolated Geometry")
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # Panel 2: Curvature (Look for spikes away from X=0)
            ax2.plot(x, kappa_up, 'b-', label='Upper Curvature')
            ax2.plot(x, kappa_lo, 'r-', label='Lower Curvature')
            ax2.set_yscale('log')
            ax2.set_title("2. Surface Curvature (Log Scale)")
            ax2.set_ylabel(r"$\kappa$")
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            # Panel 3: PCA Residual (The exact noise isolating this airfoil)
            ax3.plot(x, y_up_raw - y_up_smooth, 'b-', label='Upper Residual')
            ax3.plot(x, y_lo_raw - y_lo_smooth, 'r-', label='Lower Residual')
            ax3.set_title(f"3. PCA Reconstruction Residual (Top {k} Components)")
            ax3.set_xlabel("X Coordinate")
            ax3.set_ylabel(r"$\Delta Y$")
            ax3.axhline(0, color='black', linewidth=1)
            ax3.grid(True, alpha=0.3)
            ax3.legend()

            plt.tight_layout()
            plt.show()

    print("\nInspection complete.")
    return valid_airfoils

def evaluate_naca_overlap(valid_uiuc_dict, n_samples=5000, n_pts=128, k=32):
    """
    Generates a massive dense sampling of the NACA parameter space using jax.vmap,
    then tests if the UIUC principal components can accurately reconstruct it.
    """

    import jax

    print(f"Generating {n_samples * 2} NACA airfoils on the GPU...")
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)

    # ---------------------------------------------------------
    # 1. VMAP Setup (Vectorizing the math functions)
    # ---------------------------------------------------------
    # in_axes specifies which arguments are arrays (0) and which are static/fixed (None)
    vmap_naca4 = jax.vmap(Airfoil._naca_4_math, in_axes=(0, 0, 0, None))
    vmap_naca5 = jax.vmap(Airfoil._naca_5_math, in_axes=(0, 0, 0, 0, None))

    # ---------------------------------------------------------
    # 2. Sample the NACA 4-Series Space
    # ---------------------------------------------------------
    m_4 = jax.random.uniform(k1, (n_samples,), minval=0.0, maxval=0.09)
    p_4 = jax.random.uniform(k2, (n_samples,), minval=0.1, maxval=0.9)
    t_4 = jax.random.uniform(k3, (n_samples,), minval=0.05, maxval=0.25)

    # Execute 5,000 airfoils in one GPU call
    _, y_up_4, y_lo_4 = vmap_naca4(m_4, p_4, t_4, n_pts)
    features_4 = jnp.concatenate([y_up_4, y_lo_4], axis=1)

    # ---------------------------------------------------------
    # 3. Sample the NACA 5-Series Space
    # ---------------------------------------------------------
    # design_cl (0.15 to 0.9), p_idx (1 to 5), q_val (0 or 1), thickness (0.05 to 0.25)
    cl_5 = jax.random.uniform(k4, (n_samples,), minval=0.15, maxval=0.9)
    p_idx_5 = jax.random.randint(k5, (n_samples,), minval=1, maxval=6)
    q_val_5 = jax.random.randint(k6, (n_samples,), minval=0, maxval=2)
    t_5 = jax.random.uniform(k7, (n_samples,), minval=0.05, maxval=0.25)

    # Execute 5,000 airfoils in one GPU call
    _, y_up_5, y_lo_5 = vmap_naca5(cl_5, p_idx_5, q_val_5, t_5, n_pts)
    features_5 = jnp.concatenate([y_up_5, y_lo_5], axis=1)

    X_naca = jnp.vstack([features_4, features_5])

    # ---------------------------------------------------------
    # 4. Extract the UIUC Basis
    # ---------------------------------------------------------
    print("Computing UIUC SVD basis...")
    X_uiuc = jnp.stack(list(valid_uiuc_dict.values()))
    mu_uiuc = jnp.mean(X_uiuc, axis=0)
    X_uiuc_centered = X_uiuc - mu_uiuc

    U, S, Vt_uiuc = jnp.linalg.svd(X_uiuc_centered, full_matrices=False)

    # ---------------------------------------------------------
    # 5. The Subspace Overlap Test
    # ---------------------------------------------------------
    print("Testing NACA geometries against UIUC basis...")
    basis = Vt_uiuc[:k]

    # Project NACA airfoils into the UIUC space, then reconstruct them
    X_naca_centered = X_naca - mu_uiuc
    naca_reconstructed = jnp.dot(X_naca_centered, jnp.dot(basis.T, basis)) + mu_uiuc

    # Calculate MSE
    mse = jnp.mean((X_naca - naca_reconstructed)**2, axis=1)

    print(f"Mean Reconstruction Error: {jnp.mean(mse):.2e}")
    print(f"Max Reconstruction Error:  {jnp.max(mse):.2e}")

    if jnp.max(mse) < 1e-5:
        print("\nVerdict: UIUC already completely spans the NACA design space.")
        print("No Gram-Schmidt augmentation is required.")
        return X_uiuc
    else:
        print("\nVerdict: NACA manifold contains novel geometry.")
        print("Building joint Gram-Schmidt/SVD basis...")
        X_joint = jnp.vstack([X_uiuc, X_naca])
        return X_joint

def calculate_latent_voids(valid_airfoils_dict, k=32):
    """
    Calculates the maximum empty void (dispersion) in the latent geometry manifold 
    using the Mahalanobis Nearest-Neighbor distance.
    """
    # 1. Project into Latent Space
    X = jnp.stack(list(valid_airfoils_dict.values()))
    mu = jnp.mean(X, axis=0)
    X_centered = X - mu

    U, S, Vt = jnp.linalg.svd(X_centered, full_matrices=False)
    Z = U[:, :k] * S[:k] # Latent scores (1650, k)

    # 2. Normalize by Singular Values to get Mahalanobis Space
    # We add a tiny epsilon to prevent division by zero on low-variance components
    Z_mah = Z / (S[:k] + 1e-8)

    # 3. Compute pairwise distance matrix (O(N^2) is fast for N=1650 in JAX)
    # Using the expanding norm trick: (a-b)^2 = a^2 + b^2 - 2ab
    Z_sq = jnp.sum(Z_mah**2, axis=1)
    dist_sq = Z_sq.reshape(-1, 1) + Z_sq.reshape(1, -1) - 2 * jnp.dot(Z_mah, Z_mah.T)

    # Clip negative zeros from floating point errors, then sqrt
    dist_matrix = jnp.sqrt(jnp.clip(dist_sq, min=0.0))

    # 4. Find Nearest Neighbors (ignoring self-distance of 0 on the diagonal)
    # Fill diagonal with infinity so an airfoil doesn't pick itself
    mask = jnp.eye(dist_matrix.shape[0], dtype=bool)
    dist_matrix_no_self = jnp.where(mask, jnp.inf, dist_matrix)

    # Find the distance to the closest neighbor for every airfoil
    nearest_neighbor_dists = jnp.min(dist_matrix_no_self, axis=1)

    # 5. The Maximum Void is the maximum of these nearest-neighbor distances
    max_void = jnp.max(nearest_neighbor_dists)
    mean_void = jnp.mean(nearest_neighbor_dists)

    print(f"\nMean distance to nearest airfoil: {mean_void:.4f}")
    print(f"Maximum geometric void size:      {max_void:.4f}")

    # Optional: Identify the most isolated airfoil
    isolated_idx = jnp.argmax(nearest_neighbor_dists)
    names = list(valid_airfoils_dict.keys())
    print(f"Most isolated geometry: {names[isolated_idx]}")

    return nearest_neighbor_dists, max_void

def analyze_pruning_tradeoffs(valid_airfoils_dict, max_prunes=250, k=16, n_cond_baseline=128):
    """
    Iteratively prunes the latent space and plots the statistical tradeoff 
    between geometric density and required Sobol condition sampling.
    """
    pruned_dict = dict(valid_airfoils_dict)

    history = {
        'n_airfoils': [],
        'max_void_mah': [],
        'geom_sigma': [],
        'cond_sigma_at_128': [],
        'n_cond_for_baseline': []
    }

    # The mathematical standard to maintain
    baseline_total_points = len(valid_airfoils_dict) * n_cond_baseline

    print(f"Tracking pruning tradeoffs for {max_prunes} steps...")

    import numpy as np
    from tqdm import trange

    for step in trange(max_prunes + 1, desc="Pruning Airfoils"):
        names = list(pruned_dict.keys())
        X = jnp.stack(list(pruned_dict.values()))
        n_airfoils = len(names)

        # 1. Re-fit PCA and calculate Mahalanobis Nearest-Neighbors
        mu = jnp.mean(X, axis=0)
        X_centered = X - mu
        U, S, Vt = jnp.linalg.svd(X_centered, full_matrices=False)

        Z = U[:, :k] * S[:k]
        Z_mah = Z / (S[:k] + 1e-8)

        Z_sq = jnp.sum(Z_mah**2, axis=1)
        dist_sq = Z_sq.reshape(-1, 1) + Z_sq.reshape(1, -1) - 2 * jnp.dot(Z_mah, Z_mah.T)
        dist_matrix = jnp.sqrt(jnp.clip(dist_sq, min=0.0))

        mask = jnp.eye(dist_matrix.shape[0], dtype=bool)
        dist_matrix_no_self = jnp.where(mask, jnp.inf, dist_matrix)
        nearest_neighbor_dists = jnp.min(dist_matrix_no_self, axis=1)

        max_void = float(jnp.max(nearest_neighbor_dists))
        isolated_idx = int(jnp.argmax(nearest_neighbor_dists))
        worst_airfoil = names[isolated_idx]

        # 2. Record Metrics
        history['n_airfoils'].append(n_airfoils)
        history['max_void_mah'].append(max_void)

        # Convert to Universal Sigma units for ARD Kernel comparison
        history['geom_sigma'].append(max_void * np.sqrt(n_airfoils))

        # Calculate Condition Void Sigma (Max void in 4D Sobol * sqrt(12))
        current_total_points = n_airfoils * n_cond_baseline
        cond_void_raw = (1.0 / current_total_points)**0.25
        history['cond_sigma_at_128'].append(cond_void_raw * np.sqrt(12))

        # Calculate required conditions to maintain exact baseline density
        history['n_cond_for_baseline'].append(baseline_total_points / n_airfoils)

        # 3. Prune the worst offender for the next loop
        if step < max_prunes:
            pruned_dict.pop(worst_airfoil)

    # --- Plotting the Tradeoffs ---
    import matplotlib.pyplot as plt
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    x_axis = history['n_airfoils']

    # Panel 1: The Raw Geometric Void
    ax1.plot(x_axis, history['max_void_mah'], 'b-', linewidth=2)
    ax1.set_title("Max Geometric Void (Mahalanobis)")
    ax1.set_ylabel("Raw Distance")
    ax1.invert_xaxis() # Read left-to-right as airfoils are removed
    ax1.grid(True, alpha=0.3)

    # Panel 2: The Isotropic Convergence (Sigma Units)
    ax2.plot(x_axis, history['geom_sigma'], 'b-', label=r"Geometry Void ($\sigma$)")
    ax2.plot(x_axis, history['cond_sigma_at_128'], 'r--', label=r"Condition Void at N=128 ($\sigma$)")
    ax2.set_title("Space Convergence (ARD Kernel Confidence)")
    ax2.set_ylabel(r"Standard Deviations ($\sigma$)")
    ax2.set_yscale('log')
    ax2.invert_xaxis()
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: Required Compensation
    ax3.plot(x_axis, history['n_cond_for_baseline'], 'g-', linewidth=2)
    ax3.set_title("Conditions per Airfoil Needed to Preserve Baseline Density")
    ax3.set_xlabel("Number of Airfoils Remaining")
    ax3.set_ylabel(r"Required $N_{cond}$")
    ax3.invert_xaxis()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return pruned_dict, worst_airfoil, history

def regularize_design_space(valid_airfoils_dict, max_prunes=250, k=16, plotting=False):
    """
    Visualizes the 1400/250 split to ensure the retained core dataset 
    still covers the primary aerodynamic design space.
    """
    # 1. Run the pruner exactly as before to get the two sets
    # (Assuming you modify prune_latent_space to return both dictionaries)
    kept_dict, next_prune, history = analyze_pruning_tradeoffs(valid_airfoils_dict, max_prunes=max_prunes, k=k)
    pruned_dict = {k:v for k,v in valid_airfoils_dict.items() if k not in kept_dict}

    # 2. Project BOTH sets into the PCA space defined ONLY by the Kept airfoils
    X_keep = jnp.stack(list(kept_dict.values()))
    X_prune = jnp.stack(list(pruned_dict.values()))

    mu_keep = jnp.mean(X_keep, axis=0)
    X_centered = X_keep - mu_keep
    U, S, Vt = jnp.linalg.svd(X_centered, full_matrices=False)

    basis = Vt[:k]

    # Get coordinates in the new 16D space
    Z_keep = jnp.dot(X_centered, basis.T)
    Z_prune = jnp.dot(X_prune - mu_keep, basis.T)

    X_prune_reconstructed = jnp.dot(Z_prune, basis) + mu_keep
    mse_prune = jnp.mean((X_prune - X_prune_reconstructed)**2, axis=1)

    import numpy as np

    mse_prune_np = np.array(mse_prune)

    mean_mse = np.mean(mse_prune_np)
    median_mse = np.median(mse_prune_np)
    p95_mse = np.percentile(mse_prune_np, 95)
    max_mse = np.max(mse_prune_np)

    # 3. Plot the top 2 Principal Components to check for Family Extinction
    import matplotlib.pyplot as plt
    if plotting:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Panel 1: Latent Space Scatter
        ax1.scatter(Z_keep[:, 0], Z_keep[:, 1], c='blue', alpha=0.5, label=f'Kept Core ({len(kept_dict)})', s=15)
        ax1.scatter(Z_prune[:, 0], Z_prune[:, 1], c='red', alpha=0.8, marker='x', label=f'Pruned OOD ({len(pruned_dict)})', s=30)
        ax1.set_title("Latent Space Distribution (PC1 vs PC2)")
        ax1.set_xlabel("Principal Component 1")
        ax1.set_ylabel("Principal Component 2")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Panel 2: Reconstruction Error Histogram
        ax2.hist(mse_prune_np, bins=40, color='red', alpha=0.7, edgecolor='black')
        ax2.axvline(median_mse, color='blue', linestyle='dashed', linewidth=2, label=f'Median: {median_mse:.2e}')
        ax2.axvline(p95_mse, color='black', linestyle='dashed', linewidth=2, label=f'95th Pctl: {p95_mse:.2e}')
        ax2.set_title("OOD Reconstruction Error (MSE) via 16D Core Basis")
        ax2.set_xlabel("Mean Squared Error")
        ax2.set_ylabel("Frequency")
        # Log scale is often necessary for MSE histograms to see the extreme outliers
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    # 5. Summary Statistics Output
    print("\n--- OOD Reconstruction Error Statistics ---")
    print(f"Median MSE: {median_mse:.2e} (Typical OOD Error)")
    print(f"Mean MSE:   {mean_mse:.2e}")
    print(f"95th Pctl:  {p95_mse:.2e} (Extreme Geometry Error)")
    print(f"Max MSE:    {max_mse:.2e}")

    # 4. Print a random sample of what got thrown away
    import numpy as np
    print("\nSample of Pruned Geometries (Check for extinct families):")
    pruned_names = list(pruned_dict.keys())
    pruned_names.sort()
    for name in pruned_names:
        print(f" - {name}")

    return kept_dict, pruned_dict

if __name__ == "__main__":
    generate_stub()
    latent_dim = 16
    valid_airfoils = validate_library(k=latent_dim)
    # NND, max_void = calculate_latent_voids(valid_airfoils, k=latent_dim)
    kept_dict, pruned_dict = regularize_design_space(valid_airfoils)
    X_airfoils = evaluate_naca_overlap(kept_dict, k=latent_dim)

    # n6412i = load_foil("goe802a")
    # n6412r = load_foil("goe802a", interpolate=False)

    # import matplotlib.pyplot as plt

    # plt.plot(n6412i.x_upper, n6412i.y_upper, color='#FC6255')
    # plt.plot(n6412i.x_lower, n6412i.y_lower, color='#FC6255')
    # plt.scatter(n6412r.x_upper, n6412r.y_upper, color="#0B078A")
    # plt.scatter(n6412r.x_lower, n6412r.y_lower, color="#0B078A")
    # plt.title("Raw vs Interpolated")
    # plt.axis('equal')
    # plt.show()
