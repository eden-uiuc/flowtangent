import jax.numpy as jnp
import string
import sys


from pathlib import Path

from functools import lru_cache
from flowtangent.utils.io import _ft_root

from flowtangent.components import Airfoil

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
def _load_map_from_disk(name: str):
    """Hidden helper that does the disk I/O, safely cached."""
    if name not in _AF_REGISTRY:
        raise AttributeError(f"Airfoil '{name}' not found in library.")
        
    return Airfoil.from_file(_AF_REGISTRY[name])

def __getattr__(name: str):
    """Intercepts module-level attribute access."""
    if name.startswith("_"):
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    return _load_map_from_disk(name)

def __dir__():
    """Allows IDEs and the `dir()` command to see the available airfoils."""
    return list(_AF_REGISTRY.keys())

def generate_stub():
    """Generates the .pyi stub file for IDE autocomplete."""
    lines = [
        "from typing import Any",
        "from ._classes import Airfoil",
        "",
    ]

    for attr_name in sorted(_AF_REGISTRY.keys()):
        lines.append(f"{attr_name}: Airfoil")

    STUB_FILE.write_text("\n".join(lines))
    print(f"Generated {STUB_FILE.name} with {len(_AF_REGISTRY)} airfoils.")


def validate_library():
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
        try:
            af = getattr(sys.modules[__name__], name)
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
    threshold = 0.9
    outlier_indices = jnp.where(distances >= threshold)[0]
    safe_indices = jnp.where(distances < threshold)[0]
    
    # Sort outliers so you see the worst ones first
    sorted_outliers = sorted(outlier_indices, key=lambda idx: distances[idx], reverse=True)
    
    for idx in sorted_outliers:
        inspection_queue.append((names[idx], f"Mahalanobis Dist: {distances[idx]:.2f} (>= 0.95)"))

    # ---------------------------------------------------------
    # PASS 3: Histogram Plot
    # ---------------------------------------------------------
    print(f"\nFound {len(outlier_indices)} statistical outliers (Dist >= 0.95).")
    print("Displaying distance histogram. Close the plot to begin manual inspection...")

    import matplotlib.pyplot as plt
    
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
            # af = getattr(sys.modules[__name__], name)
            # af.plot() 
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
    
    # Reconstruct using only the top 16 "smooth" components
    k = 16
    X_smooth = jnp.dot(X_centered, jnp.dot(Vt[:k].T, Vt[:k])) + mu

    # 3. Interactive Plotting Loop
    for idx in sorted_outliers:
        name = names[idx]
        dist = distances[idx]
        
        af = getattr(sys.modules[__name__], name)
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
        ax1.plot(x, y_up_raw, 'b-', label='Upper')
        ax1.plot(x, y_lo_raw, 'r-', label='Lower')
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


if __name__ == "__main__":
    generate_stub()
    validate_library()