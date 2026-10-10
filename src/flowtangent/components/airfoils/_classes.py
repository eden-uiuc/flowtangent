# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from functools import total_ordering
from pathlib import Path
from typing import Optional

# package imports
import jax
import jax.numpy as jnp
import numpy as np  # For loading from disk
import plotly.graph_objects as go

# FlowTangent imports
from flowtangent.utils import empty_array, field

from ...core._component import Component
from ...plots import plot_airfoil, styles

# ----------------------------------------------------------------------------------------------------------------------
#  Airfoil
# ----------------------------------------------------------------------------------------------------------------------


@total_ordering
class Airfoil(Component):
    name: str = field("Airfoil", static=True)
    source_file: Optional[str | Path] = None

    thickness_to_chord: float = 0.0
    max_thickness: float = 0.0
    wedge_angle: float = 0.0

    coordinates: jax.Array = empty_array((0, 2))
    camber: jax.Array = empty_array()

    x_coordinates: jax.Array = empty_array()
    y_coordinates: jax.Array = empty_array()

    x_upper: jax.Array = empty_array()
    x_lower: jax.Array = empty_array()

    y_upper: jax.Array = empty_array()
    y_lower: jax.Array = empty_array()

    @staticmethod
    @jax.jit(static_argnames=["n_points"])
    def _naca_4_math(m: float, p: float, t: float, n_points: int = 128):
        theta = jnp.linspace(0, jnp.pi, n_points)
        x = 0.5 * (1 - jnp.cos(theta))

        yt = 5 * t * (0.2969 * jnp.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)

        # Safe denominators to prevent NaN generation in the JIT compiler
        p_safe = jnp.where(p == 0, 1e-7, p)

        yc_fwd = (m / p_safe**2) * (2 * p_safe * x - x**2)
        yc_aft = (m / (1 - p_safe) ** 2) * ((1 - 2 * p_safe) + 2 * p_safe * x - x**2)

        # Multiplex the camber line
        yc = jnp.where(p == 0, 0.0, jnp.where(x <= p, yc_fwd, yc_aft))

        return x, yc + yt, yc - yt

    @staticmethod
    @jax.jit(static_argnames=["n_points"])
    def _naca_5_math(design_cl: float, p_idx: int, q_val: int, t: float, n_points: int = 128):
        """
        design_cl: First digit * 0.15 (e.g., '2' -> 0.3)
        p_idx: Second digit (1 through 5, e.g., '3' -> 3)
        q_val: Third digit (0 for normal, 1 for reflexed)
        t: Last two digits / 100 (e.g., '12' -> 0.12)
        """
        theta = jnp.linspace(0, jnp.pi, n_points)
        x = 0.5 * (1 - jnp.cos(theta))

        # ----------------------------------------------------
        # 1. Constants Mapping (Indices 0 to 5)
        # ----------------------------------------------------
        # Normal (Q=0) Constants
        m_q0 = jnp.array([0.0, 0.0580, 0.1260, 0.2025, 0.2900, 0.3910])
        k1_q0 = jnp.array([0.0, 361.4, 51.64, 15.957, 6.643, 3.230])

        # Reflexed (Q=1) Constants
        # (Note: P=1 is theoretically undefined for reflexed, filled with 0.0)
        m_q1 = jnp.array([0.0, 0.0, 0.1300, 0.2130, 0.2980, 0.3910])
        k1_q1 = jnp.array([0.0, 0.0, 51.99, 15.793, 6.520, 3.191])
        k2k1_q1 = jnp.array([0.0, 0.0, 0.000764, 0.00677, 0.0303, 0.1355])

        # Select constants based on the Q digit
        m = jnp.where(q_val == 1, m_q1[p_idx], m_q0[p_idx])
        k1 = jnp.where(q_val == 1, k1_q1[p_idx], k1_q0[p_idx])
        k2k1 = jnp.where(q_val == 1, k2k1_q1[p_idx], 0.0)

        # Safe denominator protection for the JIT compiler
        m_safe = jnp.where(m == 0, 1e-7, m)

        # ----------------------------------------------------
        # 2. Camber Line (Q=0)
        # ----------------------------------------------------
        yc_q0_fwd = (k1 / 6.0) * (x**3 - 3 * m_safe * x**2 + (m_safe**2) * (3 - m_safe) * x)
        yc_q0_aft = (k1 * (m_safe**3) / 6.0) * (1 - x)
        yc_0 = jnp.where(x <= m, yc_q0_fwd, yc_q0_aft)

        # ----------------------------------------------------
        # 3. Reflexed Camber Line (Q=1)
        # ----------------------------------------------------
        # We multiplex the cubic weight: 1.0 for forward, (k2/k1) for aft
        cubic_weight = jnp.where(x <= m, 1.0, k2k1)

        yc_1 = (k1 / 6.0) * (
            cubic_weight * (x - m_safe) ** 3 - k2k1 * (1 - m_safe) ** 3 * x - (m_safe**3) * x + m_safe**3
        )

        # ----------------------------------------------------
        # 4. Final Assembly
        # ----------------------------------------------------
        # Select the correct camber math, apply the scale, and zero out if m=0
        camber_scale = design_cl / 0.3
        yc_raw = jnp.where(q_val == 1, yc_1, yc_0)
        yc = jnp.where(m == 0, 0.0, yc_raw) * camber_scale

        # Standard thickness distribution
        yt = 5 * t * (0.2969 * jnp.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)

        return x, yc + yt, yc - yt

    @staticmethod
    @jax.jit(static_argnames=["n_points"])
    def _interpolate_surface(points: jax.Array, n_points: int = 128):
        """Assumes Selig Format for sorting."""
        N = points.shape[0]
        idx = jnp.arange(N)

        # 1. Find Leading Edge
        LE_idx = jnp.argmin(points[:, 0])

        # 2. Fixed-shape masking
        is_upper = idx <= LE_idx
        is_lower = idx >= LE_idx

        # Push invalid points to infinity so they sort to the end
        x_upper = jnp.where(is_upper, points[:, 0], jnp.inf)
        y_upper = jnp.where(is_upper, points[:, 1], 0.0)

        x_lower = jnp.where(is_lower, points[:, 0], jnp.inf)
        y_lower = jnp.where(is_lower, points[:, 1], 0.0)

        # 3. Sort (Inf goes to the back)
        sort_u = jnp.argsort(x_upper)
        x_upper, y_upper = x_upper[sort_u], y_upper[sort_u]

        sort_l = jnp.argsort(x_lower)
        x_lower, y_lower = x_lower[sort_l], y_lower[sort_l]

        # 4. Enforce strict monotonicity for jnp.interp without dynamic jnp.unique
        epsilon = jnp.arange(N) * 1e-9
        x_upper = x_upper + epsilon
        x_lower = x_lower + epsilon

        # 5. Interpolate
        theta = jnp.linspace(0, jnp.pi, n_points)
        x_grid = 0.5 * (1 - jnp.cos(theta))

        y_upper_interp = jnp.interp(x_grid, x_upper, y_upper)
        y_lower_interp = jnp.interp(x_grid, x_lower, y_lower)

        return x_grid, y_upper_interp, y_lower_interp

    @staticmethod
    def _laplacian_smoothing(x, y, le_buffer=10, threshold_multiplier=3.0, max_passes=5):
        """
        Detects and smooths high-frequency facets caused by linear interpolation.
        Uses the 3rd derivative to isolate numerical artifacts from physical geometry.
        """
        y_smooth = np.copy(y)

        for _ in range(max_passes):
            # 1. Calculate physical derivatives using the actual x-coordinates
            dy1 = np.gradient(y_smooth, x)
            dy2 = np.gradient(dy1, x)
            dy3 = np.gradient(dy2, x)

            # 2. Isolate the mid-chord and trailing edge
            mid_chord_dy3 = np.abs(dy3[le_buffer:-le_buffer])

            if len(mid_chord_dy3) == 0:
                break

            median_jerk = np.median(mid_chord_dy3)
            std_jerk = np.std(mid_chord_dy3)

            # Dynamic threshold based on true physical jerk
            threshold = median_jerk + (threshold_multiplier * std_jerk)

            # 3. Find anomalous nodes (ignoring the LE buffer)
            anomalies = []
            for i in range(le_buffer, len(y_smooth) - 1):
                if np.abs(dy3[i]) > threshold:
                    anomalies.append(i)

            if not anomalies:
                break  # Cleaned

            # 4. Apply Distance-Weighted Smoothing for non-uniform grids
            for i in anomalies:
                dx_left = x[i] - x[i - 1]
                dx_right = x[i + 1] - x[i]
                total_dx = dx_left + dx_right

                # Weight by opposite distance (closer node has higher influence)
                w_left = dx_right / total_dx
                w_right = dx_left / total_dx

                # Replaces the spike with a clean, physically linear interpolation
                # between its immediate neighbors in the cosine space.
                y_smooth[i] = (w_left * y_smooth[i - 1]) + (w_right * y_smooth[i + 1])

        return jnp.array(y_smooth)

    @classmethod
    def _from_surfaces(cls, name: str, x: jax.Array, y_up: jax.Array, y_lo: jax.Array):
        """Internal helper to assemble the class attributes to prevent code duplication."""
        camber = (y_up + y_lo) / 2.0
        thickness = y_up - y_lo
        max_t = jnp.max(thickness)

        # Reconstruct the continuous loop for standard plotting (TE -> LE -> TE)
        x_loop = jnp.concatenate((x[::-1], x[1:]))
        y_loop = jnp.concatenate((y_up[::-1], y_lo[1:]))

        return cls(
            name=name,
            camber=camber,
            max_thickness=float(max_t),
            thickness_to_chord=float(max_t / 1.0),
            coordinates=jnp.column_stack((x_loop, y_loop)),
            x_coordinates=x_loop,
            y_coordinates=y_loop,
            x_upper=x,
            x_lower=x,
            y_upper=y_up,
            y_lower=y_lo,
        )

    @classmethod
    def from_naca(cls, code: str, n_pts: int = 128):
        """
        Generates an airfoil from a NACA 4-series or 5-series code.
        """
        code = code.strip()
        name = f"NACA {code}"

        if len(code) == 4:
            m = int(code[0]) / 100.0
            p = int(code[1]) / 10.0
            t = int(code[2:]) / 100.0
            x, y_up, y_lo = cls._naca_4_math(m, p, t, n_pts)

        elif len(code) == 5:
            design_cl = int(code[0]) * 0.15
            p_idx = int(code[1])
            q_val = int(code[2])
            t = int(code[3:]) / 100.0
            x, y_up, y_lo = cls._naca_5_math(design_cl, p_idx, q_val, t, n_pts)

        else:
            raise ValueError(f"Invalid NACA code: '{code}'. Must be 4 or 5 digits.")

        return cls._from_surfaces(name, x, y_up, y_lo)

    @classmethod
    def from_file(cls, file_path: str | Path, interpolate: bool = False, n_pts: int = 128):
        """
        Parses Selig and Lednicer format airfoil .dat files.
        Converts all inputs to standard Selig topology before utilizing
        the JAX-native interpolator.
        """

        file_path = Path(file_path)

        with open(file_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        is_lednicer = False
        data_start_idx = 0
        n_up, n_lo = 0, 0

        # Detect Format (Lednicer header check)
        for i, line in enumerate(lines[:5]):
            parts = line.replace(",", " ").split()
            if len(parts) == 2:
                try:
                    val1, val2 = float(parts[0]), float(parts[1])
                    if val1 > 1.5 and val2 > 1.5:
                        is_lednicer = True
                        n_up, n_lo = int(val1), int(val2)
                        data_start_idx = i + 1
                        break
                except ValueError:
                    continue

        if not is_lednicer:
            for i, line in enumerate(lines):
                parts = line.replace(",", " ").split()
                if len(parts) >= 2:
                    try:
                        _ = float(parts[0]), float(parts[1])
                        data_start_idx = i
                        break
                    except ValueError:
                        continue

        # Extract Raw Coordinates
        raw_coords = []
        for line in lines[data_start_idx:]:
            parts = line.replace(",", " ").split()
            if len(parts) >= 2:
                try:
                    raw_coords.append([float(parts[0]), float(parts[1])])
                except ValueError:
                    continue

        # Extract Raw Coordinates
        raw_coords = np.array(raw_coords)
        x_raw, y_raw = raw_coords[:, 0], raw_coords[:, 1]

        # 1. Standardize to LE -> TE for both surfaces to apply the filter
        if is_lednicer:
            x_up, y_up = x_raw[:n_up], y_raw[:n_up]
            x_lo, y_lo = x_raw[n_up : n_up + n_lo], y_raw[n_up : n_up + n_lo]
        else:
            le_idx = np.argmin(x_raw)
            # Reverse upper so it flows LE -> TE
            x_up, y_up = x_raw[: le_idx + 1][::-1], y_raw[: le_idx + 1][::-1]
            x_lo, y_lo = x_raw[le_idx:], y_raw[le_idx:]

        # 2. The Trailing Edge De-Hooking Filter
        def remove_te_hook(x, y):
            """Detects forced closures at X=1.0 and extrapolates the natural slope."""
            if len(x) > 3 and np.isclose(x[-1], 1.0):
                # Calculate natural slope using the two points just before the TE
                dx = x[-2] - x[-3]
                dy = y[-2] - y[-3]

                # Prevent divide-by-zero if data has vertical stacked points
                if dx > 1e-5:
                    linear_y = y[-2] + (dy / dx) * (1.0 - x[-2])
                    # if np.abs(y[-1] - linear_y) > 0.001:
                    y[-1] = linear_y
            return x, y

        # Check for closed TE, if it's open, check for a "hook" on the bottom and remove it
        if not np.isclose(x_up[-1], x_lo[-1]) and not np.isclose(y_up[-1], y_lo[-1]):
            x_up, y_up = remove_te_hook(x_up, y_up)
            x_lo, y_lo = remove_te_hook(x_lo, y_lo)

        # 3. Restitch into standard Selig topology (TE -> LE -> TE) for the JAX interpolator
        # Reverse upper back to TE -> LE
        x_up_rev, y_up_rev = x_up[::-1], y_up[::-1]

        # Prevent duplicating the exact Leading Edge point during concatenation
        if np.allclose([x_up_rev[-1], y_up_rev[-1]], [x_lo[0], y_lo[0]]):
            x_lo, y_lo = x_lo[1:], y_lo[1:]

        selig_x = np.concatenate([x_up_rev, x_lo])
        selig_y = np.concatenate([y_up_rev, y_lo])

        # 4. Pass clean, un-hooked data to your JIT-compiled interpolator
        points_jax = jnp.column_stack((selig_x, selig_y))
        x_grid, y_up_interp, y_lo_interp = cls._interpolate_surface(points_jax, n_pts)

        y_up_clean = cls._laplacian_smoothing(np.array(x_grid), np.array(y_up_interp))
        y_lo_clean = cls._laplacian_smoothing(np.array(x_grid), np.array(y_lo_interp))

        # Clean up overlapping surfaces
        y_max = jnp.maximum(y_up_clean, y_lo_clean)
        y_min = jnp.minimum(y_up_clean, y_lo_clean)

        y_up_clean = y_max
        y_lo_clean = y_min

        if interpolate:
            return cls._from_surfaces(file_path.stem, x_grid, y_up_clean, y_lo_clean)
        else:
            return cls(
                name=file_path.stem,
                source_file=file_path,
                camber=jnp.asarray((y_up_clean + y_lo_clean) / 2.0),
                max_thickness=float(jnp.max(y_up_clean - y_lo_clean)),
                coordinates=jnp.column_stack((selig_x, selig_y)),
                x_coordinates=jnp.asarray(selig_x),
                y_coordinates=jnp.asarray(selig_y),
                x_upper=jnp.asarray(x_up_rev),
                x_lower=jnp.asarray(x_lo),
                y_upper=jnp.asarray(y_up_rev),
                y_lower=jnp.asarray(y_lo),
            )

    def plot(
        self,
        title: Optional[str] = None,
        show_markers: bool = False,
        show_camber: bool = False,
        theme: go.layout.Template = styles.nord_dark,
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> go.Figure:
        """
        Generates a 1:1 aspect ratio visualization of the airfoil geometry.
        """
        return plot_airfoil(
            airfoil=self,
            title=title,
            show_markers=show_markers,
            show_camber=show_camber,
            theme=theme,
            save_path=save_path,
            show=show,
        )

    def __eq__(self, other):
        if not isinstance(other, Airfoil):
            return NotImplemented
        return self.name == other.name

    def __lt__(self, other):
        if not isinstance(other, Airfoil):
            return NotImplemented
        return self.name < other.name


def NACA(code: str, n_pts: int = 128):
    return Airfoil.from_naca(code=code, n_pts=n_pts)
