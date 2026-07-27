import json

from pathlib import Path

import equinox as eqx
import jax.numpy as jnp

from eden_trace.utils import empty_array, init_field, register

# -----------------------------------------------------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------------------------------------------------


def get_fractional_coords(grid_1d, value):
    """
    Converts physical value to fraction index for 2D interpolation.
    Assumes the 1D grid is strictly increasing
    """

    idx = jnp.interp(value, grid_1d, jnp.arange(len(grid_1d)))
    return idx

def interp_2d_extrapolate(x, y, x_grid, y_grid, z_table):
    """
    Performs 2D bilinear interpolation with linear extrapolation.
    z_table must have shape (len(x_grid), len(y_grid)).
    """
    # 1. Find indices, subtract 1 to get the lower bound of the box
    idx_x = jnp.searchsorted(x_grid, x, side='right') - 1
    idx_y = jnp.searchsorted(y_grid, y, side='right') - 1
    
    # 2. Clip indices to ensure we always grab a valid 2x2 box at the edges
    # If x is off the map to the right, this anchors the box at the highest 2 grid points.
    idx_x = jnp.clip(idx_x, 0, x_grid.shape[0] - 2)
    idx_y = jnp.clip(idx_y, 0, y_grid.shape[0] - 2)
    
    x0 = x_grid[idx_x]
    x1 = x_grid[idx_x + 1]
    y0 = y_grid[idx_y]
    y1 = y_grid[idx_y + 1]
    
    # 3. Calculate fractional distances (these will safely go >1 or <0 during extrapolation)
    tx = (x - x0) / (x1 - x0)
    ty = (y - y0) / (y1 - y0)
    
    # 4. Extract the 4 corners of the box
    z00 = z_table[idx_x, idx_y]
    z10 = z_table[idx_x + 1, idx_y]
    z01 = z_table[idx_x, idx_y + 1]
    z11 = z_table[idx_x + 1, idx_y + 1]
    
    # 5. Bilinear combination (carries the gradients smoothly everywhere)
    z = (1.0 - tx) * (1.0 - ty) * z00 + \
        tx * (1.0 - ty) * z10 + \
        (1.0 - tx) * ty * z01 + \
        tx * ty * z11
        
    return z

# -----------------------------------------------------------------------------------------------------------------------
# Map Classes
# -----------------------------------------------------------------------------------------------------------------------


@register
class CompressorMap(eqx.Module):
    tag: str = init_field("Compressor Map", static=True)

    # 1D Grid Axes
    alpha_grid: jnp.ndarray = empty_array()  # FADEC Inlet Guide Vane Angle
    Nc_grid: jnp.ndarray = empty_array()  # Corrected Speed
    Rline_grid: jnp.ndarray = empty_array()  # Orthogonal Coordinate

    # 3D Data Tables (Shape: [len(alpha_grid), len(Nc_grid), len(PR_grid)])
    Wc_table: jnp.ndarray = empty_array()  # Mass flow rate
    PR_table: jnp.ndarray = empty_array()  # Pressure Ratio
    eff_table: jnp.ndarray = empty_array()  # Isentropic Efficiency

    # Map scaling values
    Rline_stall: float = 1.0

    s_Wc: float = 1.0
    s_PR: float = 1.0
    s_eff: float = 1.0
    s_Nc: float = 1.0

    Nc_des: float = 1.0
    alpha_des: float = 0.0
    PR_des: float = 5.0
    Wc_des: float = 25.0
    eff_des: float = 0.85
    Rline_des: float = 2.0
    Rline_stall: float = 1.0

    def evaluate(self, alpha, Nc, Rline):
        # Speed scaling
        Nc_map = Nc / self.s_Nc

        # Helper to grab all 3 values from a specific alpha 2D slice
        def eval_slice(alpha_idx):
            Wc = interp_2d_extrapolate(Nc_map, Rline, self.Nc_grid, self.Rline_grid, self.Wc_table[alpha_idx])
            PR = interp_2d_extrapolate(Nc_map, Rline, self.Nc_grid, self.Rline_grid, self.PR_table[alpha_idx])
            eff = interp_2d_extrapolate(Nc_map, Rline, self.Nc_grid, self.Rline_grid, self.eff_table[alpha_idx])
            return PR, Wc, eff

        # Find blending weight between the first and last alpha grids
        a0, a1 = self.alpha_grid[0], self.alpha_grid[-1]
        
        # Prevent division by zero if the map only has 1 alpha value
        denom = jnp.maximum(a1 - a0, 1e-9)
        w = jnp.clip((alpha - a0) / denom, 0.0, 1.0)
        
        # Evaluate the two edge maps
        PR_0, Wc_0, eff_0 = eval_slice(0)
        PR_1, Wc_1, eff_1 = eval_slice(-1)
        
        # Linearly blend
        PR_map = PR_0 + w * (PR_1 - PR_0)
        Wc_map = Wc_0 + w * (Wc_1 - Wc_0)
        eff_map = eff_0 + w * (eff_1 - eff_0)

        # Apply output scaling
        Wc = Wc_map * self.s_Wc
        PR = (PR_map - 1.0) * self.s_PR + 1.0
        eff = eff_map * self.s_eff

        return PR, Wc, eff

    @classmethod
    def from_json(cls, filepath: str | Path):
        """Loads a JSON compressor map and initializes the Equinox module."""

        # Read raw JSON data
        with open(filepath, "r") as f:
            data = json.load(f)

        # Extract 1D grids
        alpha_grid = jnp.array(data["alpha"])
        Nc_grid = jnp.array(data["Nc"])
        Rline_grid = jnp.array(data["Rline"])

        # Define expected 3D shape
        shape = (len(alpha_grid), len(Nc_grid), len(Rline_grid))

        # Extract and reshape 3D tables
        Wc_table = jnp.array(data["Wc"]).reshape(shape)
        PR_table = jnp.array(data["PR"]).reshape(shape)
        eff_table = jnp.array(data["eff"]).reshape(shape)

        return cls(
            tag=Path(filepath).stem,
            Nc_des=float(data["Nc_des"]),
            alpha_des=float(data["alpha_des"]),
            PR_des=float(data.get("PR_des", 10.0)),
            Wc_des=float(data.get("Wc_des", 25.0)),
            eff_des=float(data.get("eff_des", 0.85)),
            Rline_des=float(data["Rline_des"]),
            Rline_stall=float(data["Rline_stall"]),
            alpha_grid=alpha_grid,
            Nc_grid=Nc_grid,
            Rline_grid=Rline_grid,
            Wc_table=Wc_table,
            PR_table=PR_table,
            eff_table=eff_table,
        )


@register
class TurbineMap(eqx.Module):
    tag: str = init_field("Turbine Map", static=True)

    # 1D Grid Axes
    alpha_grid: jnp.ndarray = empty_array()  # Turbine Nozzle Ratio
    Np_grid: jnp.ndarray = empty_array()
    PR_grid: jnp.ndarray = empty_array()

    # 2D Data Tables (Shape: [len(Nc_grid), len(PR_grid)])
    Wp_table: jnp.ndarray = empty_array()
    eff_table: jnp.ndarray = empty_array()

    # Map scaling Values
    s_Wp: float = 1.0
    s_PR: float = 1.0
    s_eff: float = 1.0
    s_Np: float = 1.0

    alpha_des: float = 0.0
    Np_des: float = 100.0
    PR_des: float = 5.0
    Wp_des: float = 5.0
    eff_des: float = 0.85

    def evaluate(self, alpha, Np, PR):
        # Un-scale the inputs to read the base map
        Np_map = Np / self.s_Np 
        PR_map = (PR - 1.0) * self.s_PR + 1.0

        def eval_slice(alpha_idx):
            Wp = interp_2d_extrapolate(Np_map, PR_map, self.Np_grid, self.PR_grid, self.Wp_table[alpha_idx])
            eff = interp_2d_extrapolate(Np_map, PR_map, self.Np_grid, self.PR_grid, self.eff_table[alpha_idx])
            return Wp, eff

        a0, a1 = self.alpha_grid[0], self.alpha_grid[-1]
        denom = jnp.maximum(a1 - a0, 1e-9)
        w = jnp.clip((alpha - a0) / denom, 0.0, 1.0)
        
        Wp_0, eff_0 = eval_slice(0)
        Wp_1, eff_1 = eval_slice(-1)
        
        Wp_map = Wp_0 + w * (Wp_1 - Wp_0)
        eff_map = eff_0 + w * (eff_1 - eff_0)

        # Apply output scalars
        Wp = Wp_map * self.s_Wp
        eff = eff_map * self.s_eff

        return Wp, eff

    @classmethod
    def from_json(cls, filepath: str | Path):
        """Loads a JSON compressor map and initializes the Equinox module."""

        # Read raw JSON data
        with open(filepath, "r") as f:
            data = json.load(f)

        # Extract 1D grids
        alpha_grid = jnp.array(data["alpha"])
        Np_grid = jnp.array(data["Np"])
        PR_grid = jnp.array(data["PR"])

        # Define expected 3D shape
        shape = (len(alpha_grid), len(Np_grid), len(PR_grid))

        # Extract and reshape 3D tables
        Wp_table = jnp.array(data["Wp"]).reshape(shape)
        eff_table = jnp.array(data["eff"]).reshape(shape)

        return cls(
            tag=Path(filepath).stem,
            alpha_des=float(data["alpha_des"]),
            Np_des=float(data["Np_des"]),
            PR_des=float(data["PR_des"]),
            Wp_des=float(data.get("Wp_des", 5.0)),
            eff_des=float(data.get("eff_des", 0.85)),
            alpha_grid=alpha_grid,
            Np_grid=Np_grid,
            PR_grid=PR_grid,
            Wp_table=Wp_table,
            eff_table=eff_table,)