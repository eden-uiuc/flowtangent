import json
from os import path
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates

from RCAIDE.utils import empty_array, init_field
from RCAIDE.Library import Units

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



# -----------------------------------------------------------------------------------------------------------------------
# Map Classes
# -----------------------------------------------------------------------------------------------------------------------


class CompressorMap(eqx.Module):
    tag: str = init_field("Compressor Map", static=True)

    # 1D Grid Axes
    alpha_grid: jnp.ndarray = empty_array()  # FADEC Inlet Guide Vane Angle
    Nc_grid: jnp.ndarray = empty_array()  # Corrected Speed
    Rline_grid: jnp.ndarray = empty_array()  # Orthogonal Coordinate

    # 3D Data Tables (Shape: [len(alpha_grid), len(Nc_grid), len(PR_grid)])
    Wc_table: jnp.ndarray = empty_array()  # Mass flow rate
    PR_table: jnp.ndarray = empty_array()  # Pressure Ratio
    eff_table: jnp.ndarray = empty_array()  # Polytropic Efficiency

    # Map scaling values
    Rline_stall: float = 1.0
    
    s_WC: float = 1.0
    s_PR: float = 1.0
    s_eff: float = 1.0
    s_Nc: float = 1.0

    Nc_des: float = 1.0
    alpha_des: float = 0.0
    Rline_des: float = 2.0
    Rline_stall: float = 1.0

    def evaluate(
        self,
        alpha,
        Nc,
        Rline,
    ):

        # Speed scaling
        Nc_map = Nc / self.s_Nc

        # Get fractional coordinates
        idx_alpha = get_fractional_coords(self.alpha_grid, alpha)
        idx_Nc = get_fractional_coords(self.Nc_grid, Nc_map)
        idx_Rline = get_fractional_coords(self.Rline_grid, Rline)
        coords = jnp.stack([idx_alpha, idx_Nc, idx_Rline])

        # Interpolate values
        Wc_map = map_coordinates(self.Wc_table, coords, order=1)
        PR_map = map_coordinates(self.PR_table, coords, order=1)
        eff_map = map_coordinates(self.eff_table, coords, order=1)

        # Apply scaling
        Wc = Wc_map * self.s_WC
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
        Nc_grid    = jnp.array(data["Nc"])
        Rline_grid = jnp.array(data["Rline"])
        
        # Define expected 3D shape
        shape = (len(alpha_grid), len(Nc_grid), len(Rline_grid))
        
        # Extract and reshape 3D tables
        Wc_table  = jnp.array(data["Wc"]).reshape(shape)
        PR_table  = jnp.array(data["PR"]).reshape(shape)
        eff_table = jnp.array(data["eff"]).reshape(shape)
        
        
        return cls(
            alpha_grid=alpha_grid,
            Nc_grid=Nc_grid,
            Rline_grid=Rline_grid,
            Wc_table=Wc_table,
            PR_table=PR_table,
            eff_table=eff_table,
            Nc_des = data["Nc_des"],
            alpha_des = data['alpha_des'],
            Rline_des=data["Rline_des"],
            Rline_stall=data["Rline_stall"],
        )


class TurbineMap(eqx.Module):
    
    tag: str = init_field("Turbine Map", static=True)
    
    # 1D Grid Axes
    alpha_grid: jnp.ndarray = empty_array() # Turbine Nozzle Ratio
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
    Np_des: float = 1.0

    def evaluate(self, alpha, Np, PR):
        # Un-scale the inputs to read the base map
        Np_map = Np / self.s_Np
        PR_map = (PR - 1.0) / self.s_PR + 1.0

        # Get fractional grid coordinates
        idx_alpha = jnp.atleast_2d(get_fractional_coords(self.alpha_grid, alpha))
        idx_Nc = get_fractional_coords(self.Np_grid, Np_map)
        idx_PR = get_fractional_coords(self.PR_grid, PR_map)
        coords = jnp.stack([idx_alpha, idx_Nc, idx_PR])

        # Interpolate values
        Wp_map = map_coordinates(self.Wp_table, coords, order=1)
        eff_map = map_coordinates(self.eff_table, coords, order=1)

        # Apply output scalars
        Wp = Wp_map * self.s_Wp
        eff = eff_map * self.s_eff

        return Wp, eff
    
    @classmethod
    def from_json(cls, filepath: str|Path):
        """Loads a JSON compressor map and initializes the Equinox module."""
        
        # Read raw JSON data
        with open(filepath, "r") as f:
            data = json.load(f)
        
        # Extract 1D grids
        alpha_grid = jnp.array(data["alpha"])
        Np_grid    = jnp.array(data["Np"])
        PR_grid = jnp.array(data["PR"])
        
        # Define expected 3D shape
        shape = (len(alpha_grid), len(Np_grid), len(PR_grid))
        
        # Extract and reshape 3D tables
        Wp_table  = jnp.array(data["Wp"]).reshape(shape)
        eff_table  = jnp.array(data["eff"]).reshape(shape)
        
        return cls(
            alpha_grid=alpha_grid,
            Np_grid=Np_grid,
            PR_grid=PR_grid,
            Wp_table=Wp_table,
            eff_table=eff_table
        )


# -----------------------------------------------------------------------------------------------------------------------
# Map Specifications (Sourced from PyCycle)
# -----------------------------------------------------------------------------------------------------------------------

MapData = Path(path.join(path.dirname(__file__), "Map_Data"))

import os
import json
import pycycle.api as pyc

def harvest_pycycle_maps(output_dir=MapData):
    """Extracts legacy NEPP maps and their design anchors from PyCycle."""
    
    os.makedirs(output_dir, exist_ok=True)

    # Now a dictionary mapping the name to a tuple: (PyCycle Map Object, Map Type)
    maps_to_harvest = {
        "AXI3_2": (pyc.AXI3_2, "compressor"),
        "AXI5": (pyc.AXI5, "compressor"),
        "Fan": (pyc.FanMap, "compressor"),
        "HPC": (pyc.HPCMap, "compressor"),
        "LPC": (pyc.LPCMap, "compressor"),
        "NCPO1": (pyc.NCP01, "compressor"),
        "HPT": (pyc.HPTMap, "turbine"),
        "HPT1269": (pyc.HPT1269, "turbine"),
        "LPT": (pyc.LPTMap, "turbine"),
        "LPT2269": (pyc.LPT2269, "turbine"),
    }

    # The 1D/2D/3D array data
    array_mapping = {
        "alphaMap": "alpha",
        "NpMap": "Np",
        "NcMap": "Nc",
        "RlineMap": "Rline",
        "PRmap": "PR",        
        "WcMap": "Wc",
        "WpMap": "Wp",
        "effMap": "eff",
        "RlineStall": "Rline_stall"
    }

    # The scalar design/anchor points PyCycle uses to center the map
    scalar_mapping = {
        "alphaMap": "alpha_des",
        "NcMap": "Nc_des",
        "NpMap": "Np_des",
        "PR": "PR_des",
        "RlineMap": "Rline_des"
    }

    for map_name, (map_obj, map_type) in maps_to_harvest.items():
        # Initialize JSON dict with the explicit component type
        json_data = {"type": map_type}
        
        # 1. Harvest Arrays
        for pyc_attr, json_key in array_mapping.items():
            if hasattr(map_obj, pyc_attr):
                val = getattr(map_obj, pyc_attr)
                val_units = map_obj.units.get(pyc_attr, None)
                if val_units is not None:
                    if val_units == 'rpm':
                        val = val * Units.rev / Units.mins
                    else:
                        val = val * Units.parse(val_units)
                if hasattr(val, "tolist"):
                    json_data[json_key] = val.tolist()
                else:
                    json_data[json_key] = val
                    
        # 2. Harvest Scalar Design Parameters
        for pyc_attr, json_key in scalar_mapping.items():
            val = map_obj.defaults.get(pyc_attr, None)
            val_units = map_obj.units.get(pyc_attr, None)
            if val_units is not None:
                if val_units == 'rpm':
                    val = val * Units.rev / Units.mins
                else:
                    val = val * Units.parse(val_units)
            if val is not None:
                json_data[json_key] = val
                
        # Write to disk
        file_path = os.path.join(output_dir, f"{map_name}.json")
        with open(file_path, "w") as f:
            json.dump(json_data, f, indent=4)
            
        print(f"Successfully harvested {map_name} ({map_type}) to {file_path}")

if __name__ == "__main__":
    harvest_pycycle_maps()