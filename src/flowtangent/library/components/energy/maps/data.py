

import json
import os
from functools import lru_cache
from typing import Any

import pycycle.api as pyc

from eden_trace.utils import get_trace_root
from eden_trace.library import units
from eden_trace.library.components.energy.maps.classes import CompressorMap, TurbineMap
# -----------------------------------------------------------------------------------------------------------------------
# Map Specifications (Sourced from PyCycle)
# -----------------------------------------------------------------------------------------------------------------------

_MAP_DIR = get_trace_root() / "library/data/turbo_maps"
STUB_FILE = get_trace_root() / "library/components/energy/maps/data.pyi"

@lru_cache(maxsize=None)
def _load_map_from_disk(name: str):
    """Hidden helper that does the disk I/O, safely cached, and routes by type."""
    file_path = _MAP_DIR / f"{name}.json"
    if not file_path.exists():
        raise AttributeError(f"Map '{name}' not found in Trace library ({_MAP_DIR}).")

    # 1. Peek inside the JSON to grab the metadata
    with open(file_path, "r") as f:
        data = json.load(f)

    # 2. Extract the type (defaulting to compressor for legacy safety)
    map_type = data.get("type", "compressor").lower()

    # 3. Dispatch to the correct class
    # (Assuming your classes have a classmethod like .from_dict() or .from_json())
    if map_type == "compressor":
        return CompressorMap.from_json(file_path)
    elif map_type == "turbine":
        return TurbineMap.from_json(file_path)
    else:
        raise ValueError(f"Unrecognized map type '{map_type}' in {name}.json")


def __getattr__(name: str) -> Any:
    """Intercepts module-level attribute access."""
    # Ignore private attributes to prevent messing with Python internals
    if name.startswith("_"):
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    return _load_map_from_disk(name)


def __dir__():
    """Allows IDEs and the `dir()` command to see the available maps."""
    # List all .json files in the directory without their extensions
    if _MAP_DIR.exists():
        return [f.stem for f in _MAP_DIR.glob("*.json")]
    return []


def harvest_pycycle_maps(output_dir=_MAP_DIR):
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
        "RlineStall": "Rline_stall",
    }

    # The scalar design/anchor points PyCycle uses to center the map
    scalar_mapping = {
        "alphaMap": "alpha_des",
        "NcMap": "Nc_des",
        "NpMap": "Np_des",
        "PRmap": "PR_des",
        "PR": "PR_des",
        "RlineMap": "Rline_des",
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
                    if val_units == "rpm":
                       val = val
                    else:
                        val = val * units.parse(val_units)
                if hasattr(val, "tolist"):
                    json_data[json_key] = val.tolist()
                else:
                    json_data[json_key] = val

        # 2. Harvest Scalar Design Parameters
        for pyc_attr, json_key in scalar_mapping.items():
            val = map_obj.defaults.get(pyc_attr, None)
            val_units = map_obj.units.get(pyc_attr, None)
            if val_units is not None:
                if val_units == "rpm":
                        val = val
                else:
                    val = val * units.parse(val_units)
            if val is not None:
                json_data[json_key] = val

        # Initial Write to Disk
        file_path = os.path.join(output_dir, f"{map_name}.json")
        with open(file_path, "w") as f:
            json.dump(json_data, f, indent=4)
        
        # Update design values
        test_map = _load_map_from_disk(map_name)
        if isinstance(test_map, CompressorMap):
            PR_map, Wc_map, eff_map = test_map.evaluate(
                alpha=test_map.alpha_des,
                Nc=test_map.Nc_des,
                Rline=test_map.Rline_des
            )
            json_data["PR_des"] = PR_map.item()
            json_data["Wc_des"] = Wc_map.item()
            json_data["eff_des"] = eff_map.item()
        
        if isinstance(test_map, TurbineMap):
            Wp_map, eff_map = test_map.evaluate(
                alpha=test_map.alpha_des,
                Np=test_map.Np_des,
                PR=test_map.PR_des)
            
            json_data["Wp_des"] = Wp_map.item()
            json_data["eff_des"] = eff_map.item()
        
        #Final Write
        with open(file_path, "w") as f:
            json.dump(json_data, f, indent=4)
        



        print(f"Successfully harvested {map_name} ({map_type}) to {file_path}")

def generate_stub():
    lines = [
        "from typing import Any",
        "from .classes import CompressorMap, TurbineMap", 
        "",
    ]
    
    for map_file in _MAP_DIR.glob("*.json"):
        with open(map_file, "r") as f:
            data = json.load(f)
        
        map_type = data.get("type", "compressor").lower()
        type_hint = "CompressorMap" if map_type == "compressor" else "TurbineMap"
        
        # Write the attribute to the stub file
        lines.append(f"{map_file.stem}: {type_hint}")

    STUB_FILE.write_text("\n".join(lines))
    print(f"Generated {STUB_FILE.name} with {len(lines) - 3} maps.")

if __name__ == "__main__":
    harvest_pycycle_maps()
    generate_stub()
