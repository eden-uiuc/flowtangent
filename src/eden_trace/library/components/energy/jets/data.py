import csv
import json
import os

from functools import lru_cache
from typing import Any

from eden_trace.utils import get_trace_root
from eden_trace.library.components.energy.jets.classes import TurbojetEngine

#-----------------------------------------------------------------------------------------------------------------------
# Data Collection (Sourced from Mattingly)
#-----------------------------------------------------------------------------------------------------------------------

_DATA_DIR = get_trace_root() / "library/data/turbo_engines"

def load_csv_as_dicts(filepath):
    """Loads a standard CSV into a list of dictionaries."""
    if not os.path.exists(filepath):
        return []
    with open(filepath, mode='r', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def parse_value(val):
    """Converts string values to floats, handling textbook dashes."""
    val = val.strip()
    if val == '-' or val == '':
        return None
    elif val =="true":
        return True
    elif val=="false":
        return False
    try:
        return float(val)
    except ValueError:
        return val

def process_station_data(data_dir=_DATA_DIR):
    """
    Processes engine-specific station thermal data into JSONs
    Source: Mattingly, 2nd Ed.
    """
    # Load the thermal data (Table C.4)
    with open(data_dir / 'station_data.csv', mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        engine_names = headers[1:]  # Skip 'Parameter'
        
        # Initialize dictionaries for each engine in C.4
        thermal_data = {engine: {"name": engine, "station_data": {}} for engine in engine_names}
        
        for row in reader:
            param = row[0]
            for i, engine in enumerate(engine_names):
                val = parse_value(row[i+1])
                # Separate general performance metrics from strict station arrays
                if param in ["Bypass ratio", "Thrust (lbf)", "Airflow (lbm/s)"]:
                    thermal_data[engine][param] = val
                else:
                    thermal_data[engine]["station_data"][param] = val

    # Dump the pivoted thermal data to JSON files
    os.makedirs('mattingly_json', exist_ok=True)
    
    for engine, data in thermal_data.items():
        # Clean up filenames (e.g., F100-PW-100)
        safe_name = engine.replace(' ', '_').replace('/', '_')
        filepath = os.path.join('mattingly_json', f'{safe_name}_thermal.json')
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Exported: {filepath}")


def process_design_data(csv_file):
    """
    Reads a design point CSV and exports individual engine JSONs.
    Source: Mattingly, 2nd Ed.
    """
    if not os.path.exists(csv_file):
        print(f"Warning: Could not find {csv_file}. Skipping.")
        return
    
    category = csv_file.stem.split('_')[0].title()
    
    out_dir = _DATA_DIR
    os.makedirs(out_dir, exist_ok=True)
    
    with open(csv_file, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Grab the model name to use as the filename, sanitizing any slashes or spaces
            engine_name = row.get('Model no.', 'Unknown_Engine')
            print(f"Exporting: {engine_name}")
            engine_type = csv_file.stem.split('_')[1].title()[:-1]
            safe_name = engine_name.replace('/', '_').replace(' ', '_').replace('-','_')
            
            header_data = {
                "category": category,
                "type": engine_type
            }

            # Clean up the dictionary values
            clean_data = header_data | {k: parse_value(v) for k, v in row.items()}

            #Assumed values for afterburner exit temperature
            if row.get('AB', False):
                if engine_type == "Turbojet":
                    clean_data['AET (F)'] = 2600.0
                elif engine_type == "Turbofan":
                    clean_data['AET (F)'] = 3200.0
            
            if category == "Civil":
                clean_data['TIT (F)'] = 2300.0
            
            # Concorde overrides for afterburning civil engine
            if "Olympus" in engine_name:
                clean_data['AB'] = True
                clean_data['AET (F)'] = 2600.0
                clean_data['TIT (F)'] = 2200.0

            
            # Write to JSON
            filepath = os.path.join(out_dir, f"{safe_name}.json")
            with open(filepath, 'w') as out_f:
                json.dump(clean_data, out_f, indent=4)

# -----------------------------------------------------------------------------------------------------------------------
# Engine Loading
# -----------------------------------------------------------------------------------------------------------------------

STUB_FILE = get_trace_root() / "library/components/energy/jets/data.pyi"

@lru_cache(maxsize=None)
def _load_engine_from_disk(name: str):
    file_path = _DATA_DIR / f"{name}.json"
    if not file_path.exists():
        raise AttributeError(f"Engine {name} not found in Trace library ({_DATA_DIR}).")

    return TurbojetEngine.from_json(filepath=file_path)

def __getattr__(name: str) -> Any:
    """Intercepts module-level attribute access."""
    # Ignore private attributes to prevent messing with Python internals
    if name.startswith("_"):
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    return _load_engine_from_disk(name)


def __dir__():
    """Allows IDEs and the `dir()` command to see the available maps."""
    # List all .json files in the directory without their extensions
    if _DATA_DIR.exists():
        return [f.stem for f in _DATA_DIR.glob("*.json")]
    return []

def generate_stub():
    lines = [
        "from .classes import TurbojetEngine",  # Update import path
        "",
    ]
    
    for eng_file in _DATA_DIR.glob("*.json"):
        # Write the attribute to the stub file
        lines.append(f"{eng_file.stem}: {"TurbojetEngine"}")

    STUB_FILE.write_text("\n".join(lines))
    print(f"Generated {STUB_FILE.name} with {len(lines) - 3} engines.")      


if __name__ == "__main__":
    print("="*50 + "\nBuilding Engine Library from Mattingly Data...\n" + "-"*50)
    process_design_data(_DATA_DIR/"CSVs/civil_turbofans.csv")
    process_design_data(_DATA_DIR/"CSVs/military_turbofans.csv")
    process_design_data(_DATA_DIR/"CSVs/military_turbojets.csv")
    print("Library Build Complete.")
    generate_stub()