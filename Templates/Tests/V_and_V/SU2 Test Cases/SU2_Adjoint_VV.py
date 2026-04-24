import json
import os
import pysu2
import numpy as np

CACHE_FILE = "su2_run_cache.json"

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_cache(cache_dict):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache_dict, f, indent=4)

def run_su2_primal_raw(alpha_val):
    cache = load_cache()
    key = f"primal_alpha_{float(alpha_val):.4f}"
    
    # If we already ran this, return the cached value immediately!
    if key in cache:
        print(f"Loading Primal from cache for Alpha = {alpha_val}")
        return np.float64(cache[key])
        
    print(f"Running SU2 Primal for Alpha = {alpha_val}...")
    # modify_cfg("ONERA_M6_Euler.cfg", "AOA", float(alpha_val))
    driver = pysu2.CSinglezoneDriver("ONERA_M6_Euler.cfg", 1, [])
    driver.StartSolver()
    driver.Postprocessing()
    
    # Example extraction (depends on SU2 API version)
    cl = 0.25 # Replace with actual extraction
    
    # Save the result before returning
    cache[key] = float(cl)
    save_cache(cache)
    
    return np.float64(cl)

def run_su2_adjoint_raw(alpha_val):
    cache = load_cache()
    key = f"adjoint_alpha_{float(alpha_val):.4f}"
    
    if key in cache:
        print(f"Loading Adjoint from cache for Alpha = {alpha_val}")
        return np.float64(cache[key])
        
    print(f"Running SU2 Adjoint for Alpha = {alpha_val}...")
    driver = pysu2.CDiscAdjSinglezoneDriver("ONERA_M6_Euler.cfg", 1, [])
    driver.StartSolver()
    driver.Postprocessing()
    
    # Example extraction
    dcl_dalpha = 4.5 # Replace with actual extraction
    
    cache[key] = float(dcl_dalpha)
    save_cache(cache)
    
    return np.float64(dcl_dalpha)