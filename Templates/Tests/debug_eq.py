import jax.numpy as np
import dataclasses
import chex

def check_equality_safety(obj, path="root", visited=None):
    if visited is None:
        visited = set()
    
    # Handle primitives and basic types to avoid noise
    if isinstance(obj, (int, float, str, bool, type(None))):
        return

    # Skip actual arrays (they are expected to return arrays on eq)
    try:
        if hasattr(obj, 'shape') and hasattr(obj, 'dtype'):
            return
    except Exception:
        pass # Object crashes on attribute access (e.g. UnitRegistry), not an array

    # Detect classes (types) which shouldn't be recursed into like instances
    if isinstance(obj, type):
        # print(f"[INFO - CLASS FOUND] {path}: Found class '{obj.__name__}'. Skipping recursion.")
        return
    
    # Handle cycles
    obj_id = id(obj)
    if obj_id in visited:
        return
    visited.add(obj_id)

    # Check if object equality is safe
    try:
        # Perform comparison
        is_equal = (obj == obj)
        
        # Check 1: Does it return an array?
        if hasattr(is_equal, 'shape') and is_equal.shape != ():
             print(f"[UNSAFE EQ - ARRAY RETURN] {path}: Type={type(obj).__name__} returned array {is_equal.shape} for (obj == obj). Needs identity __eq__.")
        
        # Check 2: Does it fail boolean conversion?
        try:
            if hasattr(is_equal, 'size') and is_equal.size > 1:
                 # This is the exact condition that triggers ValueError in equinox/jax
                 bool(is_equal) 
        except ValueError:
             print(f"[CRITICAL FAILURE - BOOL CONVERSION] {path}: Type={type(obj).__name__} raises ValueError on bool(obj == obj). FIX THIS CLASS.")
             
    except Exception as e:
        # Some objects might crash on equality for other reasons
        pass
        # print(f"[EXCEPTION ON EQ] {path}: Type={type(obj).__name__} crashed: {e}")

    # Recurse
    if dataclasses.is_dataclass(obj):
        for field in dataclasses.fields(obj):
            val = getattr(obj, field.name)
            check_equality_safety(val, f"{path}.{field.name}", visited)
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            check_equality_safety(item, f"{path}[{i}]", visited)
    elif isinstance(obj, dict):
        for k, v in list(obj.items()):
            check_equality_safety(v, f"{path}['{k}']", visited)
            # Keys must be hashable/comparable too
            check_equality_safety(k, f"{path}.key({k})", visited)
    elif hasattr(obj, '__dict__'):
         # Custom objects without dataclass structure
         for k, v in list(obj.__dict__.items()):
            if k.startswith("__"): continue
            check_equality_safety(v, f"{path}.{k}", visited)
