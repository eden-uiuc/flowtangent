# RCAIDE/Library/Attributes/Solids/Solid.py
# (c) Copyright 2023 Aerospace Research Community LLC 
 
#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

import chex

#-------------------------------------------------------------------------------
# Solid Data Class
#------------------------------------------------------------------------------- 


@chex.dataclass(kw_only=True, slots=True)
class Solid:

    ultimate_tensile_strength:  float = None
    ultimate_shear_strength:    float = None
    ultimate_bearing_strength:  float = None
    yield_tensile_strength:     float = None
    yield_shear_strength:       float = None
    yield_bearing_strength:     float = None
    minimum_gage_thickness:     float = None
    density:                    float = None
