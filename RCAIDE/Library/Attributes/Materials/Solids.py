# RCAIDE/Library/Attributes/Solids/Solid.py
# (c) Copyright 2023 Aerospace Research Community LLC

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

# package imports
import equinox as eqx

from RCAIDE.utils import init_field

#-------------------------------------------------------------------------------
# Solid Data Class
#-------------------------------------------------------------------------------


class Solid(eqx.Module):

    ultimate_tensile_strength:  float | None = init_field(None, static=True)
    ultimate_shear_strength:    float | None = init_field(None, static=True)
    ultimate_bearing_strength:  float | None = init_field(None, static=True)
    yield_tensile_strength:     float | None = init_field(None, static=True)
    yield_shear_strength:       float | None = init_field(None, static=True)
    yield_bearing_strength:     float | None = init_field(None, static=True)
    minimum_gage_thickness:     float | None = init_field(None, static=True)
    density:                    float | None = init_field(None, static=True)

class Aluminum(Solid):
    """
    Physical Constants Specific to 6061-T6 Aluminum

    Source:
            Cao W, Zhao C, Wang Y, et al. Thermal modeling of full-size-scale cylindrical battery pack cooled
            by channeled liquid flow[J]. International journal of heat and mass transfer, 2019, 138: 1178-1187.
    """

    density                    : float | None = init_field(2719, static=True)
    thermal_conductivity       : float | None = init_field(202.4, static=True)
    specific_heat_capacity     : float | None = init_field(871, static=True)
