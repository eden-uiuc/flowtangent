# RCAIDE/Library/Attributes/Solids/Aluminum.py
# (c) Copyright 2023 Aerospace Research Community LLC
 
# Created: Mar 2024 M. Clarke

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

import chex

from RCAIDE.Library.Attributes.Materials import Solid

#-------------------------------------------------------------------------------
# Aluminum
#------------------------------------------------------------------------------- 


@chex.dataclass
class Aluminum(Solid):
    """
    Physical Constants Specific to 6061-T6 Aluminum

    Source:
            Cao W, Zhao C, Wang Y, et al. Thermal modeling of full-size-scale cylindrical battery pack cooled
            by channeled liquid flow[J]. International journal of heat and mass transfer, 2019, 138: 1178-1187.
    """

    density                    = 2719
    thermal_conductivity       = 202.4
    specific_heat_capacity     = 871
