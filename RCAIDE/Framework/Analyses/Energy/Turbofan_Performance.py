# RCAIDE/Framework/Analyses/Propulsion/Turbofan_Performance.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: May, 2025, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
# IMPORT 
# ----------------------------------------------------------------------------------------------------------------------

import chex
from dataclasses import field, make_dataclass

import RCAIDE.Framework as rcf
import RCAIDE.Library as rcl


from RCAIDE.Library.Methods.Energy.Converters.Nozzles import *
from RCAIDE.Library.Methods.Energy.Converters.Turbines import *
from RCAIDE.Library.Methods.Energy.Converters.Turbofans import *
from RCAIDE.Library.Methods.Energy.Converters.Combustors import *
from RCAIDE.Library.Methods.Energy.Converters.Fan_Compressors import *
from RCAIDE.Library.Methods.Energy.Converters.Shaft_Offtake import *


# ----------------------------------------------------------------------------------------------------------------------
# Turbofan_Performance
# ----------------------------------------------------------------------------------------------------------------------


@chex.dataclass(kw_only=True)
class TurbofanPerformance(rcf.Process):

    name: str = 'Turbofan Performance'

    def __post_init__(self):

        self.steps = [
            inlet_nozzle_performance,
            fan_performance,
            compressor_performance,
            turbojet_combustor_performance,
            turbine_performance,
            core_nozzle_performance,
            fan_nozzle_performance,
            thrust_and_power
        ]


