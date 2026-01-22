# RCAIDE/Framework/Analyses/Propulsion/Turbofan_Performance.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: May, 2025, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
# IMPORT 
# ----------------------------------------------------------------------------------------------------------------------

import chex

import RCAIDE.Framework as rcf
from RCAIDE.Framework import Process, ProcessStep

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

    tag: str = 'Turbofan Performance'

    def __post_init__(self):
        default_steps = [
            ("Inlet Nozzle", inlet_nozzle_performance),
            ("Fan", fan_performance),
            ("Compressor", compressor_performance),
            ("Combustor", turbojet_combustor_performance),
            ("Turbine", turbine_performance),
            ("Core Nozzle", core_nozzle_performance),
            ("Fan Nozzle", fan_nozzle_performance),
            ("Thrust", thrust_and_power)
        ]

        for name, function in default_steps:
            self.append(ProcessStep(tag=name, function=function))


