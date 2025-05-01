# RCAIDE/Framework/Analyses/Propulsion/Turbofan_Performance.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: May, 2025, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
# IMPORT 
# ----------------------------------------------------------------------------------------------------------------------

from dataclasses import dataclass, field, make_dataclass

import RCAIDE.Framework as rcf
import RCAIDE.Library as rcl

from RCAIDE.Library.Methods.Propulsors.Converters import *
from RCAIDE.Library.Methods.Propulsors.Turbofan import thrust_and_power


# ----------------------------------------------------------------------------------------------------------------------
# Turbofan_Performance
# ----------------------------------------------------------------------------------------------------------------------


@dataclass(kw_only=True)
class TurbofanPerformance(rcf.Process):

    name: str = 'Turbofan Performance'

    def __post_init__(self):

        self.steps = [
            compression_nozzle_performance,
            fan_performance,
            compressor_performance,
            turbojet_combustor_performance,
            turbine_performance,
            core_nozzle_performance,
            fan_nozzle_performance,
            thrust_and_power
        ]


