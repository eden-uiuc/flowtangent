# RCAIDE/Framework/Analyses/Propulsion/Turbofan_Performance.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: May, 2025, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
# IMPORT 
# ----------------------------------------------------------------------------------------------------------------------

import equinox as eqx

from RCAIDE.utils import init_field

from RCAIDE.Framework import Process, ProcessStep

from RCAIDE.Library.Methods.Energy.Transmission.Nozzles import *
from RCAIDE.Library.Methods.Energy.Transmission.Turbines import *
from RCAIDE.Library.Methods.Energy.Transmission.Turbofans import *
from RCAIDE.Library.Methods.Energy.Transmission.Combustors import *
from RCAIDE.Library.Methods.Energy.Transmission.Fan_Compressors import *
from RCAIDE.Library.Methods.Energy.Transmission.Shaft_Offtake import *


# ----------------------------------------------------------------------------------------------------------------------
# Turbofan_Performance
# ----------------------------------------------------------------------------------------------------------------------

def _build_turbofan_steps() -> tuple[ProcessStep, ...]:
    """Builds the static pipeline of turbofan cycle analysis steps."""
    return (
        ProcessStep(tag="Thrust", function=thrust_and_power),
    )

class TurbofanPerformance(Process):
    tag: str = init_field('Turbofan Performance', static=True)
    
    # Hand the builder function directly to the factory
    steps: tuple[ProcessStep, ...] = init_field(_build_turbofan_steps)


