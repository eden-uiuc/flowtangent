# RCAIDE/Library/Methods/Propulsors/Converters/__init__.py
# (c) Copyright 2023 Aerospace Research Community LLC

""" RCAIDE Package Setup
"""

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from . import Combustor
from . import Compression_Nozzle
from . import DC_Motor
from . import Engine


from .combustor import func_combustor_performance, turbojet_combustor_performance
from .fan_compressor import func_fan_compressor_performance, fan_performance, compressor_performance
from .nozzles import (func_isentropic_nozzle_performance,
                      func_compression_nozzle_performance,
                      func_expansion_nozzle_performance,
                      compression_nozzle_performance,
                      fan_nozzle_performance,
                      core_nozzle_performance)
from .turbine import func_turbine_performance, turbine_performance