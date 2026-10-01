# flowtangent/Framework/Analyses/Energy/__init__.py
# (c) Copyright 2023 Aerospace Research Community LLC

"""FlowTangent Package Setup"""

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from ._energy_network import PACTAnalysis

from .jets import (
    JetSettings,
    build_turbofan_design,
    build_turbofan_performance,
    build_turbojet_design,
    build_turbojet_performance,
)

__all__ = [
    "PACTAnalysis",
    "JetSettings",
    "build_turbofan_design",
    "build_turbofan_performance",
    "build_turbojet_design",
    "build_turbojet_performance",
]
