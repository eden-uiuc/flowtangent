# flowtangent/Library/Methods/Mass/Propulsion/Jet_Mass_from_SLS.py
# (c) Copyright 2025 Aerospace Research Community LLC#
# Created:  May 2025, J. Smart
# Modified:
# -------------------------------------------------------------------------------
#  Imports
# -------------------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


# package imports

# FlowTangent Imports
# from FlowTangent.Library.Components.Energy.Propulsors import TurbofanEngine

# -------------------------------------------------------------------------------
#  Functional/Library Version
# -------------------------------------------------------------------------------


def func_tf_mass_from_SLS(sls_thrust: float):

    t_lbf = sls_thrust * 0.224809  # Convert to lbf
    mass = (0.4054 * t_lbf**0.9255) * 0.453592

    return mass
