# RCAIDE/Library/Methods/Mass/Propulsion/Jet_Mass_from_SLS.py
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


# package imports

# RCAIDE Imports
# from RCAIDE.Library.Components.Energy.Propulsors import TurbofanEngine

# -------------------------------------------------------------------------------
#  Functional/Library Version
# -------------------------------------------------------------------------------


def func_tf_mass_from_SLS(sls_thrust: float):

    t_lbf = sls_thrust * 0.224809  # Convert to lbf
    mass = (0.4054 * t_lbf**0.9255) * 0.453592

    return mass


# -------------------------------------------------------------------------------
#  Stateful/Framework Version
# -------------------------------------------------------------------------------


# def tf_mass_from_SLS(
#     state: State,
#     system: Aircraft,
#     settings: Settings,
# ):
#     """
#     Framework version of tf_mass_from_SLS. Assumes a turbofan engine.

#     See Also
#     --------
#     func_tf_Mass_from_SLS:
#         Functional implementation which this method calls.
#     """

#     def update_tf_mass(node):
#         if (
#             isinstance(node, TurbofanEngine)
#             and node.mass_properties.total == 0.0
#             and node.design_parameters.SLS_thrust != 0.0
#         ):
#             return eqx.tree_at(
#                 lambda t: t.mass_properties.total, node, func_tf_mass_from_SLS(node.design_parameters.SLS_thrust)
#             )

#     updated_system = jax.tree_util.tree_map(update_tf_mass, system, is_leaf=lambda x: isinstance(x, TurbofanEngine))

#     return state, updated_system, settings
