# flowtangent/Framework/Missions/Conditions/Conditions.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Jul 2024, Flowtangent Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from dataclasses import fields
from typing import Optional, Self, Sequence

import equinox as eqx

# package imports
import jax
import jax.numpy as jnp

from flowtangent.utils import field

# ----------------------------------------------------------------------------------------------------------------------
#  Conditions
# ----------------------------------------------------------------------------------------------------------------------

STATIC_DATA = ("AtmosphericBreakpoints")

def _is_static_node(node):
        return hasattr(node, "__class__") and node.__class__.__name__ in STATIC_DATA

class StateData(eqx.Module):
    tag: str = field("Conditions", static=True)

    @property
    def subconditions(self) -> tuple:
        return tuple(
            getattr(self, f.name)
            for f in fields(self)
            if f.name != "subconditions" and isinstance(getattr(self, f.name), StateData)
        )

    def __getitem__(self, item):
        if isinstance(item, (int, slice)):
            return self.subconditions[item]
        elif isinstance(item, str):
            attr_name = item.replace(" ", "_").lower()
            return getattr(self, attr_name)
        else:
            raise TypeError(f"Conditions indices must be slices, integers or strings, not {type(item).__name__}")

    def __iter__(self):
        return iter(self.subconditions)

    def expand_time(self, N: Optional[int] = None):

        if N is None:
            if hasattr(self, "time"):
                if hasattr(self.time, "N"):
                    N = self.time.N
            else:
                N = 1

        def _expand(leaf):
            if isinstance(leaf, (jnp.ndarray)):
                # Zero-copy expansion for actual data
                if leaf.ndim == 1:
                    # e.g., Shape (X,) -> Shape (n, X)
                    return jnp.broadcast_to(leaf, (N,) + leaf.shape)
                elif leaf.ndim == 2 and leaf.shape[0] == 1:
                    # e.g., Shape (1, X) -> Shape (n, X)
                    return jnp.broadcast_to(leaf, (N, leaf.shape[1]))

            return leaf

        return jax.tree_util.tree_map(_expand, self, is_leaf=_is_static_node)

    def expand_batch(self, batch_size: int):

        def _expand(leaf):
            if _is_static_node(leaf):
                return leaf
            if isinstance(leaf, jnp.ndarray):
                # # Intercept the empty placeholders
                # if leaf.size==0:
                #     trailing_dims = (1,) if leaf.ndim==1 else leaf.shape[1:]
                #     return jnp.zeros((batch_size,) + trailing_dims, dtype=leaf.dtype)
                # # Zero-copy expansion prepending batch dim
                return jnp.broadcast_to(leaf, (batch_size,) + leaf.shape)
            return leaf

        return jax.tree_util.tree_map(_expand, self, is_leaf=_is_static_node)

    @classmethod
    def concatenate(cls, states: Sequence[Self]):
        def _concat(*leaves):
            first_leaf = leaves[0]
            if _is_static_node(first_leaf):
                return first_leaf
            if isinstance(first_leaf, jnp.ndarray):
                return jnp.concatenate(leaves, axis=0)
            return first_leaf
        return jax.tree_util.tree_map(_concat, *states, is_leaf=_is_static_node)

    def truncate(self, size: int):
        def _trunc(leaf):
            # 1. Ignore static classes
            if _is_static_node(leaf):
                return leaf

            # 2. Slice the batch dimension (axis 0) of the array
            if isinstance(leaf, jnp.ndarray):
                return leaf[:size]

            return leaf

        return jax.tree_util.tree_map(_trunc, self, is_leaf=_is_static_node)

    def get_vmap_axes(self):
        def _get_axis(leaf):
            if _is_static_node(leaf):
                # JAX allows prefix-trees for in_axes. Returning None for the whole
                # object tells JAX to broadcast everything inside this class.
                return None
            if isinstance(leaf, jnp.ndarray):
                return 0
            return None

        return jax.tree_util.tree_map(_get_axis, self, is_leaf=_is_static_node)

    def add_subcondition(self, subcondition: "StateData"):

        new_subconditions = self.subconditions + (subcondition,)
        new_self = eqx.tree_at(lambda c: c.subconditions, self, new_subconditions)

        return new_self

    def insert_subcondition(self, subcondition: "StateData", index: int):
        new_subconditions = self.subconditions[:index] + (subcondition,) + self.subconditions[index:]

        return eqx.tree_at(lambda c: c.subconditions, self, new_subconditions)

    def replace_subcondition(self, subcondition: "StateData", index: int):
        new_subconditions = self.subconditions[:index] + (subcondition,) + self.subconditions[index + 1 :]

        return eqx.tree_at(lambda c: c.subconditions, self, new_subconditions)

    def __repr__(self):
        repr_str = self.tag + " - Subconditions: [" + ", ".join([sc.tag for sc in self.subconditions]) + "]"
        return repr_str
