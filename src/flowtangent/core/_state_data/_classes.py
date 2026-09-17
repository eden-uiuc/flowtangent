# flowtangent/Framework/Missions/Conditions/Conditions.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Jul 2024, Flowtangent Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from dataclasses import fields
from typing import Self, Sequence

# package imports
import jax
import jax.numpy as jnp

from ...utils import Module
from ...utils.base import StateDataMeta

# ----------------------------------------------------------------------------------------------------------------------
#  Conditions
# ----------------------------------------------------------------------------------------------------------------------

STATIC_DATA = ("AtmosphericBreakpoints",)


def _is_static_node(node):
    return hasattr(node, "__class__") and node.__class__.__name__ in STATIC_DATA


class StateData(Module, metaclass=StateDataMeta):
    @property
    def substates(self) -> tuple:
        return tuple(
            getattr(self, f.name)
            for f in fields(self)
            if f.name != "substates" and isinstance(getattr(self, f.name), StateData)
        )

    def __getitem__(self, item):
        if isinstance(item, (int, slice)):
            return self.substates[item]
        elif isinstance(item, str):
            attr_name = item.replace(" ", "_").lower()
            return getattr(self, attr_name)
        else:
            raise TypeError(f"Conditions indices must be slices, integers or strings, not {type(item).__name__}")

    def __iter__(self):
        return iter(self.substates)

    def expand_time(self, N: int = 1):

        def _expand(leaf):
            if isinstance(leaf, (jax.Array)):
                if leaf.size == 0:
                    trailing_dims = leaf.shape[1:] if leaf.ndim > 0 else ()
                    return jnp.zeros((N,) + trailing_dims, dtype=leaf.dtype)
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
            if isinstance(leaf, jax.Array):
                # # Intercept the empty placeholders
                if leaf.size == 0:
                    trailing_dims = (1,) if leaf.ndim == 1 else leaf.shape[1:]
                    return jnp.zeros((batch_size,) + trailing_dims, dtype=leaf.dtype)
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
            if isinstance(first_leaf, jax.Array):
                return jnp.concatenate(leaves, axis=0)
            return first_leaf

        return jax.tree_util.tree_map(_concat, *states, is_leaf=_is_static_node)

    def truncate(self, size: int):
        def _trunc(leaf):
            # 1. Ignore static classes
            if _is_static_node(leaf):
                return leaf

            # 2. Slice the batch dimension (axis 0) of the array
            if isinstance(leaf, jax.Array):
                return leaf[:size]

            return leaf

        return jax.tree_util.tree_map(_trunc, self, is_leaf=_is_static_node)

    def get_vmap_axes(self):
        def _get_axis(leaf):
            if _is_static_node(leaf):
                # JAX allows prefix-trees for in_axes. Returning None for the whole
                # object tells JAX to broadcast everything inside this class.
                return None
            if isinstance(leaf, jax.Array):
                return 0
            return None

        return jax.tree_util.tree_map(_get_axis, self, is_leaf=_is_static_node)

    # def add_substate(self, substate: "StateData"):

    #     new_substates = self.substates + (substate,)
    #     new_self = update(self, "substates", new_substates)

    #     return new_self

    # def insert_substate(self, substate: "StateData", index: int):
    #     new_substates = self.substates[:index] + (substate,) + self.substates[index:]

    #     return update(self, "substates", new_substates)

    # def replace_substate(self, substate: "StateData", index: int):
    #     new_substates = self.substates[:index] + (substate,) + self.substates[index + 1 :]

    #     return update(self, "substates", new_substates)

    def __repr__(self):
        repr_str = str(self.name) + " - Substates: [" + ", ".join([sc.name for sc in self.substates]) + "]"
        return repr_str
