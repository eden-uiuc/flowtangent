# RCAIDE/Framework/Missions/Conditions/Conditions.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Jul 2024, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from dataclasses import fields

import equinox as eqx

# package imports
import jax
import jax.numpy as jnp

from RCAIDE.utils import init_field

# ----------------------------------------------------------------------------------------------------------------------
#  Conditions
# ----------------------------------------------------------------------------------------------------------------------


class Conditions(eqx.Module):

    tag: str = init_field('Conditions', static=True)

    subconditions: tuple = init_field(tuple)

    def __post_init__(self):
        subcons = tuple(
            getattr(self, f.name)
            for f in fields(self)
            if f.name != "subconditions" and isinstance(getattr(self, f.name), Conditions)
        )

        object.__setattr__(self, "subconditions", subcons)

    def __getitem__(self, item):
        if isinstance(item, (int, slice)):
            return self.subconditions[item]
        elif isinstance(item, str):
            attr_name = item.replace(' ', '_').lower()
            return getattr(self, attr_name)
        else:
            raise TypeError(f"Conditions indices must be slices, integers or strings, not {type(item).__name__}")

    def __iter__(self):
        return iter(self.subconditions)

    def expand_rows(self, n: int):

        CLASSES_TO_SKIP = (
            'AtmosphericBreakpoints',
            'SolverConditions'
        )

        def _expand(leaf):
            if isinstance(leaf, (jnp.ndarray)):

                # 1. Intercept the empty placeholders
                if leaf.size == 0:
                    if leaf.ndim == 1:
                        # Create a single zero, then broadcast it to (n, 1)
                        base = jnp.zeros((1, 1), dtype=leaf.dtype)
                        return jnp.broadcast_to(base, (n, 1))
                    elif leaf.ndim == 2:
                        base = jnp.zeros((1, leaf.shape[1]), dtype=leaf.dtype)
                        return jnp.broadcast_to(base, (n, leaf.shape[1]))

                # 2. Zero-copy expansion for actual data
                if leaf.ndim == 1:
                    # e.g., Shape (X,) -> Shape (n, X)
                    return jnp.broadcast_to(leaf, (n,) + leaf.shape)
                elif leaf.ndim == 2 and leaf.shape[0] == 1:
                    # e.g., Shape (1, X) -> Shape (n, X)
                    return jnp.broadcast_to(leaf, (n, leaf.shape[1]))

            return leaf

        def _is_static_node(node):
            return hasattr(node, '__class__') and node.__class__.__name__ in CLASSES_TO_SKIP

        return jax.tree_util.tree_map(_expand, self, is_leaf=_is_static_node)

    def add_subcondition(self, subcondition: "Conditions"):

        new_subconditions = self.subconditions + (subcondition,)
        new_self = eqx.tree_at(lambda c: c.subconditions, self, new_subconditions)

        return new_self

    def insert_subcondition(self, subcondition: "Conditions", index: int):
        new_subconditions = self.subconditions[:index] + (subcondition,) + self.subconditions[index:]

        return eqx.tree_at(lambda c: c.subconditions, self, new_subconditions)

    def replace_subcondition(self, subcondition: "Conditions", index: int):
        new_subconditions = self.subconditions[:index] + (subcondition,) + self.subconditions[index + 1:]

        return eqx.tree_at(lambda c: c.subconditions, self, new_subconditions)

    def __repr__(self):
        repr_str = self.tag + " - Subconditions: [" + ', '.join([sc.tag for sc in self.subconditions])+"]"
        return repr_str
