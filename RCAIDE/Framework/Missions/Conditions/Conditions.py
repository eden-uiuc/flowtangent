# RCAIDE/Framework/Missions/Conditions/Conditions.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Jul 2024, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------



# package imports
import jax
import numpy as np
import equinox as eqx
import jax.numpy as jnp

# ----------------------------------------------------------------------------------------------------------------------
#  Conditions
# ----------------------------------------------------------------------------------------------------------------------


class Conditions(eqx.Module):

    tag: str = eqx.field(static=True, default='Conditions')

    subconditions: tuple = eqx.field(default_factory=tuple)

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

        def _expand(leaf):
            if isinstance(leaf, (jnp.ndarray, np.ndarray)):
                
                # 1. Intercept the empty placeholders we created to avoid 'None'
                if leaf.size == 0:
                    if leaf.ndim == 1:
                        # jnp.empty(0) -> shape (n, 1)
                        return jnp.zeros((n, 1), dtype=leaf.dtype)
                    elif leaf.ndim == 2:
                        # jnp.empty((0, 3)) -> shape (n, 3)
                        return jnp.zeros((n, leaf.shape[1]), dtype=leaf.dtype)
                
                # 2. Standard expansion for actual data
                if leaf.ndim == 1:
                    return jnp.tile(leaf, (n, 1))
                elif leaf.ndim == 2 and leaf.shape[0] == 1:
                    return jnp.repeat(leaf, n, axis=0)
                    
            return leaf
        
        return jax.tree_util.tree_map(_expand, self)
    
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