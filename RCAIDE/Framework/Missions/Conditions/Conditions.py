# RCAIDE/Framework/Missions/Conditions/Conditions.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Jul 2024, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------



# package imports
import equinox as eqx

# ----------------------------------------------------------------------------------------------------------------------
#  Conditions
# ----------------------------------------------------------------------------------------------------------------------


class Conditions(eqx.Module):

    tag: str = eqx.field(static=True, default='Conditions')

    def _get_subconditions(self):
        subcons = []
        for field_name in self.__dataclass_fields__:
            val = getattr(self, field_name)
            if isinstance(val, Conditions):
                subcons.append(val)
        return tuple(subcons)

    def __getitem__(self, item):
        if isinstance(item, (int, slice)):
            return self._get_subconditions()[item]
        elif isinstance(item, str):
            attr_name = item.replace(' ', '_').lower()
            return getattr(self, attr_name)
        else:
            raise TypeError(f"Conditions indices must be slices, integers or strings, not {type(item).__name__}")
    
    def __setitem__(self, key, value):
        if isinstance(key, int):
            subconkeys = [k for k in vars(self) if isinstance(getattr(self, k), Conditions)]
            if key < len(subconkeys):
                setattr(self, subconkeys.index(key), value)
            else:
                setattr(self, str(key), value)
        else:
            setattr(self, key, value)
    
    def __iter__(self):
        return iter(self._get_subconditions())

    def set_condition(self, key, value):
        if isinstance(key, int):
            subconkeys = [
                k for k in self.__dataclass_fields__
                if isinstance(getattr(self, k), Conditions)
            ]
            if key < len(subconkeys):
                attr_name = subconkeys[key]
            else:
                attr_name = str(key)
        elif isinstance(key, str):
            attr_name = key.replace(' ', '_').lower()
        else:
            raise TypeError(f"Conditions must be set using, integers or strings, not {type(key).__name__}")
        
        return eqx.tree_at(lambda c: getattr(c, attr_name), self, value)