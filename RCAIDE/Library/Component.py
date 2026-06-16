# RCAIDE/Library/Compoments/Component.py
# (c) Copyright 2023 Aerospace Research Community LLC
# 
# Created: Jul 2024, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------  


from __future__ import annotations
from warnings import warn
from typing import Any

# package imports 
import equinox as eqx
import jax.numpy as jnp

# RCAIDE imports
from RCAIDE.utils import empty_array, init_field
from RCAIDE.Library.Attributes.Materials import Solid, Aluminum

# ----------------------------------------------------------------------------------------------------------------------
#  Component
# ----------------------------------------------------------------------------------------------------------------------         


class ComponentFineness(eqx.Module):

    # Attribute     Type    Default Value
    effective:      float   = 1.0
    nose:           float   = 0.0
    tail:           float   = 0.0
    def __repr__(self):
        return f"Eff.: {self.effective}"


class ComponentDimensions(eqx.Module):

    # Attribute         Type    Default Value
    ordinal_direction:  bool    = init_field(False, static=True)

    reference:          float   = 0.0
    total:              float   = 0.0
    maximum:            float   = 0.0
    effective:          float   = 0.0

    projected:          float   = 0.0
    front_projected:    float   = 0.0
    top_projected:      float   = 0.0
    side_projected:     float   = 0.0

    def __repr__(self):
        return ""


class ComponentAreas(eqx.Module):

    # Attribute         Type    Default Value
    reference:          float   = 0.0
    total:              float   = 0.0
    maximum:            float   = 0.0
    effective:          float   = 0.0

    inflow:             float   = 0.0
    outflow:            float   = 0.0

    inlet:              float   = 0.0
    exit:               float   = 0.0

    projected:          float   = 0.0
    front_projected:    float   = 0.0
    top_projected:      float   = 0.0
    side_projected:     float   = 0.0

    wetted:             float   = 0.0
    exposed:            float   = 0.0

    def __repr__(self):
        return ""


class MaterialProperties(eqx.Module):

    # Attribute                 Type        Default Value
    tensile_stress_carrier:     Solid   = init_field(Aluminum)
    torsional_stress_carrier:   Solid   = init_field(Aluminum)
    shear_stress_carrier:       Solid   = init_field(Aluminum)

    def __repr__(self):
        return ""

class MassProperties(eqx.Module):

    # Attribute                         Type        Default Value
    total:                              float       = 0.0
    empty:                              float       = 0.0
    subcomponent_total:                 float       = 0.0

    volume:                             float       = 1.0
    density:                            float       = 0.0

    center_of_gravity:                  jnp.ndarray  = empty_array(3)
    moments_of_inertia:                 jnp.ndarray  = empty_array((3, 3))
    subcomponent_moments_of_inertia:    jnp.ndarray  = empty_array((3, 3))


    def __repr__(self):
        return ""


class Component(eqx.Module):


    tag:                    str                   = init_field('Component', static=True)
    is_control_component:   bool                  = init_field(False, static=True)

    segments:               tuple[Component, ...] = init_field(tuple)
    subcomponents:          tuple[Component, ...] = init_field(tuple)
    origin:                 jnp.ndarray           = empty_array(3)

    # ---------------------------------------------------AREAS----------------------------------------------------------
    areas:                  ComponentAreas        = init_field(ComponentAreas)

    # -------------------------------------------------DIMENSIONS-------------------------------------------------------
    lengths:                ComponentDimensions   = init_field(ComponentDimensions)
    widths:                 ComponentDimensions   = init_field(ComponentDimensions)
    heights:                ComponentDimensions   = init_field(ComponentDimensions)
    diameters:              ComponentDimensions   = init_field(ComponentDimensions)

    # -----------------------------------------------MASS & MATERIALS---------------------------------------------------
    mass_properties:        MassProperties        = init_field(MassProperties)
    material_properties:    MaterialProperties    = init_field(MaterialProperties)

    _bookkeeping:           dict[str, Any] = init_field(dict, static=True)

    
    def __repr__(self):
        repr_str = self.tag + " - Subcomponents: [" + ', '.join([sc.tag for sc in self.subcomponents])+"]"
        return repr_str
    
    def __getitem__(self, item):
        if isinstance(item, (slice, int)):
            return self.subcomponents[item]
        elif isinstance(item, str):
            return getattr(self, item.replace(' ', '_').lower())
        else:
            raise TypeError(f"Indices must be int, slice, or str.")
    
    def __getattr__(self, item: str):
        if item.startswith("__") and item.endswith("__"):
            raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{item}'")

        if hasattr(self, '_bookkeeping') and item in self._bookkeeping:
            target_class = self._bookkeeping[item]
            filtered_subs = tuple(
                c for c in self.subcomponents if isinstance(c, target_class)
            )
            return Component(tag=item.replace('_',' ').title(), subcomponents=filtered_subs)
        for sc in self.subcomponents:
            if hasattr(sc, 'get_field_name') and sc.get_field_name() == item:
                return sc
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{item}'")

    def __bool__(self):
        return True

    def __len__(self):
        return len(self.subcomponents)

    def __iter__(self):
        return iter(self.subcomponents)
    
    def __contains__(self, item):
        if isinstance(item, str):
            return any(sc.get_field_name() == item for sc in self.subcomponents)
        
        return item in self.subcomponents

    def get_field_name(self):
        actual_tag = self.tag
        
        if not isinstance(actual_tag, str):
            if hasattr(actual_tag, 'value'):
                actual_tag = actual_tag.value
            else:
                raise AttributeError(f"Unable to resolve field name for {self}.")
        
        return actual_tag.replace(' ', '_').lower()

    def add_segment(self, segment: "Component", index: int | None = None):
        if index is None:
            # Append
            new_segments = self.segments + (segment,)
        else:
            # Insert at specific index
            new_segments = self.segments[:index] + (segment,) + self.segments[index:]

        # Functionally replace and return the new Component
        return eqx.tree_at(lambda c: c.segments, self, new_segments)
    
    def insert_segment(self, segment: "Component", index: int):
        new_segments = self.segments[:index] + (segment,) + self.segments[index:]
        
        return eqx.tree_at(lambda c: c.segments, self, new_segments)

    def replace_segment(self, segment: "Component", index: int):
        new_segments = self.segments[:index] + (segment,) + self.segments[index + 1:]
        
        return eqx.tree_at(lambda c: c.segments, self, new_segments)

    def add_subcomponent(self, subcomponent: "Component"):

        new_subcomponents = self.subcomponents + (subcomponent,)
        new_self = eqx.tree_at(lambda c: c.subcomponents, self, new_subcomponents)
    
        return new_self
    
    def insert_subcomponent(self, subcomponent: "Component", index: int):
        new_subcomponents = self.subcomponents[:index] + (subcomponent,) + self.subcomponents[index:]
        
        return eqx.tree_at(lambda c: c.subcomponents, self, new_subcomponents)

    def replace_subcomponent(self, subcomponent: "Component", index: int):
        new_subcomponents = self.subcomponents[:index] + (subcomponent,) + self.subcomponents[index + 1:]
        
        return eqx.tree_at(lambda c: c.subcomponents, self, new_subcomponents)

    def remove_subcomponent(self, index: int):
        new_subcomponents = self.subcomponents[:index] + self.subcomponents[index + 1:]

        return eqx.tree_at(lambda c: c.subcomponents, self, new_subcomponents)


class ControlComponent(Component):

    is_control_component:   bool = init_field(True, static=True)
    control_path:           tuple[str, ...] |None   = init_field(tuple, static=True)
    control_path_indices:   tuple | None            = init_field(lambda: (slice(None), 0), static=True)