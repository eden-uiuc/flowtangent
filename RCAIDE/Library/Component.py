# RCAIDE/Library/Compoments/Component.py
# (c) Copyright 2023 Aerospace Research Community LLC
# 
# Created: Jul 2024, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------  

import unittest

import chex
from dataclasses import field
from warnings import warn
from typing import TypeVar, List

# package imports 
import numpy as np

# RCAIDE imports
from RCAIDE.Library.Attributes.Materials import Solid, Aluminum

ComponentType = TypeVar("ComponentType", bound="Component")

# ----------------------------------------------------------------------------------------------------------------------
#  Component
# ----------------------------------------------------------------------------------------------------------------------         


@chex.dataclass(kw_only=True, slots=True)
class ComponentFineness:

    # Attribute     Type    Default Value
    effective:      float   = 1.0
    nose:           float   = 0.0
    tail:           float   = 0.0


@chex.dataclass(kw_only=True, slots=True)
class ComponentDimensions:

    # Attribute         Type    Default Value
    ordinal_direction:  bool    = False

    reference:          float   = 0.0
    total:              float   = 0.0
    maximum:            float   = 0.0
    effective:          float   = 0.0

    projected:          float   = 0.0
    front_projected:    float   = 0.0
    top_projected:      float   = 0.0
    side_projected:     float   = 0.0


@chex.dataclass(kw_only=True, slots=True)
class ComponentAreas:

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


@chex.dataclass(kw_only=True, slots=True)
class MaterialProperties:

    # Attribute                 Type        Default Value
    tensile_stress_carrier:     Solid   = field(default_factory=Aluminum)
    torsional_stress_carrier:   Solid   = field(default_factory=Aluminum)
    shear_stress_carrier:       Solid   = field(default_factory=Aluminum)


@chex.dataclass(kw_only=True, slots=True)
class MassProperties:

    # Attribute                         Type        Default Value
    total:                              float       = 0.0
    empty:                              float       = 0.0
    subcomponent_total:                 float       = 0.0

    volume:                             float       = 1.0
    density:                            float       = 0.0

    center_of_gravity:                  np.ndarray  = field(default_factory=lambda: np.zeros(3))
    moments_of_inertia:                 np.ndarray  = field(default_factory=lambda: np.zeros((3, 3)))
    subcomponent_moments_of_inertia:    np.ndarray  = field(default_factory=lambda: np.zeros((3, 3)))

    def __post_init__(self):
        if not np.any(self.density):
            try:
                self.density = self.total / self.volume
            except (ValueError, ZeroDivisionError) as e:
                warn("Error in calculating component density. Check mass and volume specifications.")


@chex.dataclass(kw_only=True, slots=True)
class Component:

    # ------------------------------------------------IDENTIFIERS-------------------------------------------------------

    tag:                   str                   = 'Component'
    segments:               List[ComponentType]   = field(default_factory=list)
    subcomponents:          List[ComponentType]   = field(default_factory=list)
    origin:                 np.ndarray            = field(default_factory=lambda: np.zeros(3))

    # ---------------------------------------------------AREAS----------------------------------------------------------
    areas:                  ComponentAreas        = field(default_factory=ComponentAreas)

    # -------------------------------------------------DIMENSIONS-------------------------------------------------------
    lengths:                ComponentDimensions   = field(default_factory=ComponentDimensions)
    widths:                 ComponentDimensions   = field(default_factory=ComponentDimensions)
    heights:                ComponentDimensions   = field(default_factory=ComponentDimensions)
    diameters:              ComponentDimensions   = field(default_factory=ComponentDimensions)

    # -----------------------------------------------MASS & MATERIALS---------------------------------------------------
    mass_properties:        MassProperties        = field(default_factory=MassProperties)
    material_properties:    MaterialProperties    = field(default_factory=MaterialProperties)

    def get_field_name(self):
        return self.tag.replace(' ', '_').lower()

    def add_segment(self, segment: ComponentType, index: int = -1):
        self.segments.insert(index, segment)

    def sum_mass(self):

        self.mass_properties.subcomponent_total = np.sum([c.mass_properties.total for c in self.subcomponents])

    def sum_moments_of_inertia(self):

        raise NotImplementedError("Subcomponent moments of inertia calculation is not implemented for the System class.")

    def sum_center_of_gravity(self):

        self.mass_properties.center_of_gravity = np.zeros(3)

        for sc in self.subcomponents:
            rel_origin = sc.origin - self.origin
            rel_cg = rel_origin + sc.mass_properties.center_of_gravity

            mass_fraction = sc.mass_properties.total / self.mass_properties.total
            weighted_cg = rel_cg * mass_fraction

            self.mass_properties.center_of_gravity += weighted_cg

    def add_subcomponent(self,
                         subcomponent: ComponentType,
                         sum_mass=False,
                         sum_center_of_gravity=False,
                         sum_moments_of_inertia=False
                         ):

        if isinstance(subcomponent, Component):
            setattr(self, subcomponent.get_field_name(), subcomponent)
            self.subcomponents.append(subcomponent)
        else:
            raise TypeError(f"Attempted to add a subcomponent to {self.tag} "
                            f"which was not a Component datastructure.")

        if sum_mass:
            self.sum_mass()
            if sum_center_of_gravity:
                self.sum_center_of_gravity()
            if sum_moments_of_inertia:
                self.sum_moments_of_inertia()

# Overwrite Chex dataclass's __getitem__ method to allow indexing of subcomponents.


def _component_getitem(self, item):
    if isinstance(item, slice | int):
        if hasattr(self, 'subcomponents'):
            return self.subcomponents[item]
        raise IndexError("Integer indexing is for subcomponents, but 'subcomponents' attribute not found.")
    elif isinstance(item, str):
        return self.__dict__[item]
    else:
        raise TypeError(f"Component indices must be integers or strings, not {type(item).__name__}")


Component.__getitem__ = _component_getitem

# ----------------------------------------------------------------------------------------------------------------------
# Unit Tests
# ----------------------------------------------------------------------------------------------------------------------

class TestComponent(unittest.TestCase):
    def setUp(self):
        self.component = Component(tag="TestComponent")

    def test_default_values(self):
        self.assertEqual(self.component.tag, "TestComponent")
        self.assertEqual(self.component.segments, [])
        np.testing.assert_array_equal(self.component.origin, np.zeros(3))

    def test_add_segment(self):
        segment = Component(tag="Segment")
        self.component.add_segment(segment)
        self.assertEqual(len(self.component.segments), 1)
        self.assertEqual(self.component.segments[0].tag, "Segment")


class TestMassProperties(unittest.TestCase):
    def test_density_calculation(self):
        mp = MassProperties(total=1000, volume=2)
        self.assertEqual(mp.density, 500)

    def test_density_warning(self):
        with self.assertWarns(UserWarning):
            mp = MassProperties(total=1000, volume=0)


class TestComponentAreas(unittest.TestCase):
    def test_default_values(self):
        areas = ComponentAreas()
        self.assertEqual(areas.reference, 0.0)
        self.assertEqual(areas.wetted, 0.0)
        self.assertEqual(areas.exposed, 0.0)


class TestComponentDimensions(unittest.TestCase):
    def test_default_values(self):
        dimensions = ComponentDimensions()
        self.assertFalse(dimensions.ordinal_direction)
        self.assertEqual(dimensions.reference, 0.0)
        self.assertEqual(dimensions.total, 0.0)


if __name__ == '__main__':
    unittest.main()