# RCAIDE/Library/Compoments/Component.py
# (c) Copyright 2023 Aerospace Research Community LLC
# 
# Created: Jul 2024, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------  

import unittest

from dataclasses import dataclass, field
from warnings import warn
from typing import TypeVar, List

# package imports 
import numpy as np

ComponentType = TypeVar("ComponentType", bound="Component")

# ----------------------------------------------------------------------------------------------------------------------
#  Component
# ----------------------------------------------------------------------------------------------------------------------         


@dataclass(kw_only=True)
class ComponentRatios:

    # Attribute     Type    Default Value
    effective:      float   = 1.0
    nose:           float   = 0.0
    tail:           float   = 0.0


@dataclass(kw_only=True)
class ComponentDimensions:

    # Attribute         Type    Default Value
    ordinal_direction:  bool    = field(default=False)

    reference:          float   = field(default=0.0)
    total:              float   = field(default=0.0)
    maximum:            float   = field(default=0.0)
    effective:          float   = field(default=0.0)

    projected:          float   = field(default=0.0)
    front_projected:    float   = field(default=0.0)
    top_projected:      float   = field(default=0.0)
    side_projected:     float   = field(default=0.0)


@dataclass(kw_only=True)
class ComponentAreas:

    # Attribute         Type    Default Value
    reference:          float   = field(default=0.0)
    total:              float   = field(default=0.0)
    maximum:            float   = field(default=0.0)
    effective:          float   = field(default=0.0)

    inflow:             float   = field(default=0.0)
    outflow:            float   = field(default=0.0)
    exit:               float   = field(default=0.0)

    projected:          float   = field(default=0.0)
    front_projected:    float   = field(default=0.0)
    top_projected:      float   = field(default=0.0)
    side_projected:     float   = field(default=0.0)

    wetted:             float   = field(default=0.0)
    exposed:            float   = field(default=0.0)


@dataclass(kw_only=True)
class MaterialProperties:

    # Attribute                 Type        Default Value
    tensile_stress_carrier:     dataclass   = field(default_factory=dataclass)
    torsional_stress_carrier:   dataclass   = field(default_factory=dataclass)
    shear_stress_carrier:       dataclass   = field(default_factory=dataclass)


@dataclass(kw_only=True)
class MassProperties:

    # Attribute                         Type        Default Value
    total:                              float       = field(default=0.0)
    subcomponent_total:                 float       = field(default=0.0)

    volume:                             float       = field(default=1.0)
    density:                            float       = field(default=0.0)

    center_of_gravity:                  np.ndarray  = field(default_factory=lambda: np.zeros(3))
    moments_of_inertia:                 np.ndarray  = field(default_factory=lambda: np.zeros((3, 3)))
    subcomponent_moments_of_inertia:    np.ndarray  = field(default_factory=lambda: np.zeros((3, 3)))

    def __post_init__(self):
        if not np.any(self.density):
            try:
                self.density = self.total / self.volume
            except (ValueError, ZeroDivisionError) as e:
                warn("Error in calculating component density. Check mass and volume specifications.")


@dataclass(kw_only=True)
class Component:

    # ------------------------------------------------IDENTIFIERS-------------------------------------------------------

    name:                   str                   = 'Component'
    segments:               List[ComponentType]   = field(default_factory=list)
    origin:                 np.ndarray            = field(default_factory=lambda: np.zeros(3))

    # ---------------------------------------------------AREAS----------------------------------------------------------
    areas:                  ComponentAreas        = field(default_factory=ComponentAreas)

    # -------------------------------------------------DIMENSIONS-------------------------------------------------------
    lengths:                ComponentDimensions   = field(default_factory=ComponentDimensions)
    widths:                 ComponentDimensions   = field(default_factory=ComponentDimensions)
    heights:                ComponentDimensions   = field(default_factory=ComponentDimensions)

    # -----------------------------------------------MASS & MATERIALS---------------------------------------------------
    mass_properties:        MassProperties        = field(default_factory=MassProperties)
    material_properties:    MaterialProperties    = field(default_factory=MaterialProperties)

    def add_segment(self, segment: ComponentType, index: int = -1):
        self.segments.insert(index, segment)


# ----------------------------------------------------------------------------------------------------------------------
# Unit Tests
# ----------------------------------------------------------------------------------------------------------------------

class TestComponent(unittest.TestCase):
    def setUp(self):
        self.component = Component(name="TestComponent")

    def test_default_values(self):
        self.assertEqual(self.component.name, "TestComponent")
        self.assertEqual(self.component.segments, [])
        np.testing.assert_array_equal(self.component.origin, np.zeros(3))

    def test_add_segment(self):
        segment = Component(name="Segment")
        self.component.add_segment(segment)
        self.assertEqual(len(self.component.segments), 1)
        self.assertEqual(self.component.segments[0].name, "Segment")


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