# RCAIDE/Framework/State.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Jul 2024, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

import unittest
from dataclasses import dataclass, field, make_dataclass
from typing import TypeVar

# package imports
import numpy as np

# RCAIDE imports
import RCAIDE.Library as rcl

ComponentType = TypeVar("ComponentType", bound="Component")

# ----------------------------------------------------------------------------------------------------------------------
# Components
# ----------------------------------------------------------------------------------------------------------------------


@dataclass(kw_only=True)
class VehicleEnvelope:
    # Attribute                 Type        Default Value
    ultimate_load:             float        = 0.0
    limit_load_factor:         float        = 0.0

# ----------------------------------------------------------------------------------------------------------------------
#  System
# ----------------------------------------------------------------------------------------------------------------------


@dataclass(kw_only=True)
class System(rcl.Component):

    name: str = 'System'

    energy: rcl.Components.Energy.EnergyNetwork = field(default_factory=lambda: rcl.Components.Energy.EnergyNetwork())

    configurations: dataclass = field(default_factory=lambda: make_dataclass('SystemConfigurations', []))

# ----------------------------------------------------------------------------------------------------------------------
# Unit Tests
# ----------------------------------------------------------------------------------------------------------------------

class TestSystem(unittest.TestCase):
    def setUp(self):
        self.system = System(name="TestSystem")
        self.system.mass_properties.total = 300

    def test_default_values(self):
        self.assertEqual(self.system.name, "TestSystem")
        self.assertEqual(self.system.subcomponents, [])

    def test_add_subcomponent(self):
        subcomponent = Component(name="Subcomponent", mass_properties=MassProperties(total=100))
        self.system.add_subcomponent(subcomponent)
        self.assertEqual(len(self.system.subcomponents), 1)
        self.assertEqual(self.system.subcomponents[0].name, "Subcomponent")
        self.assertEqual(self.system.mass_properties.subcomponent_total, 100)

    def test_sum_mass(self):
        self.system.add_subcomponent(Component(name="Sub1", mass_properties=MassProperties(total=100)))
        self.system.add_subcomponent(Component(name="Sub2", mass_properties=MassProperties(total=200)))
        self.system.sum_mass()
        self.assertEqual(self.system.mass_properties.subcomponent_total, 300)

    def test_sum_center_of_gravity(self):
        sub1 = Component(name="Sub1", mass_properties=MassProperties(total=100, center_of_gravity=np.array([1, 0, 0])))
        sub2 = Component(name="Sub2", mass_properties=MassProperties(total=200, center_of_gravity=np.array([0, 1, 0])))
        self.system.add_subcomponent(sub1)
        self.system.add_subcomponent(sub2)
        self.system.sum_center_of_gravity()
        np.testing.assert_array_almost_equal(self.system.mass_properties.center_of_gravity, np.array([1/3, 2/3, 0]))

    def test_add_subcomponent_type_error(self):
        with self.assertRaises(TypeError):
            self.system.add_subcomponent("Not a Component")



