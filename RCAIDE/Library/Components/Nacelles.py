# $NAME.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: Apr 2025, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from dataclasses import dataclass, field

# package imports
import numpy as np

# RCAIDE imports
import RCAIDE.Library as rcl

# ----------------------------------------------------------------------------------------------------------------------
#  Nacelle
# ----------------------------------------------------------------------------------------------------------------------


@dataclass(kw_only=True)
class Nacelle(rcl.Component):

    name:                       str     = 'Nacelle'

    flow_through:               bool    = False
    has_pylon:                  bool    = True
    fuselage_integrated:        bool    = False

    aerodynamic_center:         np.ndarray              = field(default_factory=np.zeros(3))
    orientation_euler_angles:   np.ndarray              = field(default_factory=np.zeros(3))

    airfoil:                    rcl.Components.Airfoil  = field(default_factory=
                                                                rcl.Components.Airfoil.NACA_4_Series('2410'))
    cowling_airfoil_angle:      float                   = 0.0

    diameters:                  rcl.ComponentDimensions = field(default_factory=rcl.ComponentDimensions)

    def __post_init__(self):

        self.diameters.inlet = 0.0
