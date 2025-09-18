# RCAIDE/Framework/State.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Jul 2024, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

import chex
from dataclasses import field
from typing import Self

# package imports
import numpy as np

# RCAIDE imports
from RCAIDE.Framework.Missions.Conditions import (
    Conditions, Numerics, FrameConditions, FreestreamConditions, MassConditions, EnergyNetworkConditions,
    AerodynamicsConditions, ControlsConditions)

# ----------------------------------------------------------------------------------------------------------------------
#  State
# ----------------------------------------------------------------------------------------------------------------------


@chex.dataclass(kw_only=True)
class State(Conditions):

    # Attribute         Type                        Default Value
    name:               str                         = 'State'
    initials:           Self                        = None
    numerics:           Numerics                    = field(default_factory=Numerics)

    frames:             FrameConditions             = field(default_factory=FrameConditions)
    freestream:         FreestreamConditions        = field(default_factory=FreestreamConditions)

    mass:               MassConditions              = field(default_factory=MassConditions)
    energy:             EnergyNetworkConditions     = field(default_factory=EnergyNetworkConditions)

    aerodynamics:       AerodynamicsConditions      = field(default_factory=AerodynamicsConditions)
    controls:           ControlsConditions          = field(default_factory=ControlsConditions)

    unknowns:           Conditions                  = field(default_factory=lambda: Conditions(name='Unknowns'))
    residuals:          Conditions                  = field(default_factory=lambda: Conditions(name='Residuals'))
    objective:          Conditions                  = field(default_factory=lambda: Conditions(name='Objective'))
