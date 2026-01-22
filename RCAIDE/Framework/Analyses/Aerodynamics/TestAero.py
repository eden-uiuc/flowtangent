# RCAIDE/Framework/Analyses/Aerodynamics/Test_Aero.py
# (c) Copyright 2026 Aerospace Research Community LLC
#
# Created: Jan, 2026, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
# IMPORT 
# ----------------------------------------------------------------------------------------------------------------------

import chex

from dataclasses import field, make_dataclass

import RCAIDE.Framework as rcf
import RCAIDE.Library as rcl

# ----------------------------------------------------------------------------------------------------------------------
# Inverse_Range
# ----------------------------------------------------------------------------------------------------------------------


@chex.dataclass(kw_only=True)
class TestAero(rcf.Process):

    tag = "Test Aerodynamic Analysis"

    steps = [
        rcf.Methods.Aerodynamics.Test_Aero.aero_from_mass
    ]