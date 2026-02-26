# RCAIDE/Framework/Analyses/Aerodynamics/Test_Aero.py
# (c) Copyright 2026 Aerospace Research Community LLC
#
# Created: Jan, 2026, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
# IMPORT 
# ----------------------------------------------------------------------------------------------------------------------

import equinox as eqx

import RCAIDE.Framework as rcf


# 1. Define the builder function outside the class
def _build_test_aero_steps():
    """Builds and returns the default tuple of process steps."""
    return (
        rcf.ProcessStep(
            tag="Direct Aero Calculation",
            # Remember to wrap in Equinox if ProcessStep was converted!
            function=rcf.Methods.Aerodynamics.Test_Aero.direct_aero 
        ),
    )


class TestAero(rcf.Process):
    # Shield the tag from the XLA compiler
    tag: str = eqx.field(static=True, default="Test Aerodynamic Analysis")

    # 2. Use the helper function as the default_factory!
    steps: tuple = eqx.field(default_factory=_build_test_aero_steps)