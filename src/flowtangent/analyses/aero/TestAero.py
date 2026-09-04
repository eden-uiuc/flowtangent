# flowtangent/Framework/Analyses/Aerodynamics/Test_Aero.py
# (c) Copyright 2026 Aerospace Research Community LLC
#
# Created: Jan, 2026, Flowtangent Team

# ----------------------------------------------------------------------------------------------------------------------
# IMPORT
# ----------------------------------------------------------------------------------------------------------------------


from flowtangent.utils import field

from flowtangent.framework import Process, ProcessStep
from flowtangent.framework.methods.aero.Test_Aero import direct_aero


# 1. Define the builder function outside the class
def _build_test_aero_steps():
    """Builds and returns the default tuple of process steps."""
    return (ProcessStep(tag="Direct Aero Calculation", function=direct_aero),)


class TestAero(Process):
    tag: str = field("Aerodynamics", static=True)
    steps: tuple = field(_build_test_aero_steps)
