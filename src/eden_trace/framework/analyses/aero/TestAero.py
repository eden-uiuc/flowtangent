# Trace/Framework/Analyses/Aerodynamics/Test_Aero.py
# (c) Copyright 2026 Aerospace Research Community LLC
#
# Created: Jan, 2026, Trace Team

# ----------------------------------------------------------------------------------------------------------------------
# IMPORT
# ----------------------------------------------------------------------------------------------------------------------


from eden_trace.utils import init_field

from eden_trace.framework import Process, ProcessStep
from eden_trace.framework.methods.aero.Test_Aero import direct_aero


# 1. Define the builder function outside the class
def _build_test_aero_steps():
    """Builds and returns the default tuple of process steps."""
    return (ProcessStep(tag="Direct Aero Calculation", function=direct_aero),)


class TestAero(Process):
    tag: str = init_field("Aerodynamics", static=True)
    steps: tuple = init_field(_build_test_aero_steps)
