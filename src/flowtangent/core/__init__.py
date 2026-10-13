"""Core classes and datastructures for FlowTangent.

This package provides the Process, System, State, Settings, and related classes.
"""

from ._component import Component
from ._processes import Process, ProcessStep
from ._settings import Settings
from ._systems import System, Aircraft
from ._state import State

__all__ = [
    "Component",
    "System",
    "Aircraft",
    "ProcessStep",
    "Process",
    "Settings",
    "State"
]