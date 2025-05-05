# RCAIDE/Library/Components/__init__.py
# (c) Copyright 2023 Aerospace Research Community LLC

"""RCAIDE Package Setup
"""

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from os import path
from pathlib import Path
Airfoil_Data = Path(path.join(path.dirname(__file__), 'Airfoil_Data'))

from .Fuselages     import Fuselage, FuselageSegment, BWBFuselage
from .Airfoils      import Airfoil
from .Nacelles      import Nacelle
from .Landing_Gear  import LandingGear
from .Wings         import Wing, WingSegment, WingControlSurface

from . import Energy

# from . import Booms
# from . import Payloads
# from . import Systems
# from . import Wings
