# RCAIDE/Framework/Missions/Converge/fsolve.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Aug, 2024, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
# IMPORT 
# ----------------------------------------------------------------------------------------------------------------------

from typing import Tuple

# package imports
import numpy as np


# RCAIDE imports
import RCAIDE.Framework as rcf

# ----------------------------------------------------------------------------------------------------------------------
# fsolve Convergence
# ----------------------------------------------------------------------------------------------------------------------


def fsolve_results_parser(
        fsolve_result: Tuple,
        state: "rcf.State",
        system: "rcf.System",
        settings: "rcf.Settings",
) -> ("rcf.State", "rcf.System", "rcf.Settings"):

        unknowns:       np.ndarray      = fsolve_result[0]
        infodict:       dict            = fsolve_result[1]
        ier:            int             = fsolve_result[2]
        mesg:           str             = fsolve_result[3]

        if ier != 1:
            print("Segment Convergence Failed:", mesg)
            state.numerics.converged = False
        else:
            print("Segment Converged.")
            print("Number of function evaluations:", infodict['nfev'])
            state.unknowns.unpack_array(unknowns)
            state.numerics.converged = True
        
        return state, system, settings

def fsolve_update_kwargs(
        fsolve_kwargs: dict,
        state: "rcf.State",
        system: "rcf.System",
        settings: "rcf.Settings",
):

        self.root_finder_kwargs = {
                'func': self._get_residuals,
                'x0': state.unknowns.pack_array(),
                'args': (state, system, settings),
                'xtol': state.numerics.solution_tolerance,
                'maxfev': state.numerics.max_evaluations,
                'epsfcn': state.numerics.step_size,
                'full_output': True
            }

        fsolve_kwargs['x0'] = state.unknowns
        fsolve_kwargs['args'] = (state, system, settings)

        return fsolve_kwargs