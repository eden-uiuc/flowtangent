# RCAIDE/Framework/Missions/Converge/fsolve.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Aug, 2024, RCAIDE Team

# ----------------------------------------------------------------------------------------------------------------------
# IMPORT 
# ----------------------------------------------------------------------------------------------------------------------

# package imports
import equinox as eqx
import jax.numpy as jnp


# RCAIDE imports
from RCAIDE.Framework import State, System, Settings

# ----------------------------------------------------------------------------------------------------------------------
# fsolve Convergence
# ----------------------------------------------------------------------------------------------------------------------


def fsolve_results_parser(
        fsolve_result: tuple,
        state: State,
        system: System,
        settings: Settings,
) -> tuple[State, System, Settings]:

        unknowns:       jnp.ndarray     = jnp.array(fsolve_result[0])
        infodict:       dict            = fsolve_result[1]
        ier:            int             = fsolve_result[2]
        mesg:           str             = fsolve_result[3]

        current_state = state

        if ier != 1:
            print("Segment Convergence Failed:", mesg)
            unconverged_numerics = eqx.tree_at(lambda n:n.converged, state.numerics, False)
            current_state = eqx.tree_at(lambda s: s.numerics, state, unconverged_numerics)
        else:
            print("Segment Converged.")
            print("Number of function evaluations:", infodict['nfev'])
            converged_numerics = eqx.tree_at(lambda n:n.converged, state.numerics, True)
            current_state = eqx.tree_at(lambda s: s.numerics, state, converged_numerics)
        
        current_state = eqx.tree_at(lambda s: s.unknowns, current_state, unknowns)
        current_state = current_state.unpack_unknowns()
        
        return state, system, settings