# RCAIDE/Library/Methods/Mass/Propulsion/Jet_Mass_from_SLS.py
# (c) Copyright 2025 Aerospace Research Community LLC#
# Created:  May 2025, J. Smart
# Modified: 
# -------------------------------------------------------------------------------
#  Imports
# -------------------------------------------------------------------------------

# RCAIDE Imports

from RCAIDE.Library.Components.Energy.Propulsors import TurbofanEngine
import RCAIDE.Framework as rcf

from RCAIDE.Library.Methods.Energy.Converters.Turbofans import func_sea_level_static_thrust

# -------------------------------------------------------------------------------
#  Functional/Library Version
# -------------------------------------------------------------------------------


def func_tf_mass_from_SLS(
    sls_thrust: float
):

    t_lbf = sls_thrust * 0.224809  # Convert to lbf
    mass = (0.4054 * t_lbf ** 0.9255) * 0.453592

    return mass


# -------------------------------------------------------------------------------
#  Stateful/Framework Version
# -------------------------------------------------------------------------------

def tf_mass_from_SLS(
    state: "rcf.State",
    system: "rcf.Aircraft",
    settings: "rcf.Settings",
    ):
    """
    Framework version of Jet_Mass_from_SLS. Assumes a turbofan engine.
    
    See Also
    --------
    func_Jet_Mass_from_SLS: 
        Functional implementation which this method calls.
    """



    for line in system.energy.lines:
        for jet in line.converters:

            jet: TurbofanEngine

            if jet.design_parameters.SLS_thrust == 0.:
                try:
                    F = func_sea_level_static_thrust(
                        F_ref=jet.design_parameters.total_thrust
                        T_ref=jet.reference_temperature,
                        T_t_ref=jet.reference_total_temperature,
                        P_ref=jet.reference_pressure,
                        P_t_ref=jet.reference_total_pressure,
                        delta_SFC=,
                        f,
                        v_fan_nozzle,
                        AR_fan_nozzle,
                        P_fan_nozzle,
                        v_core_nozzle,
                        AR_core_nozzle,
                        P_core_nozzle,
                        alpha,
                        throttle
                    ):

                jet.design_parameters.SLS_thrust = F


            sls_thrust = jet.design_parameters.SLS_thrust

            jet_mass = func_tf_mass_from_SLS(sls_thrust)
            jet.mass_properties.total = jet_mass

    # TODO: Unpack results

    return state, system, settings
