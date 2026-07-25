import pandas as pd

import equinox as eqx

from eden_trace.library.components.energy.jets import data as jet_data
from eden_trace.library.components.energy.networks import TurbofanNetwork

from eden_trace.framework import State, Settings, Aircraft
from eden_trace.framework.analyses.energy.jets import DesignTurbofan, JetSettings

if __name__ == "__main__":
    
    results = pd.DataFrame(columns=[
        "Thrust",
        "Thrust Rel. Error",
        "TSFC"
        "TSFC Rel. Error"
    ])

    e_st = State()
    sys = Aircraft(subcomponents=(TurbofanNetwork(),))
    setts = Settings(verbose=True, DEBUG_MODE=True)
    e_setts = eqx.tree_at(lambda s: s.analysis.energy, setts, JetSettings("Imperial", True))

    for e_name in [
        "CF6_50_C2",
        # "CF6_80_C2",
        # "GE90_B4",
        # "JT8D_15A",
        # "JT9D_59A",
        # "PW2037",
        # "PW4052",
        # "CFM56_3",
        # "CFM56_5C",
        # "RB211_524B",
        # "RB211_535E",
        # "RB211_882",
        # "V2528_D5",
        # "TFE731_5",
        # "PW300",
        # "FJ44",
    ]:
        engine = getattr(jet_data, e_name, False)
        
        if not engine:
            print(f"Failed to import '{e_name}'. Skipping...")
        else:
            print(f"Designing '{e_name}'...")
            e_sys = eqx.tree_at(lambda n: n.energy.line.engine, sys, engine)
            d_st, d_sys, d_setts = DesignTurbofan(e_st, e_sys, e_setts)
