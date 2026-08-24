# import os
# node_0_cores = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}

# try:
#     # 0 means "current process"
#     os.sched_setaffinity(0, node_0_cores)
#     print(f"NUMA Lock Engaged. Running on cores: {os.sched_getaffinity(0)}")
# except AttributeError:
#     pass # Fails gracefully if you ever run this on Windows/Mac

import pandas as pd

import equinox as eqx

from eden_trace.library.components.energy.jets import data as jet_data
from eden_trace.library.components.energy.networks import TurbofanNetwork

from eden_trace.framework import State, Settings, Aircraft
from eden_trace.framework.analyses.energy.jets import DesignTurbofan, JetSettings



if __name__ == "__main__":
    
    records = []

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
            print("="*70)
            print(f"Designing {e_name.replace('_','-')}")
            print("-"*70)
            e_sys = eqx.tree_at(lambda n: n.energy.line.engine, sys, engine)
            d_st, d_sys, d_setts = DesignTurbofan(e_st, e_sys, e_setts)

            thrust = d_st.energy.nodes['network.line.engine'].outputs.force.thrust.item()
            fan_MFR = d_st.energy.nodes['network.line.engine.fan'].outputs.flow.mass_flow_rate.item()
            lpc_MFR = d_st.energy.nodes['network.line.engine.lpc'].outputs.flow.mass_flow_rate.item()

            print(f"Fan MFR: {fan_MFR:.2e}")
            print(f"LPC MFR: {lpc_MFR:.2e}")

            MFR = d_st.energy.mass_flow_rate.item()
            dMFR = d_sys.energy.line.engine.design_parameters.mass_flow_rate
            eMFR = (MFR - dMFR)/dMFR

            records.append({
                "MFR": MFR,
                "Tgt. MFR": dMFR,
                "MFR Err.": eMFR,
            })
    
    df = pd.DataFrame(records)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.float_format', '{:.2e}'.format)

    print("\n" + "="*80)
    print(f" Civil HBTF Design Validation")
    print("-"*80)
    print(df.to_string(index=False))


            

