import pandas as pd

import equinox as eqx

from eden_trace.utils import configure_environment

from eden_trace.library.components.energy.jets import data as jet_data
from eden_trace.library.components.energy.networks import TurbofanNetwork

from eden_trace.framework import State, Settings, Aircraft
from eden_trace.framework.analyses.energy.jets import design_turbofan_mp, JetSettings

if __name__ == "__main__":
    
    records = []

    e_st = State()
    sys = Aircraft(subcomponents=(TurbofanNetwork(),))
    setts = Settings(verbose=True, DEBUG_MODE=True)
    configure_environment(setts)

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
        e_setts = eqx.tree_at(lambda s: s.analysis.energy, setts, JetSettings("Imperial", True))
        
        if not engine:
            print(f"Failed to import '{e_name}'. Skipping...")
        else:
            print("="*70)
            print(f"Designing {e_name.replace('_','-')}")
            print("-"*70)
            e_sys = eqx.tree_at(lambda n: n.energy.line.engine, sys, engine)
            d_st, d_sys, d_setts = design_turbofan_mp(e_st, e_sys, e_setts)

            # thrust = d_st.energy.nodes['network.line.engine'].force.thrust.item()
            # fan_MFR = d_st.energy.nodes['network.line.engine.fan'].flow.mass_flow_rate.item()
            # lpc_MFR = d_st.energy.nodes['network.line.engine.lpc'].flow.mass_flow_rate.item()

            # print(f"Fan MFR: {fan_MFR:.2e}")
            # print(f"LPC MFR: {lpc_MFR:.2e}")

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


            

