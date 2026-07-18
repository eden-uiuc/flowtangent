import json

import jax.numpy as jnp
import equinox as eqx
import numpy as np
import pandas as pd

from pathlib import Path

from eden_trace.utils import save_data, load_data, format_array

from eden_trace.library import units
from eden_trace.library.components.energy.networks import TurbofanNetwork, TurbofanDesign
from eden_trace.library.components.energy.jets import TurbofanEngine, JetDesign
from eden_trace.library.components.energy.lines import TurbofanLine

from eden_trace.framework import State, Aircraft, Settings
from eden_trace.framework.analyses.energy.turbojets import DesignTurbofan

test_dir = Path("./tests/PyCycle/PyCycle_Examples/HBTF")

def system_setup():

    engine = TurbofanEngine(design_parameters=JetDesign(thrust=5900 * units.lbf, bypass_ratio=5.105))
    
    inlet_design = eqx.tree_at(lambda i:
        (
            i.design_parameters.exit_mach_number,
            i.design_parameters.pressure_recovery,
        ),
            engine.inlet,
        (
            0.751,
            0.999,
        )
    )

    fan_design = eqx.tree_at(lambda f:
        (
            f.design_parameters.pressure_ratio,
            f.design_parameters.exit_mach_number,
            f.efficiencies.flow,
        ),
        engine.fan,
        (
            1.685,
            0.4578,
            0.8948
        )
    )
    
    lpc_design = eqx.tree_at(lambda c:
        (
            c.design_parameters.pressure_ratio,
            c.design_parameters.rotation_speed,
            c.design_parameters.exit_mach_number,
            c.efficiencies.flow,
        ),
        engine.lpc,
        (
            1.935,
            5000. * units.rev/units.mins,
            0.3059,
            0.9243
        )
    )

    hpc_design = eqx.tree_at(lambda c:
        (
            c.design_parameters.pressure_ratio,
            c.design_parameters.rotation_speed,
            c.design_parameters.exit_mach_number,
            c.efficiencies.flow,
        ),
        engine.hpc,
        (
            9.369,
            15000. * units.rev/units.mins,
            0.2442,
            0.8707
        )
    )

    burn_design = eqx.tree_at(lambda b:
        (
            b.design_parameters.output_temperature,
            b.design_parameters.pressure_ratio,
            b.design_parameters.exit_mach_number,
        ),
        engine.combustor,
        (
            2857 * units.R,
            (1.0 - 0.054),
            0.1025,
        )
    )

    lpt_design = eqx.tree_at(lambda t:
        (
            t.efficiencies.flow,
            t.design_parameters.rotation_speed,
            t.design_parameters.exit_mach_number,
        ),
        engine.lpt,
        (
            0.8996,
            5000. * units.rev/units.mins,
            0.4127,
        )
    )

    hpt_design = eqx.tree_at(lambda t:
        (
            t.efficiencies.flow,
            t.design_parameters.rotation_speed,
            t.design_parameters.exit_mach_number,
        ),
        engine.hpt,
        (
            0.8888,
            15000. * units.rev/units.mins,
            0.3650,
        )
    )

    cn_design = eqx.tree_at(
        lambda n: n.efficiencies.flow,
        engine.core_nozzle,
        0.9933
    )

    fn_design = eqx.tree_at(
        lambda n: n.efficiencies.flow,
        engine.fan_nozzle,
        0.9939
    )

    engine = eqx.tree_at(lambda e:
        (
            e.inlet,
            e.fan,
            e.lpc,
            e.hpc,
            e.combustor,
            e.hpt,
            e.lpt,
            e.core_nozzle,
            e.fan_nozzle,
        ), engine,
        (
            inlet_design,
            fan_design,
            lpc_design,
            hpc_design,
            burn_design,
            hpt_design,
            lpt_design,
            cn_design,
            fn_design,
        )                
    )

    line = TurbofanLine(subcomponents=(engine,))
    
    net_design = TurbofanDesign(
        altitude=35_000.0 * units.ft,
        mach_number=0.8,
        thrust=5900 * units.lbf,
        initial_MFR=100.0 * units.lbm / units.s,
        initial_LPT_PR=3.0,
        initial_HPT_PR=4.0,
    )
    
    net = TurbofanNetwork(subcomponents=(line,), design_parameters=net_design)
    
    sys = Aircraft(tag="HBTF System", subcomponents=(net,))

    save_data(sys, test_dir / "HBTF_template.trs")

    return sys

if __name__ == "__main__":

    # Control Board
    DEBUG = False
    VERBOSE = True

    DESIGN_POINT = True

    # Build HBTF ---------------------------------------------------------------
    system = system_setup()
    settings = Settings(DEBUG_MODE=DEBUG, verbose=VERBOSE)

    if DESIGN_POINT:
        print("="*80)
        print(" Design Point Analysis")
        print("-"*80)
        
        st, sys, set = DesignTurbofan(
            state=State(),
            system=system,
            settings=settings,
        )

        save_data(sys, test_dir / "HBTF.trs")

        print("="*80)
        print(" System Validation")
        print("-"*80)

        for comp in sys.energy.line.engine.subcomponents:
            if hasattr(comp, "design_parameters") and comp.design_parameters:
                d = comp.design_parameters
                A_i = d.A_intake
                A_t = d.A_throat
                A_x = d.A_exit
                AR = d.A_ratio
                d_params = {"Intake Area": A_i, "Throat Area": A_t, "Exit Area":A_x, "Area_Ratio":AR}
                real_params = {k:a for k, a in d_params.items() if a != 1.0}
                if any(real_params):
                    print(f"{comp.tag}:")
                    for p in real_params:
                        print(f" - {p:<11}: {format_array(real_params[p])}")
        
        print("LPC Map Scaling:")
        c_map = sys.energy.line.engine.lpc.map
        print(f" - {'s_Wc':<11}: {format_array(c_map.s_Wc)}")
        print(f" - {'s_PR':<11}: {format_array(c_map.s_PR)}")
        print(f" - {'s_eff':<11}: {format_array(c_map.s_eff)}")
        print(f" - {'s_Nc':<11}: {format_array(c_map.s_Nc)}")

        print("HPC Map Scaling:")
        c_map = sys.energy.line.engine.hpc.map
        print(f" - {'s_Wc':<11}: {format_array(c_map.s_Wc)}")
        print(f" - {'s_PR':<11}: {format_array(c_map.s_PR)}")
        print(f" - {'s_eff':<11}: {format_array(c_map.s_eff)}")
        print(f" - {'s_Nc':<11}: {format_array(c_map.s_Nc)}")

        print("HPT Map Scaling:")
        t_map = sys.energy.line.engine.hpt.map
        print(f" - {'s_Wp':<11}: {format_array(t_map.s_Wp)}")
        print(f" - {'s_PR':<11}: {format_array(t_map.s_PR)}")
        print(f" - {'s_eff':<11}: {format_array(t_map.s_eff)}")
        print(f" - {'s_Np':<11}: {format_array(t_map.s_Np)}")

        print("LPT Map Scaling:")
        t_map = sys.energy.line.engine.lpt.map
        print(f" - {'s_Wp':<11}: {format_array(t_map.s_Wp)}")
        print(f" - {'s_PR':<11}: {format_array(t_map.s_PR)}")
        print(f" - {'s_eff':<11}: {format_array(t_map.s_eff)}")
        print(f" - {'s_Np':<11}: {format_array(t_map.s_Np)}")
        print("="*80)
        print ("\n\n")