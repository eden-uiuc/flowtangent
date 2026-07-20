import json

import jax.numpy as jnp
import equinox as eqx
import numpy as np
import pandas as pd

from pathlib import Path

from eden_trace.utils import save_data, load_data, format_array

from eden_trace.library import units
from eden_trace.library.components.energy.nodes import FlowDesign
from eden_trace.library.components.energy.nodes import Efficiencies as Eff

from eden_trace.library.components.energy.networks import TurbofanNetwork, TurbofanDesign
from eden_trace.library.components.energy.jets import TurbofanEngine, JetDesign
from eden_trace.library.components.energy.lines import TurbofanLine

from eden_trace.framework import State, Aircraft, Settings
from eden_trace.framework.analyses.energy.turbojets import DesignTurbofan

test_dir = Path("./tests/PyCycle/PyCycle_Examples/HBTF")

def system_setup():
    
    engine = TurbofanEngine(design_parameters=JetDesign(thrust=5900 * units.lbf, bypass_ratio=5.105))

    # Inlet & Fan
    inlet_des = FlowDesign(exit_mach_number=0.751, pressure_recovery=0.999)
    fan_des = FlowDesign(pressure_ratio=1.685, exit_mach_number=0.4578, eff=Eff(flow=0.8948))
    
    c_duct_des = FlowDesign(pressure_ratio=(1-0.0048), exit_mach_number=0.3121)
    f_duct_des = FlowDesign(pressure_ratio=(1-0.0149), exit_mach_number=0.4589)

    engine = eqx.tree_at(lambda e: (
        e.inlet.design_parameters,
        e.fan.design_parameters,
        e.core_duct.design_parameters,
        e.fan_duct.design_parameters,
    ), engine, (inlet_des, fan_des, c_duct_des, f_duct_des))
    
    # Compressors
    lpc_des = FlowDesign(
            pressure_ratio=1.935,
            rotation_speed=5000. * units.rev/units.mins,
            exit_mach_number=0.3059,
            eff=Eff(flow=0.9243),)

    c_stat_des = FlowDesign(pressure_ratio=(1-0.0101), exit_mach_number=0.3563)

    hpc_des = FlowDesign(
            pressure_ratio=9.369,
            rotation_speed=15000. * units.rev/units.mins,
            exit_mach_number=0.2442,
            eff = Eff(flow=0.8707))
    
    cool_des = FlowDesign(exit_mach_number=0.3)

    engine = eqx.tree_at(lambda e: (
        e.lpc.design_parameters,
        e.compressor_stator.design_parameters,
        e.hpc.design_parameters,
        e.cooling_duct.design_parameters,
    ), engine, (lpc_des, c_stat_des, hpc_des, cool_des))
    
    # Combustor
    burn_des = FlowDesign(output_temperature=2857 * units.R, pressure_ratio=(1.0 - 0.054), exit_mach_number=0.1025,)
    
    engine = eqx.tree_at(lambda e: e.combustor.design_parameters, engine, burn_des)
    
    # Turbines
    hpt_des = FlowDesign(eff=Eff(flow=0.8888), rotation_speed=15000. * units.rev/units.mins, exit_mach_number=0.3650,)
    
    t_stat_des = FlowDesign(pressure_ratio=(1-0.0051), exit_mach_number=0.3063)
    
    lpt_des = FlowDesign(rotation_speed=5000. * units.rev/units.mins, exit_mach_number=0.4127, eff=Eff(flow=0.8996),)

    engine = eqx.tree_at(lambda e: (
        e.hpt.design_parameters,
        e.turbine_stator.design_parameters,
        e.lpt.design_parameters,
    ), engine, (hpt_des, t_stat_des, lpt_des))

    # Nozzles
    cn_duct_des = FlowDesign(pressure_ratio=(1-0.0107), exit_mach_number=0.4463)
    cn_des = FlowDesign(eff=Eff(flow=0.9933))
    fn_des = FlowDesign(eff=Eff(flow=0.9939))

    engine = eqx.tree_at(lambda e: (
        e.core_nozzle_duct.design_parameters,
        e.core_nozzle.design_parameters,
        e.fan_nozzle.design_parameters,
    ), engine, (cn_duct_des, cn_des, fn_des))

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