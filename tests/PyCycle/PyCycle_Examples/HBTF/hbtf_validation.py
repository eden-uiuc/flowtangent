import json

import jax.numpy as jnp
import equinox as eqx
import numpy as np
import pandas as pd

from eden_trace.utils import save_data, load_data, format_array

from eden_trace.library import units
from eden_trace.library.components.energy.networks import TurbojetNetwork, TurbojetDesign
from eden_trace.library.components.energy.jets import TurbofanEngine, JetDesign
from eden_trace.library.components.energy.lines import TurbofanLine

from eden_trace.framework import Aircraft
from eden_trace.framework.analyses.energy.turbojets import DesignTurbofan

def system_setup():

    engine = TurbofanEngine(design_parameters=JetDesign(thrust=5900 * units.lbf, bypass_ratio=5.105))
    
    inlet_design = eqx.tree_at(lambda i:
        i.design_parameters.exit_mach_number,
        engine.inlet,
        0.751
    )

    fan_design = eqx.tree_at(lambda f:
        (
            f.design_parameters.pressure_ratio,
            f.design_parameters.exit_mach_number,
            f.efficiencies.flow,
        ),
        engine.compressor,
        (
            1.685,
            8070. * units.rev/units.mins,
            0.4578,
            0.8948
        )
    )
    
    comp_design = eqx.tree_at(lambda c:
        (
            c.design_parameters.pressure_ratio,
            c.design_parameters.rotation_speed,
            c.design_parameters.exit_mach_number,
            c.efficiencies.flow,
        ),
        engine.compressor,
        (
            13.5,
            8070. * units.rev/units.mins,
            0.02,
            0.83
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
            2370 * units.R,
            0.97,
            0.02,
        )
    )

    turb_design = eqx.tree_at(lambda t:
        (
            t.efficiencies.flow,
            t.design_parameters.rotation_speed,
            t.design_parameters.exit_mach_number,
        ),
        engine.turbine,
        (
            0.86,
            8070. * units.rev/units.mins,
            0.4,
        )
    )

    nozz_design = eqx.tree_at(
        lambda n: n.efficiencies.flow,
        engine.core_nozzle,
        0.99
    )

    engine = eqx.tree_at(lambda e:
        (
            e.inlet,
            e.compressor,
            e.combustor,
            e.turbine,
            e.core_nozzle,
        ), engine,
        (
            inlet_design,
            comp_design,
            burn_design,
            turb_design,
            nozz_design
        )                
    )

    line = TurbojetEnergyLine(tag="Line", subcomponents=(engine,),)
    
    net_design = TurbojetDesign(
        altitude=0.0,
        mach_number=1e-6,
        thrust=11_800 * units.lbf,
        initial_MFR=168.453135137 * units.lbm / units.s,
        initial_turb_PR=4.46138725662
    )
    
    net = TurbojetNetwork(subcomponents=(line,), design_parameters=net_design)
    
    sys = Aircraft(tag="Simple Turbojet System", subcomponents=(net,))

    return sys
