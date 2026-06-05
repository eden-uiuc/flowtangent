import numpy as np
import equinox as eqx
import jax.numpy as jnp

from scipy.stats import qmc

from RCAIDE.utils import PathTuple

from RCAIDE.Library.Components.Wings import Wing, WingChords, WingDimensions, WingSweeps

from RCAIDE.Framework import Aircraft, Settings, GradientMap
from RCAIDE.Framework.Settings import AnalysisSettings
from RCAIDE.Framework.Analyses.Batched import ShardedDatasetGenerator
from RCAIDE.Framework.Analyses.Aerodynamics.VORJAX import VLMSettings, Vortices, BatchVORJAX


#-----------------------------------------------------------------------------------------------------------------------
# One Segment Wing Data
#-----------------------------------------------------------------------------------------------------------------------



def W1_dataset_plan(m=12):
    
    # Define Geometric Bounds (Lower and Upper)
    # Variables: [AR,    Taper,  QC_Sweep,  Dihedral]
    bounds_low = [5.0,   0.1,    0.0,       -30.0]
    bounds_up  = [30.0,  1.0,    60.0,      30.0]

    # Generate Sobol sequence and sample from bounds
    sampler = qmc.Sobol(d=4, scramble=True)
    unit_samples = sampler.random_base2(m=m)
    geometries = qmc.scale(unit_samples, bounds_low, bounds_up)

    alphas = np.linspace(-5.0, 15.0, 81)
    machs = np.linspace(0.1, 2.0, 20)
    betas = np.linspace(0.0, 10.0, 11)

    total_states = len(geometries) * len(alphas) * len(machs) * len(betas)
    
    print(f"Total Unique Wings: {len(geometries)}")
    print(f"Total Flow States per Wing: {len(alphas) * len(machs) * len(betas)}")
    print(f"Total Dataset Size: {total_states:,} states")
    
    return geometries, alphas, machs, betas

def wing_generator(geometries):

    for AR, taper, QC, d in geometries:

        wing = Wing(
            tag=f"W1",
            symmetric=True,
            taper=taper,
            dihedral=d,
            sweeps=WingSweeps(quarter_chord=QC),
            chords=WingChords(root=1.0),
            spans=WingDimensions(projected=AR),
            origin=jnp.array([[0., 0., 0.]]),
        ).update_geometry(calculate_reference_area=True, calculate_wetted_area=True)

        system = Aircraft(tag="W1 System", areas=wing.areas).add_subcomponent(wing)
        system = eqx.tree_at(lambda s: s.mass_properties.center_of_gravity, system, jnp.array([[0.0, 0.0, 0.0]]))

        yield system, {"AR":AR, "taper":taper, "QC_Sweep":QC, "Dihedral":d}


if __name__ == "__main__":

    geometries, alphas, machs, betas = W1_dataset_plan(12)
    
    solver=BatchVORJAX()

    mach_path   = PathTuple(("freestream", "mach_number"), tag="M")
    alpha_path  = PathTuple(("aerodynamics", "angles", "alpha"), tag="a")
    beta_path   = PathTuple(("aerodynamics", "angles", "beta"), tag="b")

    lift_path   = PathTuple(("aerodynamics", "coefficients", "lift", "total"), tag="CL")
    drag_path   = PathTuple(("aerodynamics", "coefficients", "drag", "total"), tag="CD")

    GRAD_MAP = GradientMap(
        state_inputs=(
            mach_path,
            alpha_path,
            beta_path,
        ),
        state_outputs=(
            lift_path,
            drag_path
        ))

    aero_settings = VLMSettings(vortices=Vortices(n_spanwise=16, n_chordwise=8))
    analysis_settings = AnalysisSettings(
        aerodynamics=aero_settings,
        gradient_map=GRAD_MAP
    )
    settings = Settings(analysis=analysis_settings, DEBUG_MODE=False)

    generator = ShardedDatasetGenerator(
        batch_analysis=solver,
        cache_dir="./Tests/VORJAX/Wing Data Generation/W1",
        storage_dir="/media/jordan/Ashley_Backup/Wing Data Generation/W1",
        shard_size=3_000_000,
        tag="W1"
    )
    
    generator.run(
        settings=settings,
        system_iter=wing_generator(geometries),
        total_systems=len(geometries),
        state_kwargs={"alpha":alphas, "mach": machs, "beta": betas},
        state_mode="mesh",
        batch_size=1024
    )