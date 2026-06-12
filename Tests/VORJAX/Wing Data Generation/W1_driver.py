import jax
jax.config.update("jax_log_compiles", True)

import logging

import numpy as np
import pandas as pd
import equinox as eqx
import jax.numpy as jnp
import plotly.graph_objects as go

from plotly._subplots import make_subplots

from scipy.stats import qmc, beta

from RCAIDE.utils import PathTuple

from RCAIDE.Library import Units
from RCAIDE.Library.Components.Wings import Wing, WingChords, WingDimensions, WingSweeps

from RCAIDE.Framework import Aircraft, Settings, GradientMap
from RCAIDE.Framework.Settings import AnalysisSettings
from RCAIDE.Framework.Analyses.Batched import ShardedDatasetGenerator
from RCAIDE.Framework.Analyses.Aerodynamics.VORJAX import VORJAX_Settings, Vortices, BatchVORJAX

#-----------------------------------------------------------------------------------------------------------------------
# One Segment Wing Data
#-----------------------------------------------------------------------------------------------------------------------

def generate_flow_state_grid(
        
):
    alphas = np.linspace(-5.0, 15.0, 81) * Units.deg
    machs = np.linspace(0.1, 2.0, 20)
    betas = np.linspace(0.0, 10.0, 11) * Units.deg

    total_states = len(alphas) * len(machs) * len(betas)
    
    print(f"Total Flow States per Wing: {total_states}")

    return alphas, betas, machs

def generate_weighted_sobol(
        n_samples,
        aspect_ratio ={'a':2.0, 'b':7.0, 'loc':0.100, 'scale':30.00},
        taper_ratio  ={'a':4.0, 'b':6.0, 'loc':0.000, 'scale':1.000},
        sweep        ={'a':2.5, 'b':6.5, 'loc':-25.0, 'scale':115.0},
        twist        ={'a':9.0, 'b':5.0, 'loc':-10.0, 'scale':15.00},
        dihedral     ={'a':2.8, 'b':5.2, 'loc':-5.00, 'scale':50.00}
    ):
    
    rng = np.random.default_rng(seed=137)
    sampler=qmc.Sobol(d=5, scramble=True, rng=rng)
    sobol_uniform = sampler.random_base2(m=int(np.log2(n_samples)))

    u_ar, u_taper, u_sweep, u_twist, u_dihedral = sobol_uniform.T

    ar_samples          = beta.ppf(u_ar, **aspect_ratio)
    taper_samples       = beta.ppf(u_taper, **taper_ratio)
    sweep_samples       = beta.ppf(u_sweep, **sweep)
    twist_samples       = beta.ppf(u_twist, **twist)
    dihedral_samples    = beta.ppf(u_dihedral, **dihedral)

    return pd.DataFrame({
        "aspect_ratio": ar_samples,
        "taper_ratio": taper_samples,
        "sweep": sweep_samples * Units.deg,
        "twist": twist_samples * Units.deg,
        "dihedral": dihedral_samples * Units.deg
    })

def plot_sampling_validation(df_geometry, dist_kwargs):
    """
    Plots the empirical histograms of the sampled geometry against 
    their theoretical Beta probability density functions.
    
    Args:
        df_geometry: Pandas DataFrame containing the sampled points.
        dist_kwargs: Dictionary mapping column names to beta distribution kwargs.
    """
    # Create a 2x3 grid of subplots (leaves one blank)
    fig = make_subplots(
        rows=2, cols=3, 
        subplot_titles=list(dist_kwargs.keys())
    )
    
    # Colors for UIUC branding
    color_hist = '#13294B'  # Illini Blue
    color_pdf = '#FF5F05'   # Illini Orange

    row, col = 1, 1
    
    for col_name, kwargs in dist_kwargs.items():
        if col_name not in df_geometry.columns:
            continue
            
        # 1. Plot the Empirical Histogram from the DataFrame
        data = df_geometry[col_name]
        fig.add_trace(
            go.Histogram(
                x=data, 
                histnorm='probability density', # Scales histogram area to 1 (matches PDF)
                name=f"{col_name} Samples",
                marker_color=color_hist,
                opacity=0.6,
                nbinsx=50
            ),
            row=row, col=col
        )
        
        # 2. Calculate the Theoretical PDF
        # Create a smooth x-axis spanning the exact bounds
        x_min = kwargs['loc']
        x_max = kwargs['loc'] + kwargs['scale']
        x_smooth = np.linspace(x_min, x_max, 500)
        
        # Pass the kwargs directly into scipy's beta.pdf
        y_theoretical = beta.pdf(x_smooth, **kwargs)
        
        # 3. Overlay the PDF Line
        fig.add_trace(
            go.Scatter(
                x=x_smooth, 
                y=y_theoretical, 
                mode='lines',
                name=f"{col_name} PDF",
                line=dict(color=color_pdf, width=3)
            ),
            row=row, col=col
        )
        
        # Increment subplot grid positions
        col += 1
        if col > 3:
            col = 1
            row += 1

    fig.update_layout(
        title_text="Quasi-Monte Carlo (Sobol) Sampling vs. Theoretical Distributions",
        height=700, 
        width=1100,
        showlegend=False, # Hiding legend since subplots are titled
        template="plotly_white"
    )
    
    return fig

def encode_wing_id(ar, taper, sweep, twist, dihedral, prefix="UIUC-W1"):
    """
    Encodes continuous geometric parameters into a fixed-width string.
    Format: [Prefix]-[AR(4)]-[Taper(3)]-[Sweep(S+4)]-[Twist(S+4)]-[Dihedral(S+4)]
    """
    # Aspect Ratio: 4 digits (Always positive)
    ar_int = int(round(ar * 100))
    ar_str = f"{ar_int:04d}"
    
    # Taper Ratio: 3 digits (Always positive)
    taper_int = int(round(taper * 100))
    taper_str = f"{taper_int:03d}"
    
    # Helper for signed angle parameters
    def format_signed(val):
        val_int = int(round(val * 100))
        # Ensure that a rounded 0 isn't accidentally labeled negative
        sign = 'P' if val_int >= 0 else 'N'
        return f"{sign}{abs(val_int):04d}"
        
    sweep_str = format_signed(sweep)
    twist_str = format_signed(twist)
    dihedral_str = format_signed(dihedral)
    
    return f"{prefix}-{ar_str}-{taper_str}-{sweep_str}-{twist_str}-{dihedral_str}"

def wing_generator(df_geometries):

    for row in df_geometries.itertuples(index=False):

        wing = Wing(
            tag="W1 Wing",
            symmetric=True,
            taper=row.taper_ratio,
            dihedral=row.dihedral,
            sweeps=WingSweeps(quarter_chord=row.sweep),
            chords=WingChords(root=1.0),
            twists=WingDimensions(tip=row.twist),
            spans=WingDimensions(projected=row.aspect_ratio * (1 + row.taper_ratio)/2),
            origin=jnp.array([[0., 0., 0.]]),
        ).update_geometry(calculate_reference_area=True, calculate_wetted_area=True)

        system = Aircraft(tag=f"W1_System", areas=wing.areas).add_subcomponent(wing)
        system = eqx.tree_at(lambda s: s.mass_properties.center_of_gravity, system, jnp.array([[0.0, 0.0, 0.0]]))

        meta = {
            "aspect_ratio": row.aspect_ratio,
            "taper_ratio": row.taper_ratio,
            "sweep": row.sweep,
            "twist": row.twist,
            "dihedral": row.dihedral
        }

        yield system, meta


if __name__ == "__main__":
    GEN_W1 = False
    GEN_W2 = True
    GEN_W3 = False

    if GEN_W1:
        beta_distributions = {
            "aspect_ratio": {'a': 2.0, 'b': 7.0, 'loc':  0.10, 'scale': 30.00},
            "taper_ratio":  {'a': 4.0, 'b': 6.0, 'loc':  0.00, 'scale': 1.00},
            "sweep":        {'a': 2.5 , 'b': 6.5, 'loc': -25.0, 'scale': 115.0}, # Max Sweep = 90
            "twist":        {'a': 9.0, 'b': 5.0, 'loc': -10.0, 'scale': 15.00}, # Max Twist = +5
            "dihedral":     {'a': 2.8, 'b': 5.2, 'loc': -5.00, 'scale': 50.00}  # Max Dihedral = +45
        }

        df_geometry_W1 = generate_weighted_sobol(n_samples=4096, **beta_distributions)
        
        df_geometry_W1["wing_ID"] = df_geometry_W1.apply(
            lambda row: encode_wing_id(
                row["aspect_ratio"], 
                row["taper_ratio"], 
                row["sweep"], 
                row["twist"], 
                row["dihedral"]
            ), 
            axis=1
        )

        cols = ["wing_ID"] + [c for c in df_geometry_W1.columns if c != "wing_ID"]
        df_geometry_W1 = df_geometry_W1[cols]

        fig = plot_sampling_validation(df_geometry_W1, beta_distributions)
        fig.write_image("./Tests/VORJAX/Wing Data Generation/sampling_validation.png")
        # fig.show()

        alphas, betas, machs = generate_flow_state_grid()
        
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

        aero_settings = VORJAX_Settings(vortices=Vortices(n_spanwise=16, n_chordwise=8))
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
            system_iter=wing_generator(df_geometry_W1),
            total_systems=len(df_geometry_W1),
            state_kwargs={"alpha":alphas, "mach": machs, "beta": betas},
            state_mode="mesh",
            batch_size=1024
        )
    
    if GEN_W2:
        two_segment_beta_distributions = {
            "aspect_ratio": {'a': 2.0, 'b': 7.0, 'loc':  0.10, 'scale': 30.00},
            "segment_1": {
                "span_break": None,
                "taper_ratio": {'a': 4.0, 'b': 6.0, 'loc':  0.00, 'scale': 1.00},
                "sweep": {'a': 2.5 , 'b': 6.5, 'loc': -25.0, 'scale': 115.0},
                "twist": {'a': 9.0, 'b': 5.0, 'loc': -10.0, 'scale': 15.00},
                "dihedral": {'a': 2.8, 'b': 5.2, 'loc': -5.00, 'scale': 50.00},
            },
            "segment_2": {
                "span_break": 0.0,
                "taper_ratio": {'a': 4.0, 'b': 6.0, 'loc':  0.00, 'scale': 1.00}, # Compounded, monotonic taper
                "sweep": 0.0,
                "twist": 0.0,
                "dihedral": 0.0,
            },
        }