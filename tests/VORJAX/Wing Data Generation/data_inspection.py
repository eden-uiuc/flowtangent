import os
import glob

import dask.dataframe as dd
import dask.array as da
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import equinox as eqx

from scipy.stats import entropy
from dask import compute as dc

from tqdm import tqdm

from flowtangent.library.components.wings import Wing, Chords, WingDimensions, Sweeps

from flowtangent.framework import Aircraft, State, Settings
from flowtangent.core._settings import AnalysisSettings
from flowtangent.framework.plotting import plot_vlm_panels

from flowtangent.framework.analyses.aero.VORJAX import VORJAX_Settings, Vortices
from flowtangent.framework.methods.aero.VORJAX import discretize_surfaces

def get_zarr_root():
    """
    Finds all shards and stitches them into a single lazy Dask dictionary.
    Zero RAM used for the actual data arrays here.
    """
    data_dir = "/media/jordan/Ashley_Backup/Wing Data Generation/W1/"
    
    # Grab all shards and sort them so row indices remain perfectly consistent
    shard_paths = sorted(glob.glob(os.path.join(data_dir, "*_shard_*.zarr")))
    
    if not shard_paths:
        raise FileNotFoundError(f"No shards found in {data_dir}")
        
    # The variables we want to load
    available_cols = [
        "alpha",
        "beta",
        "mach",
        "aspect_ratio",
        "dihedral",
        "taper_ratio",
        "sweep",
        "twist",
        "CL",
        "CD",
        "CX",
        "CY",
        "CZ",
        "C_l",
        "C_m",
        "C_n",
        "dCL_da",
        "dCL_db",
        "dCL_dM",
        "dCD_da",
        "dCD_db",
        "dCD_dM",
        ]
    
    stitched_data = {}
    for col in available_cols:
        lazy_arrays = [da.from_zarr(p, component=col) for p in shard_paths]
        stitched_data[col] = da.concatenate(lazy_arrays, axis=0)
    
    dask_series_list = [
        dd.from_dask_array(array, columns=[col])
        for col, array in stitched_data.items()
    ]

    df = dd.concat(dask_series_list, axis=1)
        
    return df


# 1. Load the Zarr data into Dask Arrays
# (Update these paths to match your Zarr group structure)
print("Loading data ...")
df = get_zarr_root()
print("Load complete.")

# Pull summary stats for just the Mach column
flow_stats = df[["mach", "alpha", "beta"]].describe().compute()
print("--- Flow Statistics ---")
print(flow_stats)


# 2. Define what makes a point "Anomalous" (Non-destructive tagging)
min_cd = (df["CL"]**2) / (np.pi * df["aspect_ratio"]) - 1e-6

# Add it as a boolean column
df["anom_low_drag"] = df["CD"] < min_cd
df["anom_large_lift"] = ~df["CL"].between(-10.0, 10.0)
df["anom_high_drag"] = df["CD"] >= 4.0

df["is_normal"] = ~(df["anom_low_drag"] | df["anom_large_lift"] | df["anom_high_drag"])
percent_anomalous = (~df['is_normal']).mean().compute() * 100.0
print(f"\nPercent Anomalous: {percent_anomalous:.2f}%")

# Define the inputs we want to investigate
input_columns = ["aspect_ratio", "taper_ratio", "sweep", "twist", "dihedral", "alpha", "mach", "beta"]

anomaly_types = ["anom_low_drag", "anom_large_lift", "anom_high_drag"]
colors = {"anom_low_drag": "red", "anom_large_lift": "orange", "anom_high_drag": "purple"}
labels = {"anom_low_drag": "Low Drag", "anom_large_lift": "Large Lift", "anom_high_drag": "High Drag"}

kl_divergences = {}
histograms = {}

print("\nComputing distribution metrics...")

for col in tqdm(input_columns, desc="Computing Inputs..."):
    # Get the global min and max to define consistent histogram bins
    col_min, col_max = dc(df[col].min(), df[col].max())
    bins = np.linspace(col_min, col_max, 50)
    
    # Calculate histograms lazily in Dask
    norm_vals = df[df["is_normal"]][col].to_dask_array(lengths=True)
    norm_counts, _ = da.histogram(norm_vals, bins=bins)
    norm_counts = norm_counts.compute()
    
    # Convert counts to Probability Mass Functions (PMFs)
    # Add a tiny epsilon to prevent divide-by-zero or log(0) in KL Divergence
    epsilon = 1e-9
    norm_pmf = (norm_counts / norm_counts.sum()) + epsilon
    norm_pmf /= norm_pmf.sum()
    
    col_data = {'bins': bins, 'normal': norm_pmf, 'anomalies': {}}
    
    # Calculate distributions for EACH anomaly type
    for anom in tqdm(anomaly_types, desc="Computing Anomalies"):
        anom_vals = df[df[anom]][col].to_dask_array(lengths=True)
        anom_counts, _ = da.histogram(anom_vals, bins=bins)
        anom_counts = anom_counts.compute()
        
        # Only process if this anomaly actually exists for this column
        if anom_counts.sum() > 0:
            anom_pmf = (anom_counts / anom_counts.sum()) + epsilon
            anom_pmf /= anom_pmf.sum()
            
            # Calculate KL Div for this specific anomaly mode
            kl_div = entropy(pk=anom_pmf, qk=norm_pmf)
            col_data['anomalies'][anom] = {'pmf': anom_pmf, 'kl': kl_div}
        else:
            col_data['anomalies'][anom] = None
            
    histograms[col] = col_data

# 3. Rank the inputs by KL Divergence to find the biggest culprits
print("\n--- Anomaly Drivers (Ranked by KL Divergence) ---")
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()

for i, col in enumerate(input_columns):
    ax = axes[i]
    data = histograms[col]
    bins = data['bins']
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    # Plot Normal Baseline
    ax.plot(bin_centers, data['normal'], label='Normal', color='blue', linewidth=2)
    
    # Plot each anomaly type
    title_kls = []
    for anom in anomaly_types:
        anom_data = data['anomalies'][anom]
        if anom_data is not None:
            ax.plot(bin_centers, anom_data['pmf'], label=labels[anom], color=colors[anom], linewidth=2, linestyle='--')
            # Keep fill transparent to avoid muddying the overlapping areas
            ax.fill_between(bin_centers, anom_data['pmf'], color=colors[anom], alpha=0.1)
            title_kls.append(f"{labels[anom][0]}: {anom_data['kl']:.1f}")
            
    # Format Title with KL Divergences
    kl_str = " | ".join(title_kls)
    ax.set_title(f"{col}\nKL -> {kl_str}", fontsize=10)
    ax.set_ylabel("Probability Density")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("./tests/VORJAX/Wing Data Generation/anomaly_distributions.png")
print("\nPlot saved as 'anomaly_distributions.png'")

print("--- Extracting Anomaly Geometries ---")

def wing_generator(df_geometries):

    for row in df_geometries.itertuples(index=False):

        wing = Wing(
            tag="W1 Wing",
            symmetric=True,
            taper=row.taper_ratio,
            dihedral=row.dihedral,
            sweeps=Sweeps(quarter_chord=row.sweep),
            chords=Chords(root=1.0),
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

def wing_renderer(wing_system):

    aero_settings = VORJAX_Settings(vortices=Vortices(n_spanwise=16, n_chordwise=8))
    analysis_settings = AnalysisSettings(
        aerodynamics=aero_settings,
    )
    settings = Settings(analysis=analysis_settings, DEBUG_MODE=False)

    _, full_system, _ = discretize_surfaces(State(), wing_system, settings)

    return plot_vlm_panels(full_system.analysis_data["vortex_distribution"])

def encode_wing_id(aspect_ratio, taper_ratio, sweep, twist, dihedral, prefix=""):
    """
    Encodes continuous geometric parameters into a fixed-width string.
    Format: [Prefix]-[AR(4)]-[Taper(3)]-[Sweep(S+4)]-[Twist(S+4)]-[Dihedral(S+4)]
    """
    # Aspect Ratio: 4 digits (Always positive)
    ar_int = int(round(aspect_ratio * 100))
    ar_str = f"{ar_int:04d}"
    
    # Taper Ratio: 3 digits (Always positive)
    taper_int = int(round(taper_ratio * 100))
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

for anom in anomaly_types:
    # 1. Filter for the anomaly
    # 2. Select only the geometric columns
    # 3. Drop all duplicate rows to find the unique bad wings
    # 4. Compute to bring it into a local Pandas DataFrame
    unique_geom_df = df[df[anom]][["aspect_ratio", "taper_ratio", "sweep", "twist", "dihedral"]].drop_duplicates().compute()
    
    if len(unique_geom_df) > 0:
        print(f"\n[ {anom.upper()} ] - Found {len(unique_geom_df)} unique problem geometries:")
        
        # Print cleanly to console
        print(unique_geom_df.to_string(index=False))
        
        # Save to CSV for your mesher
        csv_filename = f"./tests/VORJAX/Wing Data Generation/unique_failed_geometries_{anom}.csv"
        unique_geom_df.to_csv(csv_filename, index=False)
        print(f"-> Saved unique geometries to {csv_filename}")

        # faulty_wing_gen = wing_generator(unique_geom_df)

        # for faulty_wing, meta in faulty_wing_gen:
            # wing_fig = wing_renderer(faulty_wing)
            # wing_fig.write_html("./tests/VORJAX/Wing Data Generation/Wing Renders/"+encode_wing_id(**meta, prefix=f"{anom}")+".html")
        
    else:
        print(f"\n[ {anom.upper()} ] - 0 instances found.")


geom_cols = ["aspect_ratio", "taper_ratio", "sweep", "twist", "dihedral"]
flow_cols = ["alpha", "beta", "mach", "CL", "CD"]

print("\n--- Analyzing Anomalous Flow States Per Wing ---")

# 1. Calculate the total number of flow states generated per wing 
# (This runs on Dask to count the full dataset)
total_states_per_wing = df.groupby(geom_cols).size().compute().rename("Total_States")

# 2. Extract ONLY the anomalous rows into a local Pandas DataFrame
anom_pd = df[~df["is_normal"]].compute()

# 3. Group the local anomalies by their geometry
anom_grouped = anom_pd.groupby(geom_cols)

# 4. Loop through each bad wing and print the diagnostics
for geom, group_df in anom_grouped:
    
    # Safely get the total states evaluated for this specific wing geometry
    total_states = total_states_per_wing.loc[geom]
    anom_count = len(group_df)
    anom_percent = (anom_count / total_states) * 100
    
    print(f"\n========================================================")
    print(f"WING GEOMETRY:")
    for col_name, val in zip(geom_cols, geom):
        print(f"  {col_name:<15}: {val:.4f}")
        
    print(f"\n-> Anomalous States: {anom_count} / {total_states} ({anom_percent:.2f}%)")
    
    # Print the summary stats for Alpha, Beta, and Mach for these specific failures
    print("\nFlow State Distribution for these Anomalies:")
    print(group_df[flow_cols].describe().round(3))