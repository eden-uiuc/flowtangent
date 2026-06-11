import os
import glob
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd

from dask import compute as dc

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

print("Computing Anomalies...")
min_cd = (df["CL"]**2) / (np.pi * df["aspect_ratio"]) - 1e-6
valid_physics = (df["CD"] >= min_cd) & (df["CL"].between(-10.0, 10.0)) & (df["CD"] < 4.0)

# Add it as a boolean column
df["anom_low_drag"] = df["CD"] < min_cd
df["anom_large_lift"] = ~df["CL"].between(-10.0, 10.0)
df["anom_high_drag"] = df["CD"] >= 4.0


df["is_normal"] = ~(df["anom_low_drag"] | df["anom_large_lift"] | df["anom_high_drag"])
percent_anomalous = (~df['is_normal']).mean().compute() * 100.0
print(f"\nPercent Anomalous: {percent_anomalous:.2f}%")

geom_cols = ["aspect_ratio", "taper_ratio", "sweep", "twist", "dihedral"]

print("--- Calculating Geometry Anomaly Rates ---")

# 1. Calculate the anomaly rate for each unique geometry (Executes on Dask)
# taking the mean of 'is_normal' gives the valid rate. 1.0 - valid = anomaly rate.
validity_rates = df.groupby(geom_cols)["is_normal"].mean().compute()
anomaly_rates = 1.0 - validity_rates
anomaly_rates.name = "geom_anomaly_rate"

# Convert the pandas series back to a dataframe for merging
anomaly_rates_df = anomaly_rates.reset_index()

print(f"Total Unique Geometries: {len(anomaly_rates_df)}")

# 2. Merge the rates back into the main Dask DataFrame
# Dask is extremely efficient at merging a small Pandas DF onto a huge Dask DF
df = df.merge(anomaly_rates_df, on=geom_cols, how="left")

# 3. Define the Splits based on your rules
# We only want to keep the rows where the specific flow state was ALSO normal
gold_mask = (df["geom_anomaly_rate"] <= 0.02) & df["is_normal"]
silver_mask = (df["geom_anomaly_rate"] > 0.02) & (df["geom_anomaly_rate"] <= 0.10) & df["is_normal"]
bronze_mask = (df["geom_anomaly_rate"] > 0.10) & (df["geom_anomaly_rate"] <= 0.20) & df["is_normal"]
problem_mask = (df["geom_anomaly_rate"] > 0.20) & df["is_normal"]

# Apply the masks lazily
df_gold = df[gold_mask]
df_silver = df[silver_mask]
df_bronze = df[bronze_mask]
df_problem  = df[problem_mask]

print("\n--- Computing Dataset Yields (This will take a moment) ---")

# 4. Compute the final row counts to see if we hit the 50M target
gold_yield, silver_yield, bronze_yield, problem_yield = dc(df_gold.shape[0], df_silver.shape[0], df_bronze.shape[0], df_problem.shape[0])

print("\n================ FINAL YIELD ================")
print(f"Gold Standard (>98% geom valid) : {gold_yield:,} rows")
print(f"Silver Set    (90-98% geom valid): {silver_yield:,} rows")
print(f"Bronze Set    (80-90% geom valid): {bronze_yield:,} rows")
print(f"Problem Set   (<80% geom valid): {problem_yield:,} rows")
print("=============================================")