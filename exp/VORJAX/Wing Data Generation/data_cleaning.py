import os
import dtale
import glob
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd

from pathlib import Path
from dask.base import compute as dc
from dask.diagnostics.progress import ProgressBar
from pysr import PySRRegressor

def get_zarr_root(data_dir, verbose=False):
    """
    Finds all shards and stitches them into a single lazy Dask dictionary.
    Zero RAM used for the actual data arrays here.
    """

    if verbose:
        print(f"Loading data from {data_dir}...")
    
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

    if verbose:
        print(f"Load complete.")
    
    return df

def get_gold_data(data_dir, verbose=False):

    df = get_zarr_root(data_dir, verbose=verbose)

    min_cd = (df["CL"]**2) / (np.pi * df["aspect_ratio"]) - 1e-6

    df["min_cd"] = min_cd
    df["anom_low_drag"] =  min_cd - df["CD"] > 0.1
    df["drag_anomaly_mag"] = min_cd - df["CD"]

    df["anom_large_lift"] = ~df["CL"].between(-10.0, 10.0)
    df["anom_high_drag"] = df["CD"] >= 4.0

    df["is_normal"] = ~(df["anom_low_drag"] | df["anom_large_lift"] | df["anom_high_drag"])

    geom_cols = ["aspect_ratio", "taper_ratio", "sweep", "twist", "dihedral"]

    # 1. Calculate the anomaly rate for each unique geometry (Executes on Dask)
            # taking the mean of 'is_normal' gives the valid rate. 1.0 - valid = anomaly rate.
    if verbose:
        print("\nCalculating Geometry Anomaly Rates...")
        with ProgressBar():
            validity_rates = df.groupby(geom_cols)["is_normal"].mean().compute()
    else:
        validity_rates = df.groupby(geom_cols)["is_normal"].mean().compute()

    anomaly_rates = 1.0 - validity_rates
    anomaly_rates.name = "geom_anomaly_rate"

    # Convert the pandas series back to a dataframe for merging
    anomaly_rates_df = anomaly_rates.reset_index()
    df = df.merge(anomaly_rates_df, on=geom_cols, how="left")

    gold_mask = (df["geom_anomaly_rate"] <= 0.02) & df["is_normal"]

    df_gold = df[gold_mask]

    return df_gold

if __name__ == "__main__":

    print("\nCalculating Geometry Anomaly Rates...")


    # 1. Load the Zarr data into Dask Arrays
    # (Update these paths to match your Zarr group structure)
    print("Loading data ...")
    df = get_zarr_root("/home/jordan/dev/data/Wing Data Generation/W1/")
    print("Load complete.")

    min_cd = (df["CL"]**2) / (np.pi * df["aspect_ratio"]) - 1e-6

    # Add it as a boolean column
    df["min_cd"] = min_cd
    df["anom_low_drag"] =  min_cd - df["CD"] > 0.1
    df["drag_anomaly_mag"] = min_cd - df["CD"]

    # with ProgressBar():
    #     drag_stats = df[df["anom_low_drag"]][["min_cd", "drag_anomaly_mag"]].describe().compute()
    # print(f"--- Low Drag Anomaly Summary---")
    # print(drag_stats)

    # print(f"Preparing Low Drag DataFrame...")
    # with ProgressBar():
    #     low_drag_pdf = df[df["anom_low_drag"]][["CL", "CD", "aspect_ratio", "min_cd", "drag_anomaly_mag"]].compute()
    # d = dtale.show(low_drag_pdf, host='localhost')
    # print(f"Data View: {d.main_url()}")

    # input("D-Tale server is running. Press Enter to continue...")

    df["anom_large_lift"] = ~df["CL"].between(-10.0, 10.0)
    df["anom_high_drag"] = df["CD"] >= 4.0

    df["is_normal"] = ~(df["anom_low_drag"] | df["anom_large_lift"] | df["anom_high_drag"])

    print("Computing Anomalies...")
    with ProgressBar():
        percent_anomalous = (~df['is_normal']).mean().compute() * 100.0
    print(f"\nPercent Anomalous: {percent_anomalous:.2f}%")

    geom_cols = ["aspect_ratio", "taper_ratio", "sweep", "twist", "dihedral"]

    print("\nCalculating Geometry Anomaly Rates...")

    # 1. Calculate the anomaly rate for each unique geometry (Executes on Dask)
    # taking the mean of 'is_normal' gives the valid rate. 1.0 - valid = anomaly rate.
    with ProgressBar():
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

    # print("\nComputing Dataset Yields...")

    # # 4. Compute the final row counts to see if we hit the 50M target
    # with ProgressBar():
    #     gold_yield, silver_yield, bronze_yield, problem_yield = dc(df_gold.shape[0], df_silver.shape[0], df_bronze.shape[0], df_problem.shape[0])

    # print("\n================ FINAL YIELD ================")
    # print(f"Gold Standard (>98% geom valid) : {gold_yield:,} rows")
    # print(f"Silver Set    (90-98% geom valid): {silver_yield:,} rows")
    # print(f"Bronze Set    (80-90% geom valid): {bronze_yield:,} rows")
    # print(f"Problem Set   (<80% geom valid): {problem_yield:,} rows")
    # print("=============================================")

    print("Extracting reference Mach 0.1 data...")
    group_keys = geom_cols + ['alpha', 'beta']
    ref_df = df_gold[df_gold['mach'] == 0.1][group_keys + ['CL', 'mach']]
    ref_df = ref_df.rename(columns={'CL': 'CL_ref', 'mach': 'mach_ref'})

    with ProgressBar():
        ref_pdf = ref_df.compute()

    # duplicates = ref_pdf.duplicated(subset=group_keys, keep=False)
    # num_dupes = duplicates.sum()

    # print(f"Number of non-unique reference rows: {num_dupes}")

    # if num_dupes > 0:
    #     print("Example duplicate rows:")
    #     print(ref_pdf[duplicates].sort_values(group_keys).head(6))

    print(f"Sampling Mach data...")
    with ProgressBar():
        high_mach = df_gold[(df_gold['mach'] <= 0.95) & (df_gold['mach'] > 0.6)].sample(frac=0.01).compute()
        low_mach = df_gold[df_gold['mach'] <= 0.6].sample(frac=0.001).compute()
    sub_df = pd.concat([high_mach, low_mach])

    print("Merging and calculating PG scaling...")
    merged_df = sub_df.merge(ref_pdf, on=group_keys, how='inner')

    cos_sweep = da.cos(merged_df['sweep'])
    M_eff_actual = merged_df['mach'] * cos_sweep
    M_eff_ref = merged_df['mach_ref'] * cos_sweep

    pg_scale_factor = da.sqrt(1 - M_eff_ref**2) / da.sqrt(1 - M_eff_actual**2)
    merged_df['CL_PG_pred'] = merged_df['CL_ref'] * pg_scale_factor

    print("Calculating PG error...")
    merged_df['PG_error'] = merged_df['CL'] - merged_df['CL_PG_pred']
    merged_df['PG_error_percent'] = ((merged_df['CL'] - merged_df['CL_PG_pred']) / merged_df['CL_PG_pred']) * 100

    print("Computing summary statistics...")
    summary_stats = merged_df[['PG_error', 'PG_error_percent']].describe()

    print("\n--- Error Summary ---")
    print(summary_stats)

    # print("Sampling Data...")
    # with ProgressBar():
    #     sample_pdf = merged_df.head(5000)
    # d = dtale.show(sample_pdf, host='localhost')
    # print(f"Data View: {d.main_url()}")
    # input("D-Tale server is running! Press Enter in this terminal to close...")

    # 1. Pull a representative training set from your clean Dask DataFrame
    print("Sampling training data for PySR...")
    train_df = merged_df.sample(frac=0.0005, random_state=42)

    # 2. Define our inputs (Features) and output (Target)
    train_df["M_eff"] = train_df['mach'] * np.cos(train_df['sweep'])
    X = train_df[['CL_ref', 'M_eff']]
    y = train_df['CL']

    # 3. Initialize the AI Theoretician
    # We give it the exact mathematical building blocks it needs to discover compressibility
    model = PySRRegressor(
        niterations=100,  # How long it evolves (increase if it needs more time)
        binary_operators=["+", "-", "*", "/"],
        unary_operators=[
            "sqrt",      # Critical for discovering the 1/sqrt(1-M^2) term
            "square",    # Critical for M^2
            # "cos",       # Critical for sweep angle effects
        ],
        # Complexity penalties force it to prefer simple, elegant physics over messy curve fits
        parsimony=1e-3,  
        # PySR will aggressively drop equations that throw errors (like dividing by zero)
        loss="loss(prediction, target) = (prediction - target)^2",
    )

    print("Fitting PySR model...")
    # 4. Run the evolutionary search
    model.fit(X, y)

    print("\nSearch complete! Here is the Pareto Front of discovered equations:")
    print(model)