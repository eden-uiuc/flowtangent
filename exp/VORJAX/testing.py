import os
import glob
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dask import compute as dc 
import zarr
import time
start = time.perf_counter()
path = sorted(glob.glob("/Users/jameskwak/Desktop/UIUC/W1/w1_shard_*.zarr"))
# path = "/Users/jameskwak/Desktop/UIUC/W1/w1_shard_0000.zarr" #single test case
components = [
    "alpha",
    "aspect_ratio",
    "beta",
    "C_l",
    "C_m",
    "C_n",
    "CD",
    "CL",
    "CX",
    "CY",
    "CZ",
    "dCD_da",
    "dCD_db",
    "dCD_dM",
    "dCL_da",
    "dCL_db",
    "dCL_dM",
    "dihedral",
    "mach",
    "sweep",
    "taper_ratio",
    "twist"
] 

data ={}
dask_series_list=[]

for name in components:
    # multiple files 
    lazy_array = [da.from_zarr(p, component = name) for p in path]
    data[name] = da.concatenate(lazy_array, axis=0)
    # single files
    # lazy_array = da.from_zarr(path, component = name)
    # data[name] = da.concatenate([lazy_array], axis=0).reshape(-1)
        

data_series_list= [ 
    dd.from_dask_array(array, columns =[col]) 
    for col, array in data.items()
]

df = dd.concat(data_series_list, axis =1)


# subset_beta0 = df[df["beta"] == 0]

group_cols =[
    "aspect_ratio",
    "sweep",
    "taper_ratio",
    "twist",
    "dihedral",
    "mach",
    "beta"
]
def find_slope(group):
    x = group["alpha"].to_numpy()
    y = group["CL"].to_numpy()
    slope, intercept = np.polyfit(x,y,1)
    return pd.Series ({
        "slope": slope,
        "intercept": intercept,
        "alpha_count": len(group)
    })
meta = {
    "slope": "float64",
    "intercept": "float64",
    "alpha_count": "int64"
}

filtered_df = df[
    (df["CL"] >= -2) &
    (df["CL"] <= 4) &
    (df["mach"]<= 0.5) & 
    (df["CD"] >=0) & 
    (df["CD"] <=0.3) 
]


slope_data = filtered_df.groupby(group_cols).apply(
    find_slope, 
    meta = meta
)

slope_data = slope_data.compute().reset_index()
slope_data["difference"] = np.abs(2*np.pi-slope_data["slope"])

index_min = slope_data["difference"].idxmin()
closest = slope_data.loc[index_min]

print(closest)
print(f"AR = {closest['aspect_ratio']:.3g}")
print(f"sweep = {closest['sweep']:.3g}")
print(f"taper_ratio = {closest['taper_ratio']:.3g}")
print(f"twist = {closest['twist']:.3g}")
print(f"dihedral = {closest['dihedral']:.3g}")
print(f"mach = {closest['mach']:.3g}")
print(f"beta = {closest['beta']*180/np.pi:.3g}")

end = time.perf_counter()
print(f"time elapsed: {(end - start):.3g} s" )

# unique = {
#     col: da.unique(data[col])
#     for col in components
# }

# unique_results = da.compute(*unique.values())
# unique_count ={}

# for i, col in enumerate(unique.keys()):
#     values = unique_results[i]
#     unique_count[col] = len(values)

# sorted_data = df.sort_values(
#     by=[
#         "aspect_ratio",
#         "sweep",
#         "taper_ratio",
#         "twist",
#         "dihedral",
#         "mach",
#         "beta",
#         "alpha"
#     ]
# )
# subset_beta0 = sorted_data[
#     (sorted_data["beta"]==0)
# ]
# subset= subset_beta0.compute()
# n_row = len(subset)
# alpha_count = unique_count["alpha"] 
# beta_count = unique_count["beta"] 
# mach_count = unique_count["mach"] 

# subset_break = np.arange(0,n_row, alpha_count)

# geometry_subsets =[]
# for i in subset_break:
#     geometry = subset.iloc[i:i+alpha_count]
#     geometry_subsets.append(geometry)

# slope = np.zeros(len(geometry_subsets))

# for i, geometry in enumerate(geometry_subsets):
#     x = geometry["alpha"]
#     y = geometry["CL"]
#     temp_slope, temp_intercept = np.polyfit(x,y,1)
#     slope[i] = temp_slope


# slope_difference = np.abs(2 * np.pi- slope)
# min_difference = np.min(slope_difference)
# index_min = np.argmin(slope_difference)
# print(index_min, slope[index_min])

# x= np.linspace(-0.1,0.26, 2)
# closest = geometry_subsets[index_min]

# plt.figure(1)
# plt.scatter(closest["alpha"]*180/np.pi, closest["CL"])
# plt.plot(x*180/np.pi,2*np.pi*x, color="tab:red")

# print (closest["aspect_ratio"].iloc[0], closest["sweep"].iloc[0], closest["dihedral"].iloc[0], closest["taper_ratio"].iloc[0],closest[ "twist"].iloc[0])
# print(closest.describe())
# plt.show()


# subset= df[
#             # (df["beta"]==0) &
#             (df["mach"]== mach_value) 
#             &
#             (df["dihedral"] ==dihedral_value) &
#             (df["twist"] == twist_value) &
#             (df["aspect_ratio"] == AR)
# ]


# plt.figure(1)
# # sc= plt.scatter(subset["alpha"]*180/np.pi,subset["beta"],c=subset["CL"])
# plt.scatter(subset["alpha"]*180/np.pi,subset["CL"])
# plt.figure(2)
# plt.scatter(subset["CD"], subset["CL"])


# # plt.colorbar(sc, label="CL")
# # plt.xlabel("Alpha")
# # plt.ylabel("Beta")

# plt.show()