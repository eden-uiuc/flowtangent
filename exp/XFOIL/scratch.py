import flowtangent as ft

ds = ft.utils.data.XArrayDataset("../../datasets/xfoil_4d.zarr").to_tabular(["run", "alpha"])
print("Done.")