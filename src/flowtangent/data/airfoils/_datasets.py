import numpy as np
import zarr

from ...utils.data import Dataset


class XFOILDataset(Dataset):
    """
    Ingests the XFOIL Zarr array and serves
    flattened, JAX-ready state vectors for surrogate training.
    """

    def __init__(self, zarr_path: str, mode: str = "regression"):
        self.mode = mode.lower()
        if self.mode not in ["classification", "regression"]:
            raise ValueError("Mode must be 'classification' or 'regression'.")

        # Zarr's lazy loading is perfect here; it only loads the metadata into RAM
        root = zarr.open_group(zarr_path, mode="r")

        raw_conditions = root["conditions"][:]
        raw_polars = root["polar_data"][:]

        # Calculate yield / feasibility
        converged_mask = ~np.isnan(raw_polars[:, :, 0])
        run_yields = np.sum(converged_mask, axis=1)

        if self.mode == "classification":
            self.X = raw_conditions.astype(np.float32)
            self.y = (run_yields > 0).astype(np.float32).reshape(-1, 1)

        elif self.mode == "regression":
            # Flatten the hypercube so every valid [Condition + Alpha] is a unique row
            valid_runs, valid_alphas = np.where(converged_mask)

            X_4D = raw_conditions[valid_runs]
            alphas = raw_polars[valid_runs, valid_alphas, 0:1]

            # X = [Flap, Hinge, Re, Mach, Alpha]
            self.X = np.hstack((X_4D, alphas)).astype(np.float32)

            # y = [CL, CD, CM]
            self.y = raw_polars[valid_runs, valid_alphas][:, [1, 2, 4]].astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Return pure numpy arrays. The ft.DataLoader handles the rest.
        return self.X[idx], self.y[idx]
