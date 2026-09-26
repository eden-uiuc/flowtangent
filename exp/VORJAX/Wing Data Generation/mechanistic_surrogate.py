import numpy as np
import pandas as pd
import hdbscan

from sklearn.preprocessing import StandardScaler
from dask.diagnostics import ProgressBar
from gtda.mapper import CubicalCover, make_mapper_pipeline, plot_static_mapper_graph
from sklearn.cluster import DBSCAN
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier, export_text

from data_cleaning import get_gold_data

df_gold = get_gold_data("/home/jordan/dev/data/Wing Data Generation/W1/", verbose=True)

# 1. Define the Vector Space
# X: The independent state and geometry
X_cols = ["mach", "alpha", "beta", "aspect_ratio", "sweep", "twist", "dihedral", "taper_ratio"]
# Y: The coupled aerodynamic response (Forces and Moments)
Y_cols = ["CL", "CD", "CX", "CY", "CZ", "C_l", "C_m", "C_n"]

grad_cols = [
    "dCL_da", "dCL_db", "dCL_dM", 
    "dCD_da", "dCD_db", "dCD_dM"
]

print("Sampling vector-valued topological skeleton...")
with ProgressBar():
    skeleton_df = df_gold.sample(frac=0.0025, random_state=42).compute()

X_raw = skeleton_df[X_cols].values
Y_raw = skeleton_df[Y_cols].values

# 2. Scale inputs and outputs so no single coefficient dominates the topology
X_scaler = StandardScaler()
Y_scaler = StandardScaler()
grad_scaler = StandardScaler()
X_scaled = X_scaler.fit_transform(X_raw)
Y_scaled = Y_scaler.fit_transform(Y_raw)
grad_scaled = grad_scaler.fit_transform(skeleton_df[grad_cols])

# print("="*120 + "\nGraphical Mapper Algorithm\n" + "-"*120)

# skeleton_df["lens_exact_sensitivity"] = np.linalg.norm(grad_scaled, axis=1)

# print("Lens computed from JAX autodiff.")

# class PrecomputedLens(BaseEstimator, TransformerMixin):
#     def __init__(self, lens_array):
#         self.lens_array = lens_array
        
#     def fit(self, X, y=None):
#         return self
        
#     def transform(self, X, y=None):
#         # When Giotto asks for the filter values, we just hand it our array
#         return self.lens_array

# # Configure the Mapper Pipeline
# # n_intervals: How many slices we cut the sensitivity into
# # overlap_frac: The overlap that connects the linear regime to the non-linear one
# cover = CubicalCover(n_intervals=20, overlap_frac=0.3)

# # DBSCAN clusters the actual wing geometries inside each sensitivity slice
# clusterer = DBSCAN(eps=0.3, min_samples=50)

# lens_array = np.log1p(skeleton_df[["lens_exact_sensitivity"]].values)
# clipped_lens = np.clip(lens_array, a_min=None, a_max=np.percentile(lens_array, 99)).reshape(-1, 1)
# lens_transformer = PrecomputedLens(clipped_lens)

# # 3. Pass the transformer to the pipeline instead of the raw array
# mapper = make_mapper_pipeline(
#     scaler=None,          
#     filter_func=lens_transformer,
#     cover=cover,
#     clusterer=clusterer
# )
# print("Building topological Mapper graph...")

# # Generate and Plot the Interactive Graph
# fig = plot_static_mapper_graph(
#     mapper, 
#     X_scaled, 
#     layout="kamada_kawai", # A force-directed layout perfect for branching manifolds
#     color_data=lens_array, # Color the nodes by the sensitivity lens
# )

# # fig.show()

# graph = mapper.fit_transform(X_scaled)
# node_elements = {v.index: v["node_elements"] for v in graph.vs}

# node_summaries = []

# summary_cols = ["mach", "alpha", "sweep", "aspect_ratio", "CL", "CD", "lens_exact_sensitivity"]

# for node_id, row_indices in node_elements.items():
#     # Pull the exact rows from our skeleton dataframe belonging to this node
#     node_df = skeleton_df.iloc[row_indices]
    
#     # Calculate key physical metrics for this specific topological cluster
#     summary = {
#         "Node_ID": node_id,
#         "Point_Count": len(node_df),
#         "Mach_Mean": node_df["mach"].mean(),
#         "Mach_Max": node_df["mach"].max(),
#         "Alpha_Mean": node_df["alpha"].mean(),
#         "Sweep_Mean": node_df["sweep"].mean(),
#         "CL_Mean": node_df["CL"].mean(),
#         "CD_Mean": node_df["CD"].mean(),
#         "Lens_Sens_Mean": node_df["lens_exact_sensitivity"].mean(),
#         "Lens_Sens_Max": node_df["lens_exact_sensitivity"].max(),
#     }
#     node_summaries.append(summary)

# # Convert to a clean summary DataFrame
# summary_df = pd.DataFrame(node_summaries).sort_values("Lens_Sens_Mean")
# print(summary_df.to_string(index=False))

print("="*120 + "\nHDBSCAN Algorithm\n" + "-"*120)

# 3. Run HDBSCAN purely on the tangent space behavior
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=500,     # "Don't bother me with equations that govern fewer than 500 points"
    min_samples=50,           # How conservative to be about the boundaries
    cluster_selection_epsilon=0.5 # Merge clusters that are mathematically practically touching
)

skeleton_df["manifold_ID"] = clusterer.fit_predict(grad_scaled)

manifold_counts = skeleton_df["manifold_ID"].value_counts().sort_index()
print("\nDiscovered Manifolds (ID -1 represents the bifurcations/chaotic boundaries):")
print(manifold_counts)

# 1. Direct Statistical Summary
print("--- Direct Statistical Summary ---")
features_to_summarize = ["mach", "alpha", "sweep", "aspect_ratio", "CL", "CD"]
summary = skeleton_df.groupby("manifold_ID")[features_to_summarize].mean().round(4)
summary["Point_Count"] = skeleton_df["manifold_ID"].value_counts()
print(summary)
print("\n" + "="*50 + "\n")

# 2. Comparative Analysis (Decision Tree)
print("--- Comparative Physical Rules ---")
print("Training decision tree to explain the manifold boundaries...\n")

# We only want to look at the inputs (geometry and state)
X_comp = skeleton_df[["mach", "alpha", "sweep", "aspect_ratio"]]
y_comp = skeleton_df["manifold_ID"]

# Train a shallow decision tree (max depth 3 keeps the rules readable)
tree = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=42)
tree.fit(X_comp, y_comp)

# Export the tree as a human-readable rule set
tree_rules = export_text(tree, feature_names=list(X_comp.columns))
print(tree_rules)

# 3. Feature Importance
print("\n--- What drives the manifold splits? ---")
importances = pd.Series(tree.feature_importances_, index=X_comp.columns)
print(importances.sort_values(ascending=False).round(3))