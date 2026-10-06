from pathlib import Path

import zarr
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
from sklearn.tree import DecisionTreeClassifier, export_text

FILE_DIR = Path(__file__).resolve().parent

print("Loading Zarr store into RAM...")
root = zarr.open_group(FILE_DIR / "data.zarr", mode='r')
print("Load complete.")

conditions = root['conditions'][:]
polar_data = root['polar_data'][:]
airfoil_names = root['foil_name'][:]

converged_alphas = np.sum(~np.isnan(polar_data[:, :, 0]), axis=1)
total_alphas = polar_data.shape[1]
yield_pct = (converged_alphas / total_alphas) * 100

unique_airfoils = sorted([str(n) for n in np.unique(airfoil_names) if str(n).strip()])
dropdown_options = [{'label': 'ALL AIRFOILS (Global View)', 'value': 'ALL'}] + \
                   [{'label': name, 'value': name} for name in unique_airfoils]

app = Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.H2("XFOIL 4D Convergence Diagnostics", style={'fontFamily': 'sans-serif'}),
        dcc.Dropdown(
            id='airfoil-dropdown',
            options=dropdown_options,
            value='ALL',
            clearable=False,
            style={'width': '400px', 'fontSize': '16px'}
        )
    ], style={'padding': '20px'}),
    
    html.Div([
        # Left side: 3D Scatter
        html.Div([dcc.Graph(id='plot-3d', style={'height': '70vh'})], style={'width': '60%', 'display': 'inline-block'}),
        # Right side: 2D Hinge View
        html.Div([dcc.Graph(id='plot-hinge', style={'height': '70vh'})], style={'width': '40%', 'display': 'inline-block'})
    ])
])

@app.callback(
    [Output('plot-3d', 'figure'), Output('plot-hinge', 'figure')],
    Input('airfoil-dropdown', 'value')
)
def update_graphs(selected_airfoil):
    if selected_airfoil == 'ALL':
        mask = np.ones(len(airfoil_names), dtype=bool)
        title = "Global View"
    else:
        mask = airfoil_names == selected_airfoil
        title = f"{selected_airfoil}"
        
    X_subset = conditions[mask] # [Flap, Hinge, Re, Mach]
    y_subset = yield_pct[mask]
    
    local_yield = np.mean(y_subset)
    is_success = y_subset > 0

    # --- MACHINE LEARNING DECISION TREE ---
    if len(np.unique(is_success)) > 1: # Requires a mix of successes and failures to train
        # Depth 3 keeps the rules highly interpretable
        clf = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)
        clf.fit(X_subset, is_success)
        rules = export_text(clf, feature_names=['Flap_Angle', 'Hinge_X', 'Reynolds', 'Mach'])
        print(f"\n{'='*40}")
        print(f" FAILURE RULES FOR: {title}")
        print(f"{'='*40}")
        print(rules)
    else:
        print(f"\n[{title}] 100% Homogeneous (Yield: {local_yield:.1f}%). No boundaries to split.")

    # --- TRACE GENERATOR ---
    def make_traces(x_col, y_col, z_col=None):
        fails = ~is_success
        
        failed_marker = dict(symbol='x', color='red', size=4 if z_col is not None else 6, opacity=0.5)
        success_marker = dict(color=y_subset[is_success], colorscale='Viridis', size=5 if z_col is not None else 8, opacity=0.8)
        
        if z_col is not None:
            tr_fail = go.Scatter3d(x=X_subset[fails, x_col], y=X_subset[fails, y_col], z=X_subset[fails, z_col], mode='markers', marker=failed_marker, name="Failed")
            tr_succ = go.Scatter3d(x=X_subset[is_success, x_col], y=X_subset[is_success, y_col], z=X_subset[is_success, z_col], mode='markers', marker=success_marker, name="Success")
        else:
            tr_fail = go.Scatter(x=X_subset[fails, x_col], y=X_subset[fails, y_col], mode='markers', marker=failed_marker, name="Failed")
            tr_succ = go.Scatter(x=X_subset[is_success, x_col], y=X_subset[is_success, y_col], mode='markers', marker=success_marker, name="Success")
            
        return [tr_fail, tr_succ]

    # 1. 3D Plot (Mach vs Re vs Flap Angle)
    fig_3d = go.Figure(data=make_traces(x_col=3, y_col=2, z_col=0))
    fig_3d.update_layout(
        title=f"<b>{title}</b> (Yield: {local_yield:.1f}%)",
        scene=dict(
            xaxis_title="Mach",
            yaxis_title="Reynolds",
            yaxis_type="log",
            zaxis_title="Flap Angle"
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # 2. 2D Plot (Hinge Location vs Flap Angle)
    fig_hinge = go.Figure(data=make_traces(x_col=1, y_col=0))
    fig_hinge.update_layout(
        title="Hinge Location Impact",
        xaxis_title="Hinge Location (X/C)",
        yaxis_title="Flap Angle (deg)",
        template="plotly_white",
        margin=dict(l=0, r=0, b=0, t=40)
    )

    return fig_3d, fig_hinge

if __name__ == '__main__':
    app.run(debug=True)