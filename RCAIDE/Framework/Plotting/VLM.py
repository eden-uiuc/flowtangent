import numpy as np
import jax.numpy as jnp
import plotly.graph_objects as go

def plot_vlm_panels(VD, panel_values=None, title="VLM Aerodynamic Distribution"):
    """
    Plots a 3D interactive mesh of the VLM panels with an optional heatmap.

    Args:
        VD: VortexDistribution object containing the VLM panel vertices and normals.
        panel_values: ndarray of shape (N_panels,)
                      representing the heatmap value (e.g., Cp, Gamma) for each panel.
    """
    panel_vertices = np.asarray(VD.panel_vertices)

    if panel_values is not None:
        panel_values = np.asarray(panel_values)

    x, y, z = [], [], []
    i_idx, j_idx, k_idx = [], [], []
    facecolor_intensities = []

    v_count = 0
    # Iterate through each panel to build the vertex and triangle arrays
    for idx, panel in enumerate(panel_vertices):
        # Flatten the coordinates for Plotly
        x.extend(panel[:, 0])
        y.extend(panel[:, 1])
        z.extend(panel[:, 2])

        # Split the quad into Triangle 1 (Vertices 0, 1, 2)
        i_idx.append(v_count)
        j_idx.append(v_count + 1)
        k_idx.append(v_count + 2)

        # Split the quad into Triangle 2 (Vertices 0, 2, 3)
        i_idx.append(v_count)
        j_idx.append(v_count + 2)
        k_idx.append(v_count + 3)

        if panel_values is not None:
            # Both triangles making up this panel get the same intensity value
            facecolor_intensities.extend([panel_values[idx], panel_values[idx]])

        # Increment vertex counter by 4 for the next panel
        v_count += 4

    # Build the 3D Mesh
    fig = go.Figure(data=[
        go.Mesh3d(
            x=x, y=y, z=z,
            i=i_idx, j=j_idx, k=k_idx,
            intensity=facecolor_intensities if panel_values is not None else None,
            intensitymode='cell',
            colorscale='Plasma',  # Deep purple to bright orange/yellow
            color='black',  # Fallback color if no panel_values are provided
            showscale=panel_values is not None,
            flatshading=True,
            name='VLM Mesh',
            hovertemplate='Value: %{intensity:.5f}<extra></extra>' if panel_values is not None else None
        )
    ])

    # Define the isometric camera
    isometric_camera = dict(
        up=dict(x=0, y=0, z=1),  # Z is pointing up
        center=dict(x=0, y=0, z=0),  # Look at the origin
        eye=dict(x=1.5, y=-1.5, z=1.5),  # Positioned diagonally (Rear-Right-Up)
        projection=dict(type='orthographic')  # Removes perspective distortion for true isometric
    )

    # Apply the styling and camera to the layout
    fig.update_layout(
        title=title,
        template='plotly_white',  # Stark white background with black text/axes
        scene=dict(
            aspectmode='data',  # Locks true 1:1:1 geometry proportions
            camera=isometric_camera,
            xaxis=dict(title='X (Streamwise)', showbackground=False),
            yaxis=dict(title='Y (Spanwise)', showbackground=False),
            zaxis=dict(title='Z (Vertical)', showbackground=False)
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    return fig