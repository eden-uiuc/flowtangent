import numpy as np
import plotly.graph_objects as go

def create_fan_traces(
    x_center=0.0, 
    hub_radius=0.3, 
    tip_radius=1.2, 
    hub_length=0.8, 
    num_blades=22, 
    blade_chord=0.15,
    blade_thickness=0.02,
    color_hub='#cbd5e1', 
    color_blades='#94a3b8'
):
    """Generates 3D Plotly traces for a turbofan spinner and scimitar fan blades."""
    traces = []
    
    # -------------------------------------------------------------------------
    # 1. Parabolic Spinner (Hub)
    # -------------------------------------------------------------------------
    n_x = 20
    n_theta = 30
    x_nose = x_center - hub_length
    x_vals = np.linspace(x_nose, x_center, n_x)
    
    hub_x, hub_y, hub_z = [], [], []
    
    for x in x_vals:
        val = np.clip((x - x_nose) / hub_length, 0.0, 1.0)
        r = hub_radius * np.sqrt(val)
        for j in range(n_theta):
            theta = 2.0 * np.pi * j / n_theta
            hub_x.append(x)
            hub_y.append(r * np.cos(theta))
            hub_z.append(r * np.sin(theta))
            
    i_hub, j_hub, k_hub = [], [], []
    for i in range(n_x - 1):
        for j in range(n_theta):
            p1 = i * n_theta + j
            p2 = i * n_theta + (j + 1) % n_theta
            p3 = (i + 1) * n_theta + j
            p4 = (i + 1) * n_theta + (j + 1) % n_theta
            i_hub.extend([p1, p1])
            j_hub.extend([p2, p4])
            k_hub.extend([p4, p3])
            
    lighting_hub = dict(ambient=0.5, diffuse=0.8, roughness=0.2, specular=0.8)
    traces.append(go.Mesh3d(
        x=hub_x, y=hub_y, z=hub_z, i=i_hub, j=j_hub, k=k_hub,
        color=color_hub, flatshading=False, lighting=lighting_hub, name="Fan Spinner"
    ))

    # -------------------------------------------------------------------------
    # 2. Scimitar Fan Blades
    # -------------------------------------------------------------------------
    blade_x, blade_y, blade_z = [], [], []
    i_blade, j_blade, k_blade = [], [], []
    
    root_stagger = np.radians(60.0)
    tip_stagger = np.radians(25.0)
    
    num_sections = 6  # Number of spanwise segments to define the curve
    v_offset = 0      # Vertex counter
    
    for b in range(num_blades):
        blade_angle = 2.0 * np.pi * b / num_blades
        
        # 2a. Generate the spanwise cross-sections for this blade
        for s in range(num_sections):
            t = s / (num_sections - 1)  # Normalized span: 0.0 (root) to 1.0 (tip)
            r = hub_radius + t * (tip_radius - hub_radius)
            
            # --- THE SCIMITAR MATH ---
            # 1. Stagger washes out from root to tip
            stagger = root_stagger * (1.0 - t) + tip_stagger * t
            
            # 2. Flared Tip: Wide at root (1.0), narrows at mid (0.8), flares at tip (1.2)
            chord_mult = 1.0 - 0.5 * t + 0.8 * (t ** 3)
            c = blade_chord * chord_mult
            
            # 3. Axial Sweep: Sweeps forward slightly (negative X), then aggressively backward
            x_sweep = blade_chord * (-0.3 * t + 0.8 * (t ** 3))
            
            # 4. Tangential Lean: Bows into the direction of rotation
            lean_angle = 0.3 * (t ** 2)
            
            # Calculate the 4 corners in 2D section space
            dx = (c / 2.0) * np.cos(stagger)
            dy_local = (c / 2.0) * np.sin(stagger)
            dt = blade_thickness / 2.0
            
            corners_local = [
                (-dx + x_sweep,  dy_local + dt), # Leading Edge Top
                (-dx + x_sweep,  dy_local - dt), # Leading Edge Bottom
                ( dx + x_sweep, -dy_local - dt), # Trailing Edge Bottom
                ( dx + x_sweep, -dy_local + dt)  # Trailing Edge Top
            ]
            
            # Rotate into global 3D space
            total_angle = blade_angle + lean_angle
            for cx, cy in corners_local:
                Z = r * np.cos(total_angle) - cy * np.sin(total_angle)
                Y = r * np.sin(total_angle) + cy * np.cos(total_angle)
                blade_x.append(x_center + cx)
                blade_y.append(Y)
                blade_z.append(Z)
                
        # 2b. Stitch the sections together into faces
        # Root cap
        i_blade.extend([v_offset, v_offset])
        j_blade.extend([v_offset + 1, v_offset + 2])
        k_blade.extend([v_offset + 2, v_offset + 3])
        
        # Tip cap
        top = v_offset + 4 * (num_sections - 1)
        i_blade.extend([top, top])
        j_blade.extend([top + 2, top + 1])
        k_blade.extend([top + 3, top + 2])
        
        # Side walls
        for s in range(num_sections - 1):
            curr = v_offset + (s * 4)
            nxt = v_offset + ((s + 1) * 4)
            
            for side in range(4):
                p1 = curr + side
                p2 = curr + ((side + 1) % 4)
                p3 = nxt + side
                p4 = nxt + ((side + 1) % 4)
                
                i_blade.extend([p1, p1])
                j_blade.extend([p2, p4])
                k_blade.extend([p4, p3])
                
        v_offset += 4 * num_sections

    lighting_blades = dict(ambient=0.6, diffuse=0.7, roughness=0.3, specular=0.6)
    traces.append(go.Mesh3d(
        x=blade_x, y=blade_y, z=blade_z, i=i_blade, j=j_blade, k=k_blade,
        color=color_blades, flatshading=True, lighting=lighting_blades, name="Fan Blades"
    ))

    return traces

def generate_engine_mesh(station_params=None, sweep_angle_deg=180):
    print("--- STARTING ENGINE MESH GENERATION ---")
    traces = []
    
    # -------------------------------------------------------------------------
    # Mock Data: A standard high-bypass turbofan layout
    # -------------------------------------------------------------------------
    if station_params is None:
        station_params = [
            {'name': 'Inlet',  'x_start': 0.0, 'length': 1.0, 'r_out_start': 1.2, 'r_out_end': 1.2, 'r_in_start': 0.4, 'r_in_end': 0.5, 'color': '#3b82f6'}, # Cool blue
            {'name': 'Fan',    'x_start': 1.0, 'length': 0.5, 'r_out_start': 1.2, 'r_out_end': 1.2, 'r_in_start': 0.5, 'r_in_end': 0.5, 'color': '#60a5fa'},
            {'name': 'LPC',    'x_start': 1.5, 'length': 1.0, 'r_out_start': 1.2, 'r_out_end': 0.9, 'r_in_start': 0.5, 'r_in_end': 0.6, 'color': '#93c5fd'},
            {'name': 'HPC',    'x_start': 2.5, 'length': 1.5, 'r_out_start': 0.9, 'r_out_end': 0.7, 'r_in_start': 0.6, 'r_in_end': 0.6, 'color': '#f87171'}, # Warming up
            {'name': 'Burner', 'x_start': 4.0, 'length': 1.0, 'r_out_start': 0.7, 'r_out_end': 0.7, 'r_in_start': 0.6, 'r_in_end': 0.5, 'color': '#dc2626'}, # Hot red
            {'name': 'HPT',    'x_start': 5.0, 'length': 0.5, 'r_out_start': 0.7, 'r_out_end': 0.8, 'r_in_start': 0.5, 'r_in_end': 0.4, 'color': '#fb923c'}, # Cooling slightly
            {'name': 'LPT',    'x_start': 5.5, 'length': 1.0, 'r_out_start': 0.8, 'r_out_end': 0.9, 'r_in_start': 0.4, 'r_in_end': 0.3, 'color': '#fbbf24'},
            {'name': 'Nozzle', 'x_start': 6.5, 'length': 1.5, 'r_out_start': 0.9, 'r_out_end': 0.6, 'r_in_start': 0.3, 'r_in_end': 0.0, 'color': '#fcd34d'}
        ]

    # -------------------------------------------------------------------------
    # Helper: Annular Frustum Generator
    # -------------------------------------------------------------------------
    def create_annular_station(station, sweep_deg):
        x_start = float(station['x_start'])
        x_end = x_start + float(station['length'])
        
        r_out_start = float(station['r_out_start'])
        r_out_end = float(station['r_out_end'])
        r_in_start = float(station['r_in_start'])
        r_in_end = float(station['r_in_end'])
        
        color = station.get('color', '#9ca3af')
        name = station.get('name', 'Station')
        
        n_theta = 40  # Resolution around the circumference
        sweep_rad = np.radians(sweep_deg)
        theta_vals = np.linspace(3/2 * np.pi, 3/2 * np.pi + sweep_rad, n_theta)
        
        # We need a mesh for the outer shroud and a mesh for the inner hub
        station_traces = []
        
        # Sub-helper to generate a simple connecting cone/cylinder
        def build_shell(r_start, r_end, surface_name):
            x_pts, y_pts, z_pts = [], [], []
            
            # Start ring
            for theta in theta_vals:
                x_pts.append(x_start)
                y_pts.append(r_start * np.cos(theta))
                z_pts.append(r_start * np.sin(theta))
                
            # End ring
            for theta in theta_vals:
                x_pts.append(x_end)
                y_pts.append(r_end * np.cos(theta))
                z_pts.append(r_end * np.sin(theta))
                
            i_faces, j_faces, k_faces = [], [], []
            
            # Stitch the two rings together with triangles
            for j in range(n_theta - 1):
                p1 = j
                p2 = j + 1
                p3 = n_theta + j
                p4 = n_theta + (j + 1)
                
                i_faces.extend([p1, p1])
                j_faces.extend([p2, p4])
                k_faces.extend([p4, p3])
                
            lighting = dict(ambient=0.7, diffuse=0.8, roughness=0.5, specular=0.2)
            
            return go.Mesh3d(
                x=x_pts, y=y_pts, z=z_pts, 
                i=i_faces, j=j_faces, k=k_faces, 
                color=color, opacity=0.9, flatshading=False, 
                lighting=lighting, name=f"{name} ({surface_name})"
            )

        # Build Outer Shroud
        if r_out_start > 0 or r_out_end > 0:
            station_traces.append(build_shell(r_out_start, r_out_end, "Outer"))
            
        # Build Inner Hub (Spool/Centerbody)
        if r_in_start > 0 or r_in_end > 0:
            station_traces.append(build_shell(r_in_start, r_in_end, "Inner"))
            
        return station_traces

    # -------------------------------------------------------------------------
    # Assembly
    # -------------------------------------------------------------------------
    for stat in station_params:
        print(f" -> Building station: {stat['name']}")
        traces.extend(create_annular_station(stat, sweep_angle_deg))

        if stat['name'].lower() == 'fan':
            print("    -> Injecting 3D Fan Geometry")
            
            x_start = float(stat['x_start'])
            length = float(stat['length'])
            r_in = float(stat['r_in_start'])
            r_out = float(stat['r_out_start'])
            
            # Parametrically size the fan based on the station's dimensions
            fan_traces = create_fan_traces(
                x_center=x_start + (length * 0.25), # Position the blades slightly forward in the duct
                hub_radius=r_in,
                tip_radius=r_out,
                hub_length=length * 1.5, # Make the spinner protrude into the Inlet station
                num_blades=24,
                blade_chord=length * 0.75, # Scale chord length based on station length
                blade_thickness=length * 0.05
            )
            traces.extend(fan_traces)


    fig = go.Figure(data=traces)
    
    no_axis = dict(visible=False)
    fig.update_layout(
        uirevision='preserve_ui_state', 
        scene=dict(
            aspectmode='data', 
            xaxis=no_axis, yaxis=no_axis, zaxis=no_axis, 
            camera=dict(eye=dict(x=-1.5, y=-2.0, z=1.5))
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    
    print("--- ENGINE MESH GENERATION COMPLETE ---")
    return fig

if __name__ == "__main__":
    fig = generate_engine_mesh()
    fig.show()