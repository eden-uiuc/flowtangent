import uuid
import tempfile

import numpy as np
from nicegui import ui, app

from utils.state import app_state
from components.context_menus import setup_context_menu

# -------------------------------------------------------------------------
# Mesh Generators
# -------------------------------------------------------------------------
def generate_shroud(x_start=-0.5, length=1.5, r_start=1.25, r_end=1.25, sweep_deg=180):
    """Generates an open-ended, cutaway shell for engine stations."""
    vertices = []
    triangles = []
    n_theta = 30
    
    # Sweep from the top of the engine (+Y) around to the bottom (-Y)
    sweep_rad = np.radians(sweep_deg)
    theta_vals = np.linspace(3/2 * np.pi, 3/2 * np.pi + sweep_rad, n_theta)
    
    x_end = x_start + length
    
    # Generate the vertices for the front and back arcs
    for theta in theta_vals:
        vertices.append((x_start, r_start * np.cos(theta), r_start * np.sin(theta)))
    for theta in theta_vals:
        vertices.append((x_end, r_end * np.cos(theta), r_end * np.sin(theta)))
        
    # Stitch the faces together
    for i in range(n_theta - 1):
        p1 = i
        p2 = i + 1
        p3 = n_theta + i
        p4 = n_theta + i + 1
        triangles.extend([(p1, p3, p4), (p1, p4, p2)])
        
    # Format as ASCII STL
    stl_lines = ["solid shroud"]
    vertices = np.array(vertices)
    for t in triangles:
        v1, v2, v3 = vertices[t[0]], vertices[t[1]], vertices[t[2]]
        # Face normal
        n = np.cross(v2 - v1, v3 - v1)
        norm = np.linalg.norm(n)
        n = n / norm if norm > 1e-12 else np.array([0.0, 0.0, 0.0])
        
        stl_lines.append(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}")
        stl_lines.append("    outer loop")
        stl_lines.append(f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}")
        stl_lines.append(f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}")
        stl_lines.append(f"      vertex {v3[0]:.6e} {v3[1]:.6e} {v3[2]:.6e}")
        stl_lines.append("    endloop")
        stl_lines.append("  endfacet")
        
    stl_lines.append("endsolid shroud")
    return "\n".join(stl_lines)

def generate_fan(
    x_center=0.0, hub_radius=0.3, tip_radius=1.2, hub_length=0.8, 
    num_blades=24, blade_chord=0.45, blade_thickness=0.02,
    root_stagger=np.radians(35.0), tip_stagger=np.radians(70.0)
):
    """Generates the raw text for an ASCII .stl file containing the hub and blades."""
    vertices = []
    triangles = [] # Will hold tuples of 3 vertex indices
    
    # --- HUB GENERATION ---
    n_x, n_theta = 20, 30
    x_nose = x_center - hub_length
    x_vals = np.linspace(x_nose, x_center, n_x)
    
    # Generate Hub Vertices
    for x in x_vals:
        val = np.clip((x - x_nose) / hub_length, 0.0, 1.0)
        r = hub_radius * np.sqrt(val)
        for j in range(n_theta):
            theta = 2.0 * np.pi * j / n_theta
            
            # AXIAL mapped to X, RADIAL mapped to Y and Z
            vertices.append((x, r * np.cos(theta), r * np.sin(theta)))
            
    # Generate Hub Triangles
    for i in range(n_x - 1):
        for j in range(n_theta):
            p1 = (i * n_theta + j)
            p2 = (i * n_theta + (j + 1) % n_theta)
            p3 = ((i + 1) * n_theta + j)
            p4 = ((i + 1) * n_theta + (j + 1) % n_theta)
            
            # SWAPPED VERTEX ORDER to flip the normals outward!
            triangles.extend([(p1, p4, p3), (p1, p2, p4)])
            
    v_idx = len(vertices)
    
    # --- SCIMITAR BLADE GENERATION ---
    num_sections = 12
    
    for b in range(num_blades):
        blade_angle = 2.0 * np.pi * b / num_blades
        
        for s in range(num_sections):
            t = s / (num_sections - 1) # Normalized span (0 to 1)
            r = hub_radius + t * (tip_radius - hub_radius)
            
            # 1. Wash out the stagger from root to tip
            stagger = root_stagger * (1.0 - t) + tip_stagger * t
            
            # 2. Aggressive Chord Flare: 
            # 1.0x at root, dips to 0.8x at mid-span, flares to 1.4x at tip
            chord_mult = 1.0 - 0.4 * t + 0.8 * (t ** 2)
            c = blade_chord * chord_mult
            
            # 3. Scimitar Sweep: sweeps forward, then hooks back at the tip
            x_sweep = blade_chord * (-0.2 * t + 0.6 * (t ** 3))
            
            # 4. Tangential Lean (Bowed slightly into rotation)
            lean_angle = 0.15 * (t ** 2)
            
            dx = (c / 2.0) * np.cos(stagger)
            dy_local = (c / 2.0) * np.sin(stagger)
            dt = blade_thickness / 2.0
            
            corners = [
                (-dx + x_sweep,  dy_local + dt),
                (-dx + x_sweep,  dy_local - dt),
                ( dx + x_sweep, -dy_local - dt),
                ( dx + x_sweep, -dy_local + dt) 
            ]
            
            total_angle = blade_angle + lean_angle
            for cx, cy in corners:
                axial_coord = x_center + cx
                y_coord = r * np.cos(total_angle) - cy * np.sin(total_angle)
                z_coord = r * np.sin(total_angle) + cy * np.cos(total_angle)
                
                vertices.append((axial_coord, y_coord, z_coord))
                
        # Stitch Blade Side Sections
        for s in range(num_sections - 1):
            curr = v_idx + (s * 4)
            nxt = v_idx + ((s + 1) * 4)
            for side in range(4):
                p1 = curr + side
                p2 = curr + ((side + 1) % 4)
                p3 = nxt + side
                p4 = nxt + ((side + 1) % 4)
                # Split CCW quad (p1, p2, p4, p3) into two triangles
                triangles.extend([(p1, p2, p4), (p1, p4, p3)])
                
        # Caps
        triangles.extend([(v_idx, v_idx+3, v_idx+2), (v_idx, v_idx+2, v_idx+1)]) # Root
        top = v_idx + 4 * (num_sections - 1)
        triangles.extend([(top, top+1, top+2), (top, top+2, top+3)]) # Tip
        
        v_idx += 4 * num_sections

    # --- FORMAT AS ASCII STL ---
    stl_lines = ["solid fan"]
    vertices = np.array(vertices)
    
    for t in triangles:
        v1, v2, v3 = vertices[t[0]], vertices[t[1]], vertices[t[2]]
        
        # Calculate face normal
        n = np.cross(v2 - v1, v3 - v1)
        norm = np.linalg.norm(n)
        n = n / norm if norm > 1e-12 else np.array([0.0, 0.0, 0.0])
        
        stl_lines.append(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}")
        stl_lines.append("    outer loop")
        stl_lines.append(f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}")
        stl_lines.append(f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}")
        stl_lines.append(f"      vertex {v3[0]:.6e} {v3[1]:.6e} {v3[2]:.6e}")
        stl_lines.append("    endloop")
        stl_lines.append("  endfacet")
        
    stl_lines.append("endsolid fan")
    return "\n".join(stl_lines)

def generate_rotors(
    x_start=1.0, length=2.0, r_hub=0.4, r_tip=[1.2]*4, 
    num_stages=4, num_blades=36, chord=0.15, thickness=0.02
):
    """Generates two STLs: one for rotating stages, one for static stages."""
    
    def build_blade_stage(x_center, stagger, r_tip_stg):
        verts, tris = [], []
        v_idx = 0
        
        for b in range(num_blades):
            blade_angle = 2.0 * np.pi * b / num_blades
            
            # Straight blade: just two sections (root and tip)
            for r in [r_hub, r_tip_stg]:
                dx = (chord / 2.0) * np.cos(stagger)
                dy_local = (chord / 2.0) * np.sin(stagger)
                dt = thickness / 2.0
                
                # 4 corners of the rectangular cross-section
                corners = [
                    (-dx,  dy_local + dt), # 0: Top Left (Leading Edge)
                    (-dx,  dy_local - dt), # 1: Bottom Left
                    ( dx, -dy_local - dt), # 2: Bottom Right (Trailing Edge)
                    ( dx, -dy_local + dt)  # 3: Top Right
                ]
                
                for cx, cy in corners:
                    X_3d = x_center + cx
                    Y_3d = r * np.cos(blade_angle) - cy * np.sin(blade_angle)
                    Z_3d = r * np.sin(blade_angle) + cy * np.cos(blade_angle)
                    verts.append((X_3d, Y_3d, Z_3d))
            
            # Stitch the 3D box (8 vertices per blade)
            # Root: 0, 1, 2, 3 | Tip: 4, 5, 6, 7
            
            # Side Walls (CCW winding for outward normals)
            tris.extend([
                (v_idx+0, v_idx+4, v_idx+5), (v_idx+0, v_idx+5, v_idx+1), # Leading Edge
                (v_idx+1, v_idx+5, v_idx+6), (v_idx+1, v_idx+6, v_idx+2), # Suction Side
                (v_idx+2, v_idx+6, v_idx+7), (v_idx+2, v_idx+7, v_idx+3), # Trailing Edge
                (v_idx+3, v_idx+7, v_idx+4), (v_idx+3, v_idx+4, v_idx+0)  # Pressure Side
            ])
            # Caps
            tris.extend([
                (v_idx+0, v_idx+3, v_idx+2), (v_idx+0, v_idx+2, v_idx+1), # Root Cap
                (v_idx+4, v_idx+5, v_idx+6), (v_idx+4, v_idx+6, v_idx+7)  # Tip Cap
            ])
            
            v_idx += 8
            
        return verts, tris

    # --- Generate Stages ---
    rotor_verts, rotor_tris = [], []
    stator_verts, stator_tris = [], []
    
    stage_length = length / num_stages
    
    for stage in range(num_stages):
        # 1. Rotor (Front half of the stage length)
        r_x = x_start + stage_length * (stage + 0.25)
        r_stagger = np.radians(40.0) # Pitched into flow
        rv, rt = build_blade_stage(r_x, r_stagger, r_tip[stage])
        
        # Offset triangle indices by the current number of vertices
        r_offset = len(rotor_verts)
        rotor_verts.extend(rv)
        rotor_tris.extend([(t[0]+r_offset, t[1]+r_offset, t[2]+r_offset) for t in rt])
        
        # 2. Stator (Back half of the stage length)
        s_x = x_start + stage_length * (stage + 0.75)
        s_stagger = np.radians(-20.0) # Flow recovery pitch
        sv, st = build_blade_stage(s_x, s_stagger, r_tip[stage])
        
        s_offset = len(stator_verts)
        stator_verts.extend(sv)
        stator_tris.extend([(t[0]+s_offset, t[1]+s_offset, t[2]+s_offset) for t in st])

    # --- Helper to format STL string ---
    def write_stl(name, v_array, t_array):
        v_np = np.array(v_array)
        lines = [f"solid {name}"]
        for t in t_array:
            v1, v2, v3 = v_np[t[0]], v_np[t[1]], v_np[t[2]]
            n = np.cross(v2 - v1, v3 - v1)
            norm = np.linalg.norm(n)
            n = n / norm if norm > 1e-12 else np.array([0.0, 0.0, 0.0])
            
            lines.append(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}")
            lines.append("    outer loop")
            lines.append(f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}")
            lines.append(f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}")
            lines.append(f"      vertex {v3[0]:.6e} {v3[1]:.6e} {v3[2]:.6e}")
            lines.append("    endloop")
            lines.append("  endfacet")
        lines.append(f"endsolid {name}")

        return "\n".join(lines)
    
    rotor_stl_string = write_stl("compressor_rotors", rotor_verts, rotor_tris)
    stator_stl_string = write_stl("compressor_stators", stator_verts, stator_tris)
    
    return rotor_stl_string, stator_stl_string

def engine_ui():

    station_geometry = [
        {'name': 'Inlet',  'x_start': 0.0, 'length': 1.0, 'r_out_start': 1.2, 'r_out_end': 1.2, 'r_in_start': 0.4, 'r_in_end': 0.5, 'color': '#3b82f6'}, # Cool blue
        {'name': 'Fan',    'x_start': 1.0, 'length': 0.5, 'r_out_start': 1.2, 'r_out_end': 1.2, 'r_in_start': 0.5, 'r_in_end': 0.5, 'color': '#60a5fa'},
        {'name': 'LPC',    'x_start': 1.5, 'length': 1.0, 'r_out_start': 1.2, 'r_out_end': 0.9, 'r_in_start': 0.5, 'r_in_end': 0.6, 'color': '#93c5fd'},
        {'name': 'HPC',    'x_start': 2.5, 'length': 1.5, 'r_out_start': 0.9, 'r_out_end': 0.7, 'r_in_start': 0.6, 'r_in_end': 0.6, 'color': '#f87171'}, # Warming up
        {'name': 'Burner', 'x_start': 4.0, 'length': 1.0, 'r_out_start': 0.7, 'r_out_end': 0.7, 'r_in_start': 0.6, 'r_in_end': 0.5, 'color': '#dc2626'}, # Hot red
        {'name': 'HPT',    'x_start': 5.0, 'length': 0.5, 'r_out_start': 0.7, 'r_out_end': 0.8, 'r_in_start': 0.5, 'r_in_end': 0.4, 'color': '#fb923c'}, # Cooling slightly
        {'name': 'LPT',    'x_start': 5.5, 'length': 1.0, 'r_out_start': 0.8, 'r_out_end': 0.9, 'r_in_start': 0.4, 'r_in_end': 0.3, 'color': '#fbbf24'},
        {'name': 'Nozzle', 'x_start': 6.5, 'length': 1.5, 'r_out_start': 0.9, 'r_out_end': 0.6, 'r_in_start': 0.3, 'r_in_end': 0.0, 'color': '#fcd34d'}
    ]

    station_centers = {s['name'].lower(): s['x_start'] + s['length'] / 1.8 for s in station_geometry}
    station_centers['c_nozz'] = 7.33
    station_centers['f_nozz'] = 7.33

    def pan_to_station(target_x, zoom_z=0.):
        scene.move_camera(
            x=target_x, y=-1.5, z=zoom_z, 
            look_at_x=target_x, look_at_y=0, look_at_z=0, 
            duration=0.8
        )

    with ui.column().classes('w-full h-full p-0 gap-0'):
        # --- 1. STATE DATA ---
        engine_state = app_state['engine']
        
        def get_selected_station():
            sel_id = engine_state['selected_id']
            stats = engine_state['stations']
            if sel_id in stats:
                return stats[sel_id]
            return None
        
        def select_node(e):
            if e.value:  
                engine_state['selected_id'] = e.value
            if e.value in station_centers:
                pan_to_station(station_centers[e.value])

        # --- 5. DYNAMIC UI COMPONENTS ---
        @ui.refreshable
        def engine_tree():
            
            # Build nested tree data
            tree_data = [
                {'id': 'inlet', 'label': 'Inlet Nozzle', 'icon': 'tornado', 'children': []},
                {'id': 'fan', 'label': 'Fan', 'icon': 'adjust', 'children': []},
                {'id': 'cat_comp', 'label': 'Compressors', 'icon': 'cyclone', 'children': [
                    {'id': 'lpc', 'label': 'Low Pressure Stage'},
                    {'id': 'hpc', 'label': 'High Pressure Stage'},
                ]},
                {'id': 'burner', 'label': 'Combustion Chamber', 'icon': 'local_fire_department', 'children': []},
                {'id': 'cat_turb', 'label': 'Turbines', 'icon': 'settings', 'children': [
                    {'id': 'hpt', 'label': 'High Pressure Stage'},
                    {'id': 'lpt', 'label': 'Low Pressure Stage'},
                ]},
                {'id': 'cat_exit', 'label': 'Exhaust', 'icon': 'air', 'children': [
                    {'id': 'c_nozz', 'label': 'Core Flow'},
                    {'id': 'f_nozz', 'label': 'Fan Bypass'},
                ]},
            ]
                
            # The tree draws once and handles its own internal visual state
            ui.tree(tree_data, on_select=select_node, tick_strategy='none').expand()

        # --- 6. MAIN LAYOUT (SPA COMPLIANT) ---
        # 1. Evaluate True Colors based on state
        is_dark = app_state.get('is_dark', False)
        
        sidebar_bg = 'bg-neutral-900' if is_dark else 'bg-gray-200'
        sidebar_border = 'border-neutral-800' if is_dark else 'border-gray-200'
        
        canvas_bg = 'bg-black text-gray-200' if is_dark else 'bg-white text-gray-800'

        # Dialogs
        with ui.dialog() as payload_dialog, ui.card().classes('w-full max-w-2xl'):
            ui.label('RCAIDE JSON Payload').classes('text-xl font-bold')
            json_display = ui.code('', language='json').classes('w-full')
            with ui.row().classes('w-full justify-end mt-4'):
                ui.button('Close', on_click=payload_dialog.close, color='gray')
                ui.button('Send to Solver', color='blue') 

        # SPLITTER 1: Left Panel
        # Added limits=(min, max) to physically prevent the Quasar collapse bug
        with ui.splitter(value=350, limits=(200, 800)).classes('w-full h-full').props('unit="px"') as outer_split:
            
            setup_context_menu()

            with outer_split.before:
                # LEFT PANEL CONTENT
                with ui.column().classes(f'w-full h-full p-4 border-r overflow-y-auto {sidebar_bg} {sidebar_border}'):
                    with ui.row().classes('w-full gap-2 flex-wrap'):
                        ui.button('Side View', on_click=lambda: scene.move_camera(
                                    x=4.0, y=-4.0, z=0.0, look_at_x=4.0, look_at_y=0, look_at_z=0, duration=1.0)
                                ).classes('flex-grow text-xs')
                        ui.button('Isometric', on_click=lambda: scene.move_camera(
                                    x=-0.3, y=-2.0, z=2.2, look_at_x=2.0, look_at_y=0, look_at_z=0, duration=1.0)
                                ).classes('flex-grow text-xs')
                    ui.label('Vehicle Tree').classes('text-lg font-bold mb-2')
                    engine_tree()  
                    ui.separator().classes('my-4 opacity-50')
                    # attribute_sliders() 
            
            with outer_split.after:
                # SPLITTER 2: Right Panel
                # reverse=True makes the 'value' and 'limits' apply to the RIGHT panel
                with ui.splitter(value=450, limits=(300, 800), reverse=True).classes('w-full h-full').props('unit="px"') as inner_split:
                    
                    with inner_split.before:
                        # MAIN 3D CANVAS
                        with ui.column().classes(f'w-full h-full flex justify-center items-center relative {canvas_bg}'):
                            # -------------------------------------------------------------------------
                            # 1. Update Internals to Align with New Station Coordinates
                            # -------------------------------------------------------------------------
                            # Fan (Centered in the 1.0 to 1.5 station)
                            fan_data = generate_fan(
                                x_center=1.25, hub_radius=0.3, tip_radius=1.05, hub_length=0.8,
                                num_blades=24, blade_chord=0.45, root_stagger=np.radians(35.0), tip_stagger=np.radians(70.0)
                            )
                            temp_fan = tempfile.NamedTemporaryFile(delete=False, suffix=".stl")
                            temp_fan.write(fan_data.encode('utf-8'))
                            temp_fan.close()
                            fan_url = f'/fan_{uuid.uuid4().hex}.stl'
                            app.add_static_file(local_file=temp_fan.name, url_path=fan_url)

                            # Compressor (Spans LPC and HPC: X=1.5 to 4.0)
                            comp_rotor_data, comp_stator_data = generate_rotors(
                                x_start=1.5, length=2.5, r_hub=0.1, r_tip=[0.45, 0.5, 0.55, 0.55, 0.55, 0.55], 
                                num_stages=6, num_blades=36, chord=0.12, thickness=0.015
                            )
                            temp_cr = tempfile.NamedTemporaryFile(delete=False, suffix=".stl")
                            temp_cr.write(comp_rotor_data.encode('utf-8'))
                            temp_cr.close()
                            comp_rotor_url = f'/comp_r_{uuid.uuid4().hex}.stl'
                            app.add_static_file(local_file=temp_cr.name, url_path=comp_rotor_url)

                            temp_cs = tempfile.NamedTemporaryFile(delete=False, suffix=".stl")
                            temp_cs.write(comp_stator_data.encode('utf-8'))
                            temp_cs.close()
                            comp_stator_url = f'/comp_s_{uuid.uuid4().hex}.stl'
                            app.add_static_file(local_file=temp_cs.name, url_path=comp_stator_url)

                            # Turbine (Spans HPT and LPT: X=5.0 to 6.5)
                            turb_rotor_data, turb_stator_data = generate_rotors(
                                x_start=5.0, length=1.5, r_hub=0.1, r_tip=[0.4, 0.35, 0.3], 
                                num_stages=3, num_blades=40, chord=0.15, thickness=0.015
                            )
                            temp_tr = tempfile.NamedTemporaryFile(delete=False, suffix=".stl")
                            temp_tr.write(turb_rotor_data.encode('utf-8'))
                            temp_tr.close()
                            turb_rotor_url = f'/turb_r_{uuid.uuid4().hex}.stl'
                            app.add_static_file(local_file=temp_tr.name, url_path=turb_rotor_url)

                            temp_ts = tempfile.NamedTemporaryFile(delete=False, suffix=".stl")
                            temp_ts.write(turb_stator_data.encode('utf-8'))
                            temp_ts.close()
                            turb_stator_url = f'/turb_s_{uuid.uuid4().hex}.stl'
                            app.add_static_file(local_file=temp_ts.name, url_path=turb_stator_url)

                            # -------------------------------------------------------------------------
                            # 2. Generate the Thermodynamic Station Shrouds
                            # -------------------------------------------------------------------------

                            station_urls = []
                            for stat in station_geometry:
                                # Generate Outer Flow Boundary
                                outer_data = generate_shroud(
                                    x_start=stat['x_start'], length=stat['length'], 
                                    r_start=stat['r_out_start'], r_end=stat['r_out_end'], sweep_deg=180
                                )
                                temp_o = tempfile.NamedTemporaryFile(delete=False, suffix=".stl")
                                temp_o.write(outer_data.encode('utf-8'))
                                temp_o.close()
                                url_o = f"/{stat['name']}_out_{uuid.uuid4().hex}.stl"
                                app.add_static_file(local_file=temp_o.name, url_path=url_o)

                                # Generate Inner Flow Boundary
                                inner_data = generate_shroud(
                                    x_start=stat['x_start'], length=stat['length'], 
                                    r_start=stat['r_in_start'], r_end=stat['r_in_end'], sweep_deg=180
                                )
                                temp_i = tempfile.NamedTemporaryFile(delete=False, suffix=".stl")
                                temp_i.write(inner_data.encode('utf-8'))
                                temp_i.close()
                                url_i = f"/{stat['name']}_in_{uuid.uuid4().hex}.stl"
                                app.add_static_file(local_file=temp_i.name, url_path=url_i)

                                station_urls.append((url_o, url_i, stat['color']))


                            def pan_to_station(target_x, zoom_z=0.):
                                scene.move_camera(
                                    x=target_x, y=-1.5, z=zoom_z, 
                                    look_at_x=target_x, look_at_y=0, look_at_z=0, 
                                    duration=0.8
                                )

                            scene_container = ui.element('div').classes('w-full h-full')
                            with scene_container:
                                scene = ui.scene(grid=False, background_color='transparent').classes('w-full h-full')
                                
                                with scene:
                                    # Initial boot-up camera position matches the new Wide Shot (-12.0 Z)
                                    scene.move_camera(x=-0.2, y=-2.5, z=2.5, look_at_x=3.0, look_at_y=0, look_at_z=0)

                                    # Mount the Thermodynamic Color Stations
                                    for url_o, url_i, color in station_urls:
                                        scene.stl(url_o).material(color=color, opacity=0.7)
                                        scene.stl(url_i).material(color=color, opacity=0.7)
                                    
                                    # Mount STATIC Internals
                                    scene.stl(comp_stator_url).material(color='#475569') 
                                    scene.stl(turb_stator_url).material(color='#475569') 
                                    
                                    # Mount ROTATING Internals
                                    rotor_group = scene.group()
                                    with rotor_group:
                                        scene.stl(fan_url).material(color='#94a3b8')
                                        scene.stl(comp_rotor_url).material(color='#94a3b8')
                                        scene.stl(turb_rotor_url).material(color='#94a3b8')
                                        
                                        # Central Turboshaft
                                        scene.cylinder(top_radius=0.1, bottom_radius=0.1, height=5.0, radial_segments=16) \
                                            .material(color='#94a3b8') \
                                            .move(x=4.0, y=0, z=0) \
                                            .rotate(0, 0, np.pi/2)
                                    
                                    anim_state = {'angle': 0.0}
                                    def animate_spin():
                                        anim_state['angle'] += 0.05 
                                        rotor_group.rotate(anim_state['angle'], 0, 0)
                                        
                                    ui.timer(0.05, animate_spin)
                                
                                with ui.row().classes('absolute top-0 left-0 z-40 bg-slate-900/90 text-white p-4 shadow-2xl backdrop-blur-md border-b border-slate-700 w-full justify-around items-center'):
                                    def telemetry_block(label, key, stage, formatter=None,):
                                        with ui.column().classes('items-center gap-1'):
                                            ui.label(label).classes('text-[10px] text-slate-400 font-bold uppercase tracking-wider')
                                            ui.label().bind_text_from(app_state['simulator'], key, backward=formatter).classes('font-mono text-lg text-slate-100')

                                    with ui.column().classes('items-center gap-1'):
                                            ui.label(engine_state['selected_id'].title()).classes('text-[10px] text-slate-400 font-bold uppercase tracking-wider')
                                            ui.label("Intake").classes('font-bold text-lg text-green-400')
                                    telemetry_block('Alt (ft)', 'current_alt', lambda a: f"{a:,.0f}")
                                    telemetry_block('Mach', 'current_mach', lambda m: f"{m:.3f}")
                                    ui.element('div').classes('w-px h-8 bg-slate-700 mx-2')
                                    telemetry_block('Alpha', 'alpha', lambda a: f"{a:.1f}°")
                                    telemetry_block('CL', 'cl', lambda c: f"{c:.3f}")
                                    telemetry_block('CD', 'cd', lambda c: f"{c:.4f}")
                                    telemetry_block('L/D', 'l_d', lambda l: f"{l:.1f}")
                                    ui.element('div').classes('w-px h-8 bg-slate-700 mx-2')
                                    telemetry_block('Throttle', 'throttle', lambda t: f"{t:.1f}%")
                                    telemetry_block('Fuel Burn', 'fuel_burn', lambda f: f"{f:,.0f} lb/hr")
                                
                                with ui.row().classes('absolute bottom-0 left-0 z-40 bg-slate-900/90 text-white p-4 shadow-2xl backdrop-blur-md border-b border-slate-700 w-full justify-around items-center'):

                                    with ui.column().classes('items-center gap-1'):
                                            ui.label(engine_state['selected_id'].title()).classes('text-[10px] text-slate-400 font-bold uppercase tracking-wider')
                                            ui.label("Exit").classes('font-bold text-lg text-green-400')
                                    telemetry_block('Alt (ft)', 'current_alt', lambda a: f"{a:,.0f}")
                                    telemetry_block('Mach', 'current_mach', lambda m: f"{m:.3f}")
                                    ui.element('div').classes('w-px h-8 bg-slate-700 mx-2')
                                    telemetry_block('Alpha', 'alpha', lambda a: f"{a:.1f}°")
                                    telemetry_block('CL', 'cl', lambda c: f"{c:.3f}")
                                    telemetry_block('CD', 'cd', lambda c: f"{c:.4f}")
                                    telemetry_block('L/D', 'l_d', lambda l: f"{l:.1f}")
                                    ui.element('div').classes('w-px h-8 bg-slate-700 mx-2')
                                    telemetry_block('Throttle', 'throttle', lambda t: f"{t:.1f}%")
                                    telemetry_block('Fuel Burn', 'fuel_burn', lambda f: f"{f:,.0f} lb/hr")
                            ui.run_javascript(f'''
                                const target = getElement({scene_container.id});
                                if (target) {{
                                    new ResizeObserver(() => {{
                                        window.dispatchEvent(new Event('resize')); 
                                    }}).observe(target);
                                }}
                            ''')

                    with inner_split.after:
                        # RIGHT PANEL CONTENT
                        with ui.column().classes(f'w-full h-full p-4 border-l overflow-y-auto gap-0 {sidebar_bg} {sidebar_border}'):
                            ui.label('Design Flight Regime').classes('text-lg font-bold mb-2')
                            
                            with ui.row().classes('w-full gap-2'):
                                ui.number('Mach', value=engine_state['design']['mach'], step=0.05).bind_value(engine_state, 'mach').props('dense').classes('flex-1')
                                ui.number('Alt. (km)', value=engine_state['design']['alt'], step=0.5).bind_value(engine_state, 'alpha').props('dense').classes('flex-1')
                                ui.number('Target Thrust (kN)', value=engine_state['design']['thrust'], step=1.0).bind_value(engine_state, 'beta').props('dense').classes('flex-1')                            
                            
                            ui.separator().classes('my-4 opacity-50')
                            run_button = ui.button('Run Design Study', on_click=lambda: None, color='blue').classes('w-full')

                            with ui.row().classes('w-full items-center justify-center mt-4 hidden') as loading_indicator:
                                ui.spinner('orbit', size='md', color='blue')
                                loading_label = ui.label('Initializing...').classes('ml-2 text-gray-500 font-medium')

                            ui.separator().classes('my-4 opacity-50')
                            
                            # Analysis Results
                            ui.label('Analysis Results').classes('text-lg font-bold mt-2 mb-2')
                            
                            notice_bg = 'bg-neutral-900 border-neutral-800' if is_dark else 'bg-gray-50 border-gray-200'
                            no_results_notice = ui.column().classes(f'w-full border rounded-lg p-4 items-center justify-center {notice_bg}')
                            with no_results_notice:
                                ui.icon('science', size='2rem', color='gray')
                                ui.label('No analysis results available.').classes('text-sm font-bold text-center mt-2')

                            results_container = ui.column().classes('w-full gap-0 hidden')
                            with results_container:
                                ui.label('Aerodynamics').classes('text-sm font-bold mt-2 mb-1')
                                result_columns = [
                                    {'name': 'coeff', 'label': 'Coeff', 'field': 'coeff', 'align': 'left'},
                                    {'name': 'base', 'label': 'Base', 'field': 'base', 'align': 'right'},
                                    {'name': 'd_alpha', 'label': '∂/∂α', 'field': 'd_alpha', 'align': 'right'}
                                ]
                                results_table = ui.table(columns=result_columns, rows=[], row_key='coeff').classes('w-full').props('dense flat bordered')
                                
                            ui.separator().classes('my-6 opacity-50')
                            
                            # RCAIDE Assistant Chat
                            with ui.row().classes('w-full items-center justify-between mb-2'):
                                ui.label('RCAIDE Assistant').classes('text-lg font-bold')
                                ui.icon('psychology', size='sm', color='primary')
                            
                            chat_border = 'border-neutral-800' if is_dark else 'border-gray-300'
                            with ui.column().classes(f'w-full h-64 border rounded-lg flex-nowrap gap-0 overflow-hidden bg-transparent shrink-0 {chat_border}'):
                                with ui.column().classes('w-full flex-grow p-3 overflow-y-auto gap-2'):
                                    ui.chat_message('Hello! I am your RCAIDE agent.', name='RCAIDE', stamp='System Ready').props('bg-color=primary text-color=white')
                                    
                                with ui.row().classes(f'w-full shrink-0 p-2 border-t items-center flex-nowrap bg-transparent {chat_border}'):
                                    chat_input = ui.input(placeholder='Ask RCAIDE...').props('dense outlined').classes('flex-grow text-sm')
                                    ui.button(icon='send', color='primary').props('round flat dense').classes('ml-2')