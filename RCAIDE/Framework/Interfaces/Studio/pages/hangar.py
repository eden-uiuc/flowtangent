from nicegui import ui, Client
import numpy as np
import plotly.graph_objects as go
import httpx
import json
import time

from utils.state import master_state, theme_config

# Import the shared navigation bar we defined in components/navigation.py
from components.navigation import navigation_header

@ui.page('/')
def hangar(client: Client):
    client.content.classes('p-0 gap-0')
    navigation_header()

    # --- 1. STATE DATA ---
    # Moved inside the page function so each browser tab gets its own isolated session state
    hangar_state = master_state['hangar']

    def get_selected_node():
        """Returns the type ('wing', 'segment', 'fuse', 'nac') and the dictionary object."""
        sel_id = hangar_state['selected_id']
        veh = hangar_state['vehicle']
        
        for w in veh['wings']:
            if w['id'] == sel_id: return 'wing', w
            for s in w['segments']:
                if s['id'] == sel_id: return 'segment', s
        for f in veh['fuselages']:
            if f['id'] == sel_id: return 'fuse', f
        for n in veh['nacelles']:
            if n['id'] == sel_id: return 'nac', n
            
        return None, None

    # --- 3. 3D GEOMETRY GENERATION ---
    # def generate_wing_mesh():
    #     x_pts, y_pts, z_pts = [], [], []
    #     y_curr, x_le, z_le = 0.0, 0.0, 0.0
    #     c_curr = hangar_state['root_chord']
    #     tw_curr = np.radians(hangar_state['root_twist'])
        
    #     x_pts.extend([x_le, x_le + c_curr * np.cos(tw_curr)])
    #     y_pts.extend([y_curr, y_curr])
    #     z_pts.extend([z_le, z_le - c_curr * np.sin(tw_curr)])
        
    #     for seg in hangar_state['segments']:
    #         b = seg['span']
    #         sw_rad = np.radians(seg['sweep'])
    #         dih_rad = np.radians(seg['dihedral'])
            
    #         x_le += b * np.tan(sw_rad)
    #         y_curr += b
    #         z_le += b * np.tan(dih_rad)
    #         c_curr *= seg['taper']
    #         tw_curr = np.radians(seg['twist'])
            
    #         x_pts.extend([x_le, x_le + c_curr * np.cos(tw_curr)])
    #         y_pts.extend([y_curr, y_curr])
    #         z_pts.extend([z_le, z_le - c_curr * np.sin(tw_curr)])
            
    #     N = len(x_pts) 
        
    #     for idx in range(N):
    #         x_pts.append(x_pts[idx])
    #         y_pts.append(-y_pts[idx]) 
    #         z_pts.append(z_pts[idx])
            
    #     i_faces, j_faces, k_faces = [], [], []
    #     num_sections = len(hangar_state['segments'])
        
    #     for n in range(num_sections):
    #         base = 2 * n
    #         i_faces.extend([base, base + 2])
    #         j_faces.extend([base + 1, base + 1])
    #         k_faces.extend([base + 2, base + 3])
            
    #     for n in range(num_sections):
    #         base = N + 2 * n
    #         i_faces.extend([base, base + 2])
    #         j_faces.extend([base + 2, base + 3])
    #         k_faces.extend([base + 1, base + 1])
            
    #     fig = go.Figure(data=[
    #         go.Mesh3d(x=x_pts, y=y_pts, z=z_pts, i=i_faces, j=j_faces, k=k_faces, color='#3b82f6', opacity=0.8, flatshading=True)
    #     ])
        
    #     no_axis = dict(visible=False)
    #     fig.update_layout(
    #         uirevision='preserve_ui_state', 
    #         scene=dict(aspectmode='data', xaxis=no_axis, yaxis=no_axis, zaxis=no_axis, camera=dict(eye=dict(x=-2.5, y=-2.5, z=2.5))),
    #         margin=dict(l=0, r=0, b=0, t=0),
    #         template=theme_config[master_state['is_dark']]['plotly_template'],
    #         paper_bgcolor='rgba(0,0,0,0)',
    #         plot_bgcolor='rgba(0,0,0,0)',
    #     )
    #     return fig


    def generate_vehicle_mesh():
        print("--- STARTING MESH GENERATION ---")
        traces = []
        
        # Safely grab the vehicle state
        veh_state = hangar_state.get('vehicle', {})
        print(f"State Check -> Wings: {len(veh_state.get('wings', []))} | Fuselages: {len(veh_state.get('fuselages', []))} | Nacelles: {len(veh_state.get('nacelles', []))}")
        
        # -------------------------------------------------------------------------
        # Helper 1: Wing Generator 
        # -------------------------------------------------------------------------
        def create_wing_traces(wing, color='#3b82f6'):
            x_pts, y_pts, z_pts = [], [], []
            x_le = float(wing.get('x_offset', 0.0))
            y_curr = float(wing.get('y_offset', 0.0))
            z_le = float(wing.get('z_offset', 0.0))
            
            c_curr = float(wing.get('root_chord', 1.0))
            tw_curr = float(np.radians(wing.get('root_twist', 0.0)))
            
            x_pts.extend([x_le, float(x_le + c_curr * np.cos(tw_curr))])
            y_pts.extend([y_curr, y_curr])
            z_pts.extend([z_le, float(z_le - c_curr * np.sin(tw_curr))])
            
            segments = wing.get('segments', [])
            for seg in segments:
                b = float(seg['span'])
                sw_rad = float(np.radians(seg['sweep']))
                dih_rad = float(np.radians(seg['dihedral']))
                
                # --- TRIGONOMETRY FIX ---
                # Break the span down into true Y and Z components
                delta_y = b * np.cos(dih_rad)
                delta_z = b * np.sin(dih_rad)
                delta_x = b * np.tan(sw_rad) 
                
                x_le += float(delta_x)
                y_curr += float(delta_y)
                z_le += float(delta_z)
                
                c_curr *= float(seg['taper'])
                tw_curr = float(np.radians(seg['twist']))
                
                x_pts.extend([x_le, float(x_le + c_curr * np.cos(tw_curr))])
                y_pts.extend([y_curr, y_curr])
                z_pts.extend([z_le, float(z_le - c_curr * np.sin(tw_curr))])
                
            N = len(x_pts) 
            is_symmetric = wing.get('symmetric', True)
            
            if is_symmetric:
                for idx in range(N):
                    x_pts.append(x_pts[idx])
                    y_pts.append(-y_pts[idx]) 
                    z_pts.append(z_pts[idx])
                    
            i_faces, j_faces, k_faces = [], [], []
            num_sections = len(segments)
            
            for n in range(num_sections):
                base = 2 * n
                i_faces.extend([base, base + 2])
                j_faces.extend([base + 1, base + 1])
                k_faces.extend([base + 2, base + 3])
                
            if is_symmetric:
                for n in range(num_sections):
                    base = N + 2 * n
                    i_faces.extend([base, base + 2])
                    j_faces.extend([base + 2, base + 3])
                    k_faces.extend([base + 1, base + 1])
                    
            wing_lighting = dict(ambient=0.6, diffuse=0.7, roughness=0.8, specular=0.1)
            
            return [go.Mesh3d(x=x_pts, y=y_pts, z=z_pts, i=i_faces, j=j_faces, k=k_faces, 
                             color=color, opacity=1.0, flatshading=False, 
                             lighting=wing_lighting, name=wing.get('name', 'Wing'))]

        # -------------------------------------------------------------------------
        # Helper 2: Bluff Body Generator
        # -------------------------------------------------------------------------
        def create_bluff_body_traces(comp, color='#9ca3af', is_nacelle=False):
            length = float(comp.get('length', 10.0))
            diam = float(comp.get('diameter', 2.0))
            x_off = float(comp.get('x_offset', 0.0))
            y_off = float(comp.get('y_offset', 0.0))
            z_off = float(comp.get('z_offset', 0.0))
            
            R_max = diam / 2.0
            n_x = 30  # Increased slightly since cosine spacing handles it gracefully
            n_theta = 32  
            
            # --- COSINE SPACING ---
            # Clusters x-points near 0 and length, spreads them out in the middle
            theta_x = np.linspace(0, np.pi, n_x)
            x_vals = (length / 2.0) * (1.0 - np.cos(theta_x))
            
            x_pts, y_pts, z_pts = [], [], []
            
            for x in x_vals:
                local_z_off = z_off
                
                if is_nacelle:
                    # --- NACELLE LOGIC ---
                    L_nose = length * 0.10
                    R_in = R_max * 0.85  # Blunt intake face
                    R_out = R_max * 0.75 # Blunt exhaust face
                    
                    if x <= L_nose:
                        # High curvature rounding out to R_max
                        val = max(0.0, 1.0 - ((L_nose - x) / L_nose)**2)
                        r = R_in + (R_max - R_in) * float(np.sqrt(val))
                    else:
                        # Gently sloping parabola from L_nose down to exhaust
                        val = ((x - L_nose) / (length - L_nose))**2
                        r = R_max - (R_max - R_out) * float(val)
                else:
                    # --- FUSELAGE LOGIC ---
                    L_nose = min(length * 0.2, R_max * 2.5)
                    L_tail = min(length * 0.35, R_max * 4.0)
                    
                    if x < L_nose:
                        val = max(0.0, 1.0 - ((L_nose - x) / L_nose)**2)
                        r = R_max * float(np.sqrt(val))
                    elif x > length - L_tail:
                        val = max(0.0, 1.0 - ((x - (length - L_tail)) / L_tail)**2)
                        r = R_max * float(np.sqrt(val))
                        local_z_off += (R_max - r)  # Tail upsweep
                    else:
                        r = R_max
                    
                for j in range(n_theta):
                    theta = 2.0 * np.pi * j / n_theta
                    x_pts.append(float(x + x_off))
                    y_pts.append(float(r * np.cos(theta) + y_off))
                    z_pts.append(float(r * np.sin(theta) + local_z_off))
                    
            i_faces, j_faces, k_faces = [], [], []
            for i in range(n_x - 1):
                for j in range(n_theta):
                    p1 = i * n_theta + j
                    p2 = i * n_theta + (j + 1) % n_theta
                    p3 = (i + 1) * n_theta + j
                    p4 = (i + 1) * n_theta + (j + 1) % n_theta
                    
                    i_faces.extend([int(p1), int(p1)])
                    j_faces.extend([int(p2), int(p4)])
                    k_faces.extend([int(p4), int(p3)])
                    
            skin_lighting = dict(ambient=0.6, diffuse=0.7, roughness=0.8, specular=0.1)
                    
            body_traces = [go.Mesh3d(x=x_pts, y=y_pts, z=z_pts, i=i_faces, j=j_faces, k=k_faces, 
                                     color=color, opacity=1.0, flatshading=False, 
                                     lighting=skin_lighting, name=comp.get('name', 'Body'))]
            
            if comp.get('symmetric', False):
                y_pts_mirrored = [-y for y in y_pts]
                body_traces.append(go.Mesh3d(x=x_pts, y=y_pts_mirrored, z=z_pts, i=i_faces, j=k_faces, k=j_faces, 
                                             color=color, opacity=1.0, flatshading=False, 
                                             lighting=skin_lighting, name=comp.get('name', 'Body') + ' (Mirrored)'))
            
            return body_traces

        # -------------------------------------------------------------------------
        # Assembly Loop with Exception Catching
        # -------------------------------------------------------------------------
        try:
            for fuse in veh_state.get('fuselages', []):
                print(f" -> Building fuselage: {fuse['id']}")
                traces.extend(create_bluff_body_traces(fuse, color='#cbd5e1')) 
                
            for nac in veh_state.get('nacelles', []):
                print(f" -> Building nacelle: {nac['id']}")
                traces.extend(create_bluff_body_traces(nac, color='#475569', is_nacelle=True)) 

            for wing in veh_state.get('wings', []):
                print(f" -> Building wing: {wing['id']}")
                traces.extend(create_wing_traces(wing))

            print(f"Total traces generated successfully: {len(traces)}")
            if len(traces) > 0:
                 print(f"Sanity Check: Trace 0 has {len(traces[0].x)} vertices and {len(traces[0].i)} faces.")
                 
        except Exception as e:
             print(f"!!! CRITICAL MATH/GEOMETRY ERROR: {e}")
             import traceback
             traceback.print_exc()

        fig = go.Figure(data=traces)
        
        no_axis = dict(visible=False)
        fig.update_layout(
            uirevision='preserve_ui_state', 
            scene=dict(aspectmode='data', xaxis=no_axis, yaxis=no_axis, zaxis=no_axis, 
                       camera=dict(eye=dict(x=-2.5, y=-2.5, z=2.5))),
            margin=dict(l=0, r=0, b=0, t=0),
            template=theme_config[master_state['is_dark']]['plotly_template'],
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        print("--- MESH GENERATION COMPLETE ---")
        return fig

    # --- 4. UI CALLBACKS ---
    def update_plot():
        print("\n=== TRIGGERING PLOT UPDATE ===")
        try:
            fig = generate_vehicle_mesh()
            plot.update_figure(fig)
            print("Plotly update_figure executed successfully.")
        except Exception as e:
            print(f"ERROR IN UPDATE_PLOT: {e}")
            import traceback
            traceback.print_exc()

    def select_node(e):
        if e.value:  
            hangar_state['selected_id'] = e.value
            attribute_sliders.refresh() 

    def add_segment():
        sel_type, sel_obj = get_selected_node()
        
        # Find the parent wing regardless of if a wing or segment is selected
        target_wing = sel_obj if sel_type == 'wing' else next(w for w in hangar_state['vehicle']['wings'] if sel_obj in w['segments'])
        
        new_id = f"seg_{int(time.time()*1000)}"
        target_wing['segments'].append({
            'id': new_id, 'name': f'Segment {len(target_wing["segments"]) + 1}',
            'span': 5.0, 'taper': 0.8, 'sweep': 10.0, 'dihedral': 0.0, 'twist': 0.0
        })
        hangar_state['selected_id'] = new_id
        vehicle_tree.refresh()
        attribute_sliders.refresh()
        update_plot()
        
    def remove_segment():
        sel_type, sel_obj = get_selected_node()
        if sel_type == 'segment':
            target_wing = next(w for w in hangar_state['vehicle']['wings'] if sel_obj in w['segments'])
            if len(target_wing['segments']) > 1:
                target_wing['segments'] = [s for s in target_wing['segments'] if s['id'] != sel_obj['id']]
                hangar_state['selected_id'] = target_wing['id']
                vehicle_tree.refresh()
                attribute_sliders.refresh()
                update_plot()
                
    def add_component():
        cat = hangar_state['selected_id']
        new_id = f"comp_{int(time.time()*1000)}"
        
        if cat == 'cat_wings':
            hangar_state['vehicle']['wings'].append({'id': new_id, 'name': 'New Wing', 'symmetric': True, 'x_offset': 0, 'y_offset': 0, 'z_offset': 0, 'root_chord': 2.0, 'root_twist': 0.0, 'segments': [{'id': f"{new_id}_s1", 'name': 'Seg 1', 'span': 5.0, 'taper': 1.0, 'sweep': 0, 'dihedral': 0, 'twist': 0}]})
        elif cat == 'cat_fuses':
            hangar_state['vehicle']['fuselages'].append({'id': new_id, 'name': 'New Fuselage', 'length': 10.0, 'diameter': 2.0, 'x_offset': 0, 'y_offset': 0, 'z_offset': 0})
        elif cat == 'cat_nacs':
             hangar_state['vehicle']['nacelles'].append({'id': new_id, 'name': 'New Engine', 'length': 3.0, 'diameter': 1.0, 'x_offset': 0, 'y_offset': 3.0, 'z_offset': -1.0, 'symmetric': True})
             
        hangar_state['selected_id'] = new_id
        vehicle_tree.refresh()
        attribute_sliders.refresh()
        update_plot()

    # --- 5. DYNAMIC UI COMPONENTS ---
    @ui.refreshable
    def vehicle_tree():
        veh = hangar_state['vehicle']
        
        # Build nested tree data
        tree_data = [
            {'id': 'cat_wings', 'label': 'Wings', 'icon': 'flight', 'children': []},
            {'id': 'cat_fuses', 'label': 'Fuselages', 'icon': 'straighten', 'children': []},
            {'id': 'cat_nacs', 'label': 'Engines', 'icon': 'cyclone', 'children': []}
        ]
        
        for w in veh['wings']:
            w_node = {'id': w['id'], 'label': w['name'], 'children': [{'id': s['id'], 'label': s['name']} for s in w['segments']]}
            tree_data[0]['children'].append(w_node)
            
        for f in veh['fuselages']:
            tree_data[1]['children'].append({'id': f['id'], 'label': f['name']})
            
        for n in veh['nacelles']:
            tree_data[2]['children'].append({'id': n['id'], 'label': n['name']})
            
        # The tree draws once and handles its own internal visual state
        ui.tree(tree_data, on_select=select_node, tick_strategy='none').expand()

    def synced_slider(label_text, obj, key, min_val, max_val, step):
        # Wrap in a column with zero gap and a small bottom margin
        with ui.column().classes('w-full gap-0 mb-1'): 
            
            # Row for Label and Number input
            with ui.row().classes('w-full items-center justify-between'):
                ui.label(label_text).classes('text-xs text-gray-600')
                
                # Dense number input, synced to the same value
                ui.number(value=obj[key], step=step, on_change=update_plot) \
                    .bind_value(obj, key) \
                    .props('dense borderless') \
                    .classes('w-16 text-xs')
                    
            # Dense slider pulled upward with a negative margin to eat the padding
            ui.slider(min=min_val, max=max_val, step=step, value=obj[key], on_change=update_plot) \
                .bind_value(obj, key) \
                .props('dense') \
                .classes('mt-[-8px]')

    @ui.refreshable
    def attribute_sliders():
        sel_type, sel_obj = get_selected_node()
        
        # If no valid type is selected, AND a category folder isn't clicked, show nothing.
        if not sel_type and not hangar_state.get('selected_id', '').startswith('cat_'):
            ui.label('Select a component to edit.').classes('text-gray-500 italic')
            return
            
        # --- SLIDERS ---
        if sel_type:
            ui.input('Name', value=sel_obj['name'], on_change=lambda _: vehicle_tree.refresh()).bind_value(sel_obj, 'name').classes('w-full mb-4 text-lg font-bold')
            
            ui.label('Origin (X, Z)').classes('text-sm font-bold text-gray-700 mt-2')
            if sel_type in ['wing', 'fuse', 'nac']:
                sel_obj['y_offset'] = 0.0
                if 'symmetric' in sel_obj:
                    sel_obj['symmetric'] = True
    
                with ui.row().classes('w-full items-center gap-2 mb-2'):
                    ui.number('X', value=sel_obj['x_offset'], step=0.5, format='%.1f', 
                              on_change=update_plot).bind_value(sel_obj, 'x_offset').props('dense').classes('flex-1')
                    ui.number('Z', value=sel_obj['z_offset'], step=0.5, format='%.1f', 
                              on_change=update_plot).bind_value(sel_obj, 'z_offset').props('dense').classes('flex-1')
                
            ui.label('Geometry').classes('text-sm font-bold text-gray-700 mt-2')
            if sel_type == 'wing':
                synced_slider('Root Chord (m)', sel_obj, 'root_chord', 0.5, 15.0, 0.1)
                synced_slider('Root Twist (°)', sel_obj, 'root_twist', -10.0, 10.0, 0.5)
            elif sel_type == 'segment':
                synced_slider('Span (m)', sel_obj, 'span', 0.5, 30.0, 0.1)
                synced_slider('Taper Ratio', sel_obj, 'taper', 0.05, 1.5, 0.05)
                synced_slider('LE Sweep (°)', sel_obj, 'sweep', -30.0, 70.0, 1.0)
                synced_slider('Dihedral (°)', sel_obj, 'dihedral', -30.0, 90.0, 1.0)
                synced_slider('Tip Twist (°)', sel_obj, 'twist', -10.0, 10.0, 0.5)
            elif sel_type in ['fuse', 'nac']:
                synced_slider('Length (m)', sel_obj, 'length', 1.0, 80.0, 0.5)
                synced_slider('Diameter (m)', sel_obj, 'diameter', 0.5, 10.0, 0.1)

        # --- DYNAMIC BUTTON LOGIC ---
        ui.separator().classes('my-4') # <--- Separator goes OUTSIDE the row!
        
        with ui.row().classes('w-full gap-2'):
            if sel_type in ['wing', 'segment']:
                ui.button('+ Add Segment', on_click=add_segment, color='primary').classes('flex-grow text-xs')
                if sel_type == 'segment':
                    ui.button('- Remove Segment', on_click=remove_segment, color='Blue').classes('flex-grow text-xs')
            elif hangar_state.get('selected_id', '').startswith('cat_'):
                ui.button('+ Add Component Here', on_click=add_component, color='blue').classes('flex-grow text-xs')

    def generate_rcaide_payload():
        return {
            "flight_regime": {"mach": hangar_state["mach"], "alpha": hangar_state["alpha"], "beta": hangar_state["beta"]},
            "vehicle": {
                "main_wing": {
                    "root_chord": hangar_state["root_chord"],
                    "root_twist": hangar_state["root_twist"],
                    "segments": [
                        {"name": seg["name"], "span": seg["span"], "taper": seg["taper"], "sweep": seg["sweep"], "dihedral": seg["dihedral"], "twist": seg["twist"]} 
                        for seg in hangar_state["segments"]
                    ]
                }
            }
        }

    def extract_scalar(val):
        while isinstance(val, list) and len(val) > 0: val = val[0]
        return val

    async def run_analysis():
        payload_dict = generate_rcaide_payload()
        run_button.disable()
        results_table.classes(add='hidden')
        loading_indicator.classes(remove='hidden')
        loading_label.text = "Connecting to server..." 
        
        async with httpx.AsyncClient() as client:
            try:
                async with client.stream('POST', 'http://localhost:8000/solve_mission', json=payload_dict, timeout=None) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line: continue 
                        data = json.loads(line)
                        
                        if data['type'] == 'status':
                            loading_label.text = data['message']
                        elif data['type'] == 'result':
                            coeffs = data['coefficients']
                            results_table.rows = [
                                {'coeff': 'Lift (CL)',
                                 'base': f"{extract_scalar(coeffs.get('CL', 0.0)):.4f}",
                                 'd_alpha': f"{extract_scalar(coeffs.get('dCL_da', 0.0)):.4f}",
                                 'd_beta': f"{extract_scalar(coeffs.get('dCL_db', 0.0)):.4f}",
                                 'd_mach': f"{extract_scalar(coeffs.get('dCL_dM', 0.0)):.4f}"},
                                {'coeff': 'Drag (CD)',
                                 'base': f"{extract_scalar(coeffs.get('CD', 0.0)):.4f}",
                                 'd_alpha': f"{extract_scalar(coeffs.get('dCD_da', 0.0)):.4f}",
                                 'd_beta': f"{extract_scalar(coeffs.get('dCD_db', 0.0)):.4f}",
                                 'd_mach': f"{extract_scalar(coeffs.get('dCD_dM', 0.0)):.4f}"},
                                {'coeff': 'Pitch (Cm)',
                                 'base': f"{extract_scalar(coeffs.get('C_m', 0.0)):.4f}",
                                 'd_alpha': f"{extract_scalar(coeffs.get('dC_m_da', 0.0)):.4f}",
                                 'd_beta': f"{extract_scalar(coeffs.get('dC_m_db', 0.0)):.4f}",
                                 'd_mach': f"{extract_scalar(coeffs.get('dC_m_dM', 0.0)):.4f}"}
                            ]
                            results_table.classes(remove='hidden')
                            ui.notify('Solve Complete!', type='positive', position='bottom-right')
                        elif data['type'] == 'error':
                            ui.notify(f"Backend Error: {data['message']}", type='negative')
            except httpx.ConnectError:
                ui.notify('Connection failed. Is the FastAPI server running?', type='negative')
            except Exception as e:
                ui.notify(f'Stream Error: {str(e)}', type='negative')
            finally:
                run_button.enable()
                loading_indicator.classes(add='hidden')

    # --- 6. MAIN LAYOUT ---
    master_state['on_theme_changed'].append(update_plot)

    # Dialogs (Must be instantiated inside the page function)
    with ui.dialog() as payload_dialog, ui.card().classes('w-full max-w-2xl'):
        ui.label('RCAIDE JSON Payload').classes('text-xl font-bold')
        json_display = ui.code('', language='json').classes('w-full')
        with ui.row().classes('w-full justify-end mt-4'):
            ui.button('Close', on_click=payload_dialog.close, color='gray')
            ui.button('Send to Solver', color='blue') 

    # Drawers
    with ui.left_drawer().classes('p-4 border-r w-80'):
        ui.label('Vehicle Tree').classes('text-lg font-bold mb-2')
        vehicle_tree()  
        ui.separator().classes('my-4')
        attribute_sliders() 

    with ui.right_drawer().props('width=450').classes('p-4 border-l'):
        ui.label('Flight Regime').classes('text-lg font-bold mb-2')
        
        with ui.row().classes('w-full gap-2'):
            ui.number('Mach', value=hangar_state['mach'], step=0.05) \
                .bind_value(hangar_state, 'mach').props('dense').classes('flex-1')
                
            ui.number('AoA (°)', value=hangar_state['alpha'], step=0.5) \
                .bind_value(hangar_state, 'alpha').props('dense').classes('flex-1')
                
            ui.number('Beta (°)', value=hangar_state['beta'], step=0.5) \
                .bind_value(hangar_state, 'beta').props('dense').classes('flex-1')
        
        ui.separator().classes('my-4')
        run_button = ui.button('Run Analysis', on_click=run_analysis, color='blue').classes('w-full')

        with ui.row().classes('w-full items-center justify-center mt-4 hidden') as loading_indicator:
            ui.spinner('orbit', size='md', color='blue')
            loading_label = ui.label('Initializing...').classes('ml-2 text-gray-600 font-medium')

        ui.separator().classes('my-4')
        # --- ANALYSIS RESULTS SECTION ---
        ui.label('Analysis Results').classes('text-lg font-bold mt-6 mb-2')
        
        # 1. The Notice Placeholder (Visible by default)
        no_results_notice = ui.column().classes('w-full bg-slate-50 border border-slate-200 rounded-lg p-4 items-center justify-center')
        with no_results_notice:
            ui.icon('science', size='2rem', color='slate-400')
            ui.label('No analysis results currently available.').classes('text-sm font-bold text-slate-700 text-center mt-2')
            ui.label('Please specify a vehicle and analysis setup, then hit "Run Analysis".').classes('text-xs text-slate-500 text-center mt-1')

        # 2. The Results Container (Hidden by default)
        results_container = ui.column().classes('w-full gap-0 hidden')
        with results_container:
            
            # Sub-section: Aerodynamics
            ui.label('Aerodynamics').classes('text-sm font-bold text-gray-700 mt-2 mb-1')
            result_columns = [
                {'name': 'coeff', 'label': 'Coeff', 'field': 'coeff', 'align': 'left'},
                {'name': 'base', 'label': 'Base', 'field': 'base', 'align': 'right'},
                {'name': 'd_alpha', 'label': '∂/∂α', 'field': 'd_alpha', 'align': 'right'},
                {'name': 'd_beta', 'label': '∂/∂β', 'field': 'd_beta', 'align': 'right'},
                {'name': 'd_mach', 'label': '∂/∂M', 'field': 'd_mach', 'align': 'right'}
            ]
            results_table = ui.table(columns=result_columns, rows=[], row_key='coeff').classes('w-full').props('dense flat bordered')
            
            # Sub-section: Propulsion
            ui.label('Propulsion').classes('text-sm font-bold text-gray-700 mt-4 mb-1')
            ui.label('Pending engine performance mapping...').classes('text-xs text-gray-500 italic ml-2')
            
            # Sub-section: Mass
            ui.label('Mass Properties').classes('text-sm font-bold text-gray-700 mt-4 mb-1')
            ui.label('Pending component weight buildup...').classes('text-xs text-gray-500 italic ml-2')

        ui.separator().classes('my-6')
        
        with ui.row().classes('w-full items-center justify-between mb-2'):
            ui.label('RCAIDE Assistant').classes('text-lg font-bold text-gray-700')
            ui.icon('psychology', size='sm', color='primary')
        
        # Main chat container (Transparent bg, strict border, NO WRAP)
        with ui.column().classes('w-full h-full border border-slate-300 rounded-lg flex-nowrap gap-0 overflow-hidden bg-transparent'):
            
            # Scrollable message area (flex-grow lets it take available space, min-h-0 prevents it from pushing out the bottom)
            with ui.column().classes('w-full h-full flex-grow p-3 overflow-y-auto gap-2 h-full'):
                
                # Agent Message: Left side, primary blue, white text
                ui.chat_message('Hello! I am your RCAIDE agent. How can I help optimize this configuration today?', 
                                name='RCAIDE', stamp='System Ready', text_html=True) \
                    .props('bg-color=primary text-color=white')
                
                # User Message: Right side (sent=True), light gray, black text
                ui.chat_message('Can you set up a parameter sweep for the wing LE sweep from 20° to 35°?', 
                                name='You',
                                stamp='just now',
                                sent=True) \
                    .props('bg-color=grey-4 text-color=black')
                
                # Agent Message
                ui.chat_message('Done. I have queued a parametric sweep for **LE Sweep (20° - 35°)**. Hit "Run Analysis" when you are ready to compute the variations.', 
                                name='RCAIDE',
                                stamp='Analysis Ready',
                                text_html=True) \
                    .props('bg-color=primary text-color=white')
    
            # Input area at the bottom (shrink-0 protects its height, transparent background)
            with ui.row().classes('w-full shrink-0 p-2 border-t border-slate-300 items-center flex-nowrap bg-transparent'):
                
                # Outlined input guarantees the text box boundaries are visible against the transparent background
                chat_input = ui.input(placeholder='Ask RCAIDE...') \
                    .props('dense outlined') \
                    .classes('flex-grow text-sm')
                
                ui.button(icon='send', color='primary') \
                    .props('round flat dense') \
                    .classes('ml-2')

    # Main 3D Canvas
    with ui.element('div').classes('absolute-full flex justify-center items-center'):
        plot = ui.plotly(generate_vehicle_mesh()).classes('w-full h-full')

    ui.timer(0.1, update_plot, once=True)