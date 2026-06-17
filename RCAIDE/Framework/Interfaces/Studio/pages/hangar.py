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
    def generate_wing_mesh():
        x_pts, y_pts, z_pts = [], [], []
        y_curr, x_le, z_le = 0.0, 0.0, 0.0
        c_curr = hangar_state['root_chord']
        tw_curr = np.radians(hangar_state['root_twist'])
        
        x_pts.extend([x_le, x_le + c_curr * np.cos(tw_curr)])
        y_pts.extend([y_curr, y_curr])
        z_pts.extend([z_le, z_le - c_curr * np.sin(tw_curr)])
        
        for seg in hangar_state['segments']:
            b = seg['span']
            sw_rad = np.radians(seg['sweep'])
            dih_rad = np.radians(seg['dihedral'])
            
            x_le += b * np.tan(sw_rad)
            y_curr += b
            z_le += b * np.tan(dih_rad)
            c_curr *= seg['taper']
            tw_curr = np.radians(seg['twist'])
            
            x_pts.extend([x_le, x_le + c_curr * np.cos(tw_curr)])
            y_pts.extend([y_curr, y_curr])
            z_pts.extend([z_le, z_le - c_curr * np.sin(tw_curr)])
            
        N = len(x_pts) 
        
        for idx in range(N):
            x_pts.append(x_pts[idx])
            y_pts.append(-y_pts[idx]) 
            z_pts.append(z_pts[idx])
            
        i_faces, j_faces, k_faces = [], [], []
        num_sections = len(hangar_state['segments'])
        
        for n in range(num_sections):
            base = 2 * n
            i_faces.extend([base, base + 2])
            j_faces.extend([base + 1, base + 1])
            k_faces.extend([base + 2, base + 3])
            
        for n in range(num_sections):
            base = N + 2 * n
            i_faces.extend([base, base + 2])
            j_faces.extend([base + 2, base + 3])
            k_faces.extend([base + 1, base + 1])
            
        fig = go.Figure(data=[
            go.Mesh3d(x=x_pts, y=y_pts, z=z_pts, i=i_faces, j=j_faces, k=k_faces, color='#3b82f6', opacity=0.8, flatshading=True)
        ])
        
        no_axis = dict(visible=False)
        fig.update_layout(
            uirevision='preserve_ui_state', 
            scene=dict(aspectmode='data', xaxis=no_axis, yaxis=no_axis, zaxis=no_axis, camera=dict(eye=dict(x=-2.5, y=-2.5, z=2.5))),
            margin=dict(l=0, r=0, b=0, t=0),
            template=theme_config[master_state['is_dark']]['plotly_template'],
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        return fig

    # --- 4. UI CALLBACKS ---
    def update_plot():
        plot.update_figure(generate_wing_mesh())

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
            {'id': 'cat_wings', 'label': 'Wings & Tails', 'icon': 'flight', 'children': []},
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
            
        ui.tree(tree_data, on_select=select_node, tick_strategy='none').expand()
        
        # Contextual Action Buttons
        sel_type, sel_obj = get_selected_node()
        ui.separator().classes('my-4')
        
        with ui.row().classes('w-full gap-2'):
            if sel_type == 'wing' or sel_type == 'segment':
                ui.button('+ Add Segment', on_click=add_segment, color='green').classes('flex-grow text-xs')
                if sel_type == 'segment':
                    ui.button('- Remove Segment', on_click=remove_segment, color='red').classes('flex-grow text-xs')
            elif hangar_state['selected_id'].startswith('cat_'):
                # Give options to add new base components if a category folder is clicked
                ui.button('+ Add Component Here', on_click=add_component, color='blue').classes('flex-grow text-xs')

    def synced_slider(label_text, state_dict, state_key, min_val, max_val, step_val):
        ui.label(label_text).classes('text-sm text-gray-500 mt-2')
        with ui.row().classes('w-full items-center justify-between no-wrap'):
            # The slider takes up ~66% of the space
            ui.slider(min=min_val, max=max_val, step=step_val, value=state_dict[state_key], on_change=update_plot) \
                .bind_value(state_dict, state_key).classes('w-2/3')
            
            # The number input takes the remaining space. 'dense' makes it compact.
            ui.number(value=state_dict[state_key], step=step_val, format='%.2f', on_change=update_plot) \
                .bind_value(state_dict, state_key).classes('w-1/4').props('dense')

    @ui.refreshable
    def attribute_sliders():
        sel_type, sel_obj = get_selected_node()
        if not sel_type:
            ui.label('Select a component to edit.').classes('text-gray-500 italic')
            return
            
        ui.input('Name', value=sel_obj['name'], on_change=lambda _: vehicle_tree.refresh()).bind_value(sel_obj, 'name').classes('w-full mb-4 text-lg font-bold')
        
        ui.label('Position (m)').classes('text-sm font-bold text-gray-700 mt-2')
        if sel_type in ['wing', 'fuse', 'nac']:
            synced_slider('X Offset (Nose-to-Tail)', sel_obj, 'x_offset', -10.0, 50.0, 0.5)
            synced_slider('Y Offset (Center-to-Tip)', sel_obj, 'y_offset', 0.0, 20.0, 0.5)
            synced_slider('Z Offset (Vertical)', sel_obj, 'z_offset', -10.0, 20.0, 0.5)
            
        ui.label('Geometry').classes('text-sm font-bold text-gray-700 mt-4')
        if sel_type == 'wing':
            ui.checkbox('Symmetric (Mirror across Y)').bind_value(sel_obj, 'symmetric').on('change', update_plot)
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
        
        ui.number('Mach Number', value=hangar_state['mach'], step=0.05).bind_value(hangar_state, 'mach').classes('w-full')
        ui.number('Angle of Attack (°)', value=hangar_state['alpha'], step=0.5).bind_value(hangar_state, 'alpha').classes('w-full mt-2')
        ui.number('Sideslip (°)', value=hangar_state['beta'], step=0.5).bind_value(hangar_state, 'beta').classes('w-full mt-2')
        
        ui.separator().classes('my-4')
        run_button = ui.button('Run Analysis', on_click=run_analysis, color='blue').classes('w-full')

        with ui.row().classes('w-full items-center justify-center mt-4 hidden') as loading_indicator:
            ui.spinner('orbit', size='md', color='blue')
            loading_label = ui.label('Initializing...').classes('ml-2 text-gray-600 font-medium')

        ui.separator().classes('my-4')
        ui.label('Aerodynamic Coefficients').classes('text-lg font-bold mb-2')
        
        result_columns = [
            {'name': 'coeff', 'label': 'Coeff', 'field': 'coeff', 'align': 'left'},
            {'name': 'base', 'label': 'Base', 'field': 'base', 'align': 'right'},
            {'name': 'd_alpha', 'label': '∂/∂α', 'field': 'd_alpha', 'align': 'right'},
            {'name': 'd_beta', 'label': '∂/∂β', 'field': 'd_beta', 'align': 'right'},
            {'name': 'd_mach', 'label': '∂/∂M', 'field': 'd_mach', 'align': 'right'}
        ]
        
        results_table = ui.table(columns=result_columns, rows=[], row_key='coeff').classes('w-full hidden').props('dense flat bordered')

    # Main 3D Canvas
    with ui.element('div').classes('absolute-full flex justify-center items-center'):
        plot = ui.plotly(generate_wing_mesh()).classes('w-full h-full')

    ui.timer(0.1, update_plot, once=True)