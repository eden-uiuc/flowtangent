from nicegui import ui
from pprint import pprint

import numpy as np
import plotly.graph_objects as go
import httpx
import json


# --- 1. STATE DATA ---
app_state = {
    'mach': 0.3,
    'alpha': 5.0,
    'beta': 0.0,
    'root_chord': 2.0,
    'root_twist': 0.0,
    'segment_counter': 1,  # <--- NEW: Monotonic counter for IDs
    'segments': [
        {'id': 'seg_1', 'name': 'Segment 1', 'span': 5.0, 'taper': 0.5, 'sweep': 20.0, 'dihedral': 5.0, 'twist': -2.0}
    ],
    'selected_id': 'root'
}

def save_project():
    """Dumps the full app_state to a downloadable JSON file."""
    # Convert the state to a formatted JSON string, then encode it to bytes
    project_json = json.dumps(app_state, indent=4).encode('utf-8')
    
    # ui.download pushes the file directly to the user's browser downloads folder
    ui.download(project_json, 'rcaide_project.rcaide')
    ui.notify('Project saved!', type='positive', position='top')

async def load_project(e):
    """Reads an uploaded JSON file and perfectly restores the app_state."""
    try:
        # In NiceGUI 3.0+, we access the file via e.file and read it asynchronously
        content = await e.file.text()
        loaded_state = json.loads(content)
        
        # IN-PLACE UPDATE: This preserves all NiceGUI data bindings
        app_state.update(loaded_state)
        
        # Force the UI to redraw with the new data
        vehicle_tree.refresh()
        attribute_sliders.refresh()
        update_plot()
        
        ui.notify('Project loaded successfully!', type='positive', position='top')
        upload_dialog.close() # Close the upload window
        
    except Exception as ex:
        ui.notify(f'Failed to load project: {str(ex)}', type='negative', position='top')

# --- 2. 3D GEOMETRY GENERATION ---
def generate_wing_mesh():
    x_pts, y_pts, z_pts = [], [], []
    
    # Track the current geometric state as we march down the span
    y_curr = 0.0
    x_le = 0.0
    z_le = 0.0
    c_curr = app_state['root_chord']
    tw_curr = np.radians(app_state['root_twist'])
    
    # 1. Add the Root Vertices [LE, TE]
    x_pts.extend([x_le, x_le + c_curr * np.cos(tw_curr)])
    y_pts.extend([y_curr, y_curr])
    z_pts.extend([z_le, z_le - c_curr * np.sin(tw_curr)])
    
    # 2. Iterate through segments to build the right wing
    for seg in app_state['segments']:
        b = seg['span']
        sw_rad = np.radians(seg['sweep'])
        dih_rad = np.radians(seg['dihedral'])
        
        # Advance coordinates
        x_le += b * np.tan(sw_rad)
        y_curr += b
        z_le += b * np.tan(dih_rad)
        c_curr *= seg['taper']
        tw_curr = np.radians(seg['twist'])
        
        # Add Segment Tip Vertices [LE, TE]
        x_pts.extend([x_le, x_le + c_curr * np.cos(tw_curr)])
        y_pts.extend([y_curr, y_curr])
        z_pts.extend([z_le, z_le - c_curr * np.sin(tw_curr)])
        
    N = len(x_pts) # Total points on the right wing
    
    # 3. Mirror the points for the left wing
    for idx in range(N):
        x_pts.append(x_pts[idx])
        y_pts.append(-y_pts[idx]) # Flip across symmetry plane
        z_pts.append(z_pts[idx])
        
    # 4. Tessellate the faces
    i_faces, j_faces, k_faces = [], [], []
    num_sections = len(app_state['segments'])
    
    # Tessellate Right Wing
    for n in range(num_sections):
        base = 2 * n
        i_faces.extend([base, base + 2])
        j_faces.extend([base + 1, base + 1])
        k_faces.extend([base + 2, base + 3])
        
    # Tessellate Left Wing (Reversed winding)
    for n in range(num_sections):
        base = N + 2 * n
        i_faces.extend([base, base + 2])
        j_faces.extend([base + 2, base + 3])
        k_faces.extend([base + 1, base + 1])
        
    fig = go.Figure(data=[
        go.Mesh3d(
            x=x_pts, y=y_pts, z=z_pts,
            i=i_faces, j=j_faces, k=k_faces,
            color='#3b82f6', opacity=0.8, flatshading=True
        )
    ])
    
    no_axis = dict(visible=False)
    fig.update_layout(
        uirevision='preserve_ui_state', 
        scene=dict(aspectmode='data', xaxis=no_axis, yaxis=no_axis, zaxis=no_axis,
                   camera=dict(eye=dict(x=-2.5, y=-2.5, z=2.5))),
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

# --- 3. UI CALLBACKS ---
def update_plot():
    plot.update_figure(generate_wing_mesh())

def select_node(e):
    if e.value:  # If a node is clicked
        app_state['selected_id'] = e.value
        attribute_sliders.refresh() # Redraw the slider panel

def add_segment():
    new_idx = len(app_state['segments']) + 1
    new_id = f'seg_{new_idx}'
    app_state['segments'].append({
        'id': new_id, 'name': f'Segment {new_idx}',
        'span': 5.0, 'taper': 0.8, 'sweep': 10.0, 'dihedral': 0.0, 'twist': 0.0
    })
    app_state['selected_id'] = new_id
    vehicle_tree.refresh()
    attribute_sliders.refresh()
    update_plot()

def remove_segment():
    if len(app_state['segments']) > 1:
        app_state['segments'] = [s for s in app_state['segments'] if s['id'] != app_state['selected_id']]
        app_state['selected_id'] = 'root'
        vehicle_tree.refresh()
        attribute_sliders.refresh()
        update_plot()

# --- 4. DYNAMIC UI COMPONENTS ---
@ui.refreshable
def vehicle_tree():
    children = [{'id': s['id'], 'label': s['name'], 'icon': 'straighten'} for s in app_state['segments']]
    tree_data = [{'id': 'root', 'label': 'Main Wing', 'children': children, 'icon': 'flight'}]
    
    ui.tree(tree_data, on_select=select_node, tick_strategy='none').expand()
    
    with ui.row().classes('w-full mt-2'):
        ui.button('+ Add Segment', on_click=add_segment, color='blue').classes('flex-grow')
        if app_state['selected_id'] != 'root' and len(app_state['segments']) > 1:
            ui.button('- Remove Segment', on_click=remove_segment, color='blue').classes('flex-grow')

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
    sel_id = app_state['selected_id']
    
    if sel_id == 'root':
        ui.label('Wing Properties').classes('text-lg font-bold mb-2')
        synced_slider('Root Chord (m)', app_state, 'root_chord', 0.5, 10.0, 0.1)
        synced_slider('Root Twist (°)', app_state, 'root_twist', -10.0, 10.0, 0.5)
        
    else:
        seg = next((s for s in app_state['segments'] if s['id'] == sel_id), None)
        if seg:
            ui.input('Segment Name', value=seg['name'], on_change=lambda _: vehicle_tree.refresh()) \
                .bind_value(seg, 'name').classes('w-full mb-4 text-lg font-bold')
            
            synced_slider('Span (m)', seg, 'span', 1.0, 20.0, 0.1)
            synced_slider('Taper Ratio', seg, 'taper', 0.1, 1.5, 0.05)
            synced_slider('LE Sweep (°)', seg, 'sweep', -20.0, 60.0, 1.0)
            synced_slider('Dihedral (°)', seg, 'dihedral', -10.0, 20.0, 1.0)
            synced_slider('Tip Twist (°)', seg, 'twist', -10.0, 10.0, 0.5)

def generate_rcaide_payload():
    """Extracts physics data from the UI state as a dictionary."""
    payload = {
        "flight_regime": {
            "mach": app_state["mach"],
            "alpha": app_state["alpha"],
            "beta": app_state["beta"]
        },
        "vehicle": {
            "main_wing": {
                "root_chord": app_state["root_chord"],
                "root_twist": app_state["root_twist"],
                "segments": [
                    {
                        "name": seg["name"],
                        "span": seg["span"],
                        "taper": seg["taper"],
                        "sweep": seg["sweep"],
                        "dihedral": seg["dihedral"],
                        "twist": seg["twist"]
                    } for seg in app_state["segments"]
                ]
            }
        }
    }
    return payload

with ui.dialog() as payload_dialog, ui.card().classes('w-full max-w-2xl'):
    ui.label('RCAIDE JSON Payload').classes('text-xl font-bold')
    # Use a code block to cleanly format the JSON string
    json_display = ui.code('', language='json').classes('w-full')
    
    with ui.row().classes('w-full justify-end mt-4'):
        ui.button('Close', on_click=payload_dialog.close, color='gray')
        ui.button('Send to Solver', color='blue') # We will hook this up next!

with ui.dialog() as upload_dialog, ui.card().classes('w-96'):
    ui.label('Load RCAIDE Project').classes('text-lg font-bold mb-2')
    # auto_upload=True means as soon as they drop the file, load_project triggers
    ui.upload(on_upload=load_project, auto_upload=True, max_files=1).classes('w-full')
    
    with ui.row().classes('w-full justify-end mt-4'):
        ui.button('Cancel', on_click=upload_dialog.close, color='gray')

def extract_scalar(val):
    """Recursively drills down into nested lists to find the first scalar value."""
    while isinstance(val, list) and len(val) > 0:
        val = val[0]
    return val

async def run_analysis():
    payload_dict = generate_rcaide_payload()
    
    run_button.disable()
    results_table.classes(add='hidden')
    loading_indicator.classes(remove='hidden')
    # Reset the label text for subsequent runs
    loading_label.text = "Connecting to server..." 
    
    async with httpx.AsyncClient() as client:
        try:
            # Use client.stream() and disable the timeout
            async with client.stream('POST', 'http://localhost:8000/solve_mission', json=payload_dict, timeout=None) as response:
                response.raise_for_status()
                
                # Iterate over the incoming stream line by line
                async for line in response.aiter_lines():
                    if not line:
                        continue # Skip empty lines
                    
                    data = json.loads(line)
                    
                    # Handle intermediate status updates
                    if data['type'] == 'status':
                        loading_label.text = data['message']
                        
                    elif data['type'] == 'result':
                        coeffs = data['coefficients']
                        
                        # Populate the table rows as a matrix
                        results_table.rows = [
                            {
                                'coeff': 'Lift (CL)',
                                'base': f"{extract_scalar(coeffs.get('CL', 0.0)):.4f}",
                                'd_alpha': f"{extract_scalar(coeffs.get('dCL_da', 0.0)):.4f}",
                                'd_beta': f"{extract_scalar(coeffs.get('dCL_db', 0.0)):.4f}",
                                'd_mach': f"{extract_scalar(coeffs.get('dCL_dM', 0.0)):.4f}"
                            },
                            {
                                'coeff': 'Drag (CD)',
                                'base': f"{extract_scalar(coeffs.get('CD', 0.0)):.4f}",
                                'd_alpha': f"{extract_scalar(coeffs.get('dCD_da', 0.0)):.4f}",
                                'd_beta': f"{extract_scalar(coeffs.get('dCD_db', 0.0)):.4f}",
                                'd_mach': f"{extract_scalar(coeffs.get('dCD_dM', 0.0)):.4f}"
                            },
                            {
                                'coeff': 'Pitch (Cm)',
                                'base': f"{extract_scalar(coeffs.get('C_m', 0.0)):.4f}",
                                'd_alpha': f"{extract_scalar(coeffs.get('dC_m_da', 0.0)):.4f}",
                                'd_beta': f"{extract_scalar(coeffs.get('dC_m_db', 0.0)):.4f}",
                                'd_mach': f"{extract_scalar(coeffs.get('dC_m_dM', 0.0)):.4f}"
                            }
                        ]
                        
                        results_table.classes(remove='hidden')
                        ui.notify('Solve Complete!', type='positive', position='bottom-right')
                        
                    # Handle backend crashes gracefully
                    elif data['type'] == 'error':
                        ui.notify(f"Backend Error: {data['message']}", type='negative')
                        
        except httpx.ConnectError:
            ui.notify('Connection failed. Is the FastAPI server running?', type='negative')
        except Exception as e:
            ui.notify(f'Stream Error: {str(e)}', type='negative')
        finally:
            run_button.enable()
            loading_indicator.classes(add='hidden')

# --- 5. MAIN LAYOUT ---
with ui.header().classes('bg-slate-800 flex items-center justify-between w-full'):
    ui.label('RCAIDE Conceptual Design').classes('text-xl font-bold text-white')
    
    # Action Buttons on the right
    with ui.row().classes('items-center'):
        # .props('flat') removes the button background so it looks like a clean nav link
        ui.button('Load Project', on_click=upload_dialog.open, icon='file_upload').props('flat color=white')
        ui.button('Save Project', on_click=save_project, icon='save').props('flat color=white')

with ui.left_drawer().classes('p-4 border-r w-80'):
    ui.label('Vehicle Tree').classes('text-lg font-bold mb-2')
    vehicle_tree()  # Call the refreshable component
    
    ui.separator().classes('my-4')
    attribute_sliders()  # Call the refreshable component

# Right Panel: Flight Regime & Execution
with ui.right_drawer().props('width=450').classes('p-4 border-l'):
    ui.label('Flight Regime').classes('text-lg font-bold mb-2')
    
    ui.number('Mach Number', value=app_state['mach'], step=0.05).bind_value(app_state, 'mach').classes('w-full')
    ui.number('Angle of Attack (°)', value=app_state['alpha'], step=0.5).bind_value(app_state, 'alpha').classes('w-full mt-2')
    ui.number('Sideslip (°)', value=app_state['beta'], step=0.5).bind_value(app_state, 'beta').classes('w-full mt-2')
    
    ui.separator().classes('my-4')
    run_button = ui.button('Run Analysis', on_click=run_analysis, color='blue').classes('w-full')

    with ui.row().classes('w-full items-center justify-center mt-4 hidden') as loading_indicator:
        ui.spinner('orbit', size='md', color='blue')
        # We assign this to a variable so we can change its text on the fly
        loading_label = ui.label('Initializing...').classes('ml-2 text-gray-600 font-medium')

    # Results Table
    ui.separator().classes('my-4')
    ui.label('Aerodynamic Coefficients').classes('text-lg font-bold mb-2')
    
    # We use standard partial derivative notation for the headers
    result_columns = [
        {'name': 'coeff', 'label': 'Coeff', 'field': 'coeff', 'align': 'left'},
        {'name': 'base', 'label': 'Base', 'field': 'base', 'align': 'right'},
        {'name': 'd_alpha', 'label': '∂/∂α', 'field': 'd_alpha', 'align': 'right'},
        {'name': 'd_beta', 'label': '∂/∂β', 'field': 'd_beta', 'align': 'right'},
        {'name': 'd_mach', 'label': '∂/∂M', 'field': 'd_mach', 'align': 'right'}
    ]
    
    # .props('dense') shrinks the padding to fit all these columns neatly
    results_table = ui.table(columns=result_columns, rows=[], row_key='coeff') \
        .classes('w-full hidden').props('dense flat bordered')

with ui.element('div').classes('w-full h-[calc(100vh-58px)] p-0 m-0 bg-white flex justify-center items-center'):
    plot = ui.plotly(generate_wing_mesh()).classes('w-full h-full')

# The silent first-load update hack
ui.timer(0.1, update_plot, once=True)

ui.run(title='RCAIDE-EDEn Configurator')