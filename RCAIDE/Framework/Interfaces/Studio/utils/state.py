# utils/state.py
from nicegui import ui
import copy
import plotly

app_state = {
    'is_dark': True,
    'on_theme_changed': [],
    'history_stack': [],
    'redo_stack': [],
    'route': '/hangar',
    'hangar': {
        'mach': 0.78, 'alpha': 2.5, 'beta': 0.0,
        'selected_id': 'wing_0', 
        'vehicle': {
            'wings': [
                {
                    'id': 'wing_0', 'name': 'Main Wing', 'symmetric': True,
                    'x_offset': 13.0, 'y_offset': 0.0, 'z_offset': -0.5,
                    'root_chord': 6.5, 'root_twist': 2.0,
                    'segments': [
                        {'id': 'wing_0_seg_0', 'name': 'Inboard', 'span': 5.0, 'taper': 0.75, 'sweep': 15.0, 'dihedral': 6.0, 'twist': -0.5},
                        {'id': 'wing_0_seg_1', 'name': 'Outboard', 'span': 12.9, 'taper': 0.3, 'sweep': 25.0, 'dihedral': 6.0, 'twist': -2.0}
                    ]
                },
                {
                    'id': 'wing_1', 'name': 'Horizontal Tail', 'symmetric': True,
                    'x_offset': 34.0, 'y_offset': 0.0, 'z_offset': 1.0,
                    'root_chord': 3.5, 'root_twist': 0.0,
                    'segments': [
                        {'id': 'wing_1_seg_0', 'name': 'Tail Span', 'span': 7.1, 'taper': 0.35, 'sweep': 32.0, 'dihedral': 0.0, 'twist': 0.0}
                    ]
                },
                {
                    'id': 'wing_2', 'name': 'Vertical Tail', 'symmetric': False,
                    'x_offset': 32.5, 'y_offset': 0.0, 'z_offset': 1.8,
                    'root_chord': 4.5, 'root_twist': 0.0,
                    'segments': [
                        {'id': 'wing_2_seg_0', 'name': 'Fin Span', 'span': 7.2, 'taper': 0.35, 'sweep': 35.0, 'dihedral': 90.0, 'twist': 0.0}
                    ]
                }
            ],
            'fuselages': [
                 {'id': 'fuse_0', 'name': '737 Fuselage', 'length': 39.5, 'diameter': 3.76, 'x_offset': 0.0, 'y_offset': 0.0, 'z_offset': 0.0}
            ],
            'nacelles': [
                 {'id': 'nac_0', 'name': 'CFM56 Engines', 'length': 3.0, 'diameter': 2.1, 'x_offset': 15.5, 'y_offset': 4.8, 'z_offset': -1.6, 'symmetric': True}
            ],
            'propulsion': {
                 'bypass_ratio': 5.4,
                 'max_fuel_kg': 20800.0
            }
        }
    },
    'simulator': {
        'takeoff': 'JFK',
        'landing': 'LHR',
        'altitude': 35000,
        'segment': 'PRE-FLIGHT',
        'is_playing': False,
        'current_alt': 0, 'current_mach': 0.0, 'alpha': 0.0,
        'cl': 0.0, 'cd': 0.0, 'l_d': 0.0, 'throttle': 0.0, 'fuel_burn': 0.0
    },
    'engine':{
        'design': {'mach': 0.2, 'alt': 5.0, 'thrust': 50.0,},
        'selected_id': '',
        'stations': {
            'inlet': {},
            'fan': {},
            'lpc': {},
            'hpc': {},
            'burner': {},
            'hpt': {},
            'lpt': {},
            'c_nozz': {},
            'f_nozz': {}

        },
    }
}

def save_snapshot():
    
    if len(app_state['history_stack']) >= 20:
        app_state['history_stack'].pop(0)
    
    clean_state = {k: v for k, v in app_state.items() if k not in ['on_theme_changed', 'history_stack', 'redo_stack']}
    app_state['history_stack'].append(copy.deepcopy(clean_state))
    
    app_state['redo_stack'].clear()

def perform_undo():
    if not app_state['history_stack']:
        ui.notify('Nothing to undo.')
        return
    
    # Save current state to redo stack before going back
    clean_state = {k: v for k, v in app_state.items() if k not in ['on_theme_changed', 'history_stack', 'redo_stack']}
    app_state['redo_stack'].append(copy.deepcopy(clean_state))
    
    previous_state = app_state['history_stack'].pop()
    for key in previous_state:
        app_state[key] = previous_state[key]
        
    from main import router_content
    router_content.refresh()

def perform_redo():
    if not app_state['redo_stack']:
        ui.notify('Nothing to redo.')
        return
        
    # Save current state to undo stack before going forward
    clean_state = {k: v for k, v in app_state.items() if k not in ['on_theme_changed', 'history_stack', 'redo_stack']}
    app_state['history_stack'].append(copy.deepcopy(clean_state))
    
    next_state = app_state['redo_stack'].pop()
    for key in next_state:
        app_state[key] = next_state[key]
        
    from main import router_content
    router_content.refresh()

theme_config = {
    True: {  # Dark Mode
        'map_tiles': 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
        'plotly_template': 'plotly_dark',
        'plot_bg': 'rgba(0,0,0,0)',
        'text_color': '#ffffff',  # Force white text
        'colorway': plotly.colors.sequential.Plasma,
    },
    False: {  # Light Mode
        'map_tiles': 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
        'plotly_template': 'plotly_white',
        'text_color': '#1e293b',  # Slate-800 for light mode readability
        'colorway': plotly.colors.sequential.ice, 
    }
}