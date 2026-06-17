# utils/state.py

import plotly

master_state = {
    'is_dark': True,
    'on_theme_changed': [],
    'hangar': {
        'mach': 0.3, 'alpha': 5.0, 'beta': 0.0,
        'root_chord': 2.0, 'root_twist': 0.0,
        'segment_counter': 1,
        'segments': [
            {'id': 'seg_1', 'name': 'Segment 1', 'span': 5.0, 'taper': 0.5, 'sweep': 20.0, 'dihedral': 5.0, 'twist': -2.0}
        ],
        'selected_id': 'root'
    },
    'simulator': {
        'takeoff': 'JFK',
        'landing': 'LHR',
        'altitude': 35000,
        'segment': 'PRE-FLIGHT',
        'is_playing': False,
        'current_alt': 0, 'current_mach': 0.0, 'alpha': 0.0,
        'cl': 0.0, 'cd': 0.0, 'l_d': 0.0, 'throttle': 0.0, 'fuel_burn': 0.0
    }
}

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