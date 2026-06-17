# utils/state.py

import plotly

master_state = {
    'is_dark': True,
    'on_theme_changed': [],
    'hangar': {
        'mach': 0.3, 'alpha': 5.0, 'beta': 0.0,
        'selected_id': 'wing_0', 
        'vehicle': {
            'wings': [
                {
                    'id': 'wing_0', 'name': 'Main Wing', 'symmetric': True,
                    'x_offset': 10.0, 'y_offset': 0.0, 'z_offset': 0.0,
                    'root_chord': 4.0, 'root_twist': 0.0,
                    'segments': [
                        {'id': 'wing_0_seg_0', 'name': 'Inboard', 'span': 8.0, 'taper': 0.8, 'sweep': 25.0, 'dihedral': 3.0, 'twist': -1.0},
                        {'id': 'wing_0_seg_1', 'name': 'Outboard', 'span': 6.0, 'taper': 0.5, 'sweep': 30.0, 'dihedral': 5.0, 'twist': -2.0}
                    ]
                },
                {
                    'id': 'wing_1', 'name': 'Horizontal Tail', 'symmetric': True,
                    'x_offset': 32.0, 'y_offset': 0.0, 'z_offset': 1.5,
                    'root_chord': 2.0, 'root_twist': 0.0,
                    'segments': [{'id': 'wing_1_seg_0', 'name': 'Tail Span', 'span': 5.0, 'taper': 0.6, 'sweep': 35.0, 'dihedral': 0.0, 'twist': 0.0}]
                },
                {
                    'id': 'wing_2', 'name': 'Vertical Tail', 'symmetric': False,
                    'x_offset': 31.0, 'y_offset': 0.0, 'z_offset': 1.5,
                    'root_chord': 3.5, 'root_twist': 0.0,
                    'segments': [{'id': 'wing_2_seg_0', 'name': 'Fin Span', 'span': 6.0, 'taper': 0.4, 'sweep': 40.0, 'dihedral': 90.0, 'twist': 0.0}] # 90 deg dihedral points it straight up
                }
            ],
            'fuselages': [
                 {'id': 'fuse_0', 'name': 'Main Fuselage', 'length': 38.0, 'diameter': 3.5, 'x_offset': 0.0, 'y_offset': 0.0, 'z_offset': 0.0}
            ],
            'nacelles': [
                 {'id': 'nac_0', 'name': 'Main Engines', 'length': 4.5, 'diameter': 1.8, 'x_offset': 12.0, 'y_offset': 4.5, 'z_offset': -1.5, 'symmetric': True}
            ],
            'propulsion': {
                 'bypass_ratio': 5.0,
                 'max_fuel_kg': 20000.0
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