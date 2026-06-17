# utils/state.py

master_state = {
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
        'takeoff': 'JFK (New York)',
        'landing': 'LHR (London)',
        'altitude': 35000,
        'segment': 'PRE-FLIGHT',
        'is_playing': False,
        'current_alt': 0, 'current_mach': 0.0, 'alpha': 0.0,
        'cl': 0.0, 'cd': 0.0, 'l_d': 0.0, 'throttle': 0.0, 'fuel_burn': 0.0
    }
}