import asyncio
import math
from nicegui import ui
import plotly.graph_objects as go

# --- Math Helpers ---
def get_great_circle_point(lat1, lon1, lat2, lon2, fraction):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    d = 2 * math.asin(math.sqrt(math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2))
    if d == 0: return math.degrees(lat1), math.degrees(lon1)
    A = math.sin((1 - fraction) * d) / math.sin(d)
    B = math.sin(fraction * d) / math.sin(d)
    x = A * math.cos(lat1) * math.cos(lon1) + B * math.cos(lat2) * math.cos(lon2)
    y = A * math.cos(lat1) * math.sin(lon1) + B * math.cos(lat2) * math.sin(lon2)
    z = A * math.sin(lat1) + B * math.sin(lat2)
    return math.degrees(math.atan2(z, math.sqrt(x**2 + y**2))), math.degrees(math.atan2(y, x))

def get_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dLon = lon2 - lon1
    y = math.sin(dLon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dLon)
    return (math.degrees(math.atan2(y, x)) + 360) % 360

# --- State Management & Coordinates ---
mission_state = {
    'takeoff': 'JFK (New York)',
    'landing': 'LHR (London)',
    'altitude': 35000,
    'segment': 'PRE-FLIGHT',
    'is_playing': False,
    'current_alt': 0, 'current_mach': 0.0, 'alpha': 0.0,
    'cl': 0.0, 'cd': 0.0, 'l_d': 0.0, 'throttle': 0.0, 'fuel_burn': 0.0
}

JFK_LAT, JFK_LNG = 40.6413, -73.7781
LHR_LAT, LHR_LNG = 51.4700, -0.4543
route_points = [get_great_circle_point(JFK_LAT, JFK_LNG, LHR_LAT, LHR_LNG, i/100) for i in range(101)]

# --- PRE-GENERATE MOCK DATA ARRAYS ---
flight_profile = {
    'Time/Progress': [i/200.0 for i in range(201)],
    'Alt (ft)': [], 'Mach': [], 'Alpha (deg)': [], 'CL': [], 'CD': [], 'L/D': [], 'Throttle (%)': [], 'Fuel Burn (lb/hr)': []
}

for step in range(201):
    prog = step / 200.0
    if prog <= 0:
        for k in ['Alt (ft)', 'Mach', 'Alpha (deg)', 'CL', 'CD', 'Throttle (%)', 'Fuel Burn (lb/hr)']: flight_profile[k].append(0.0)
    elif prog < 0.25:
        cp = prog / 0.25
        flight_profile['Alt (ft)'].append(35000 * cp)
        flight_profile['Mach'].append(0.78 * cp)
        flight_profile['Alpha (deg)'].append(6.5 + (step % 3) * 0.2)
        flight_profile['CL'].append(0.650 + (step % 2) * 0.005)
        flight_profile['CD'].append(0.0450 + (step % 2) * 0.001)
        flight_profile['Throttle (%)'].append(88.5 + (step % 4) * 0.3)
        flight_profile['Fuel Burn (lb/hr)'].append(6500 + (step % 5) * 20)
    elif prog < 0.75:
        flight_profile['Alt (ft)'].append(35000 + (step % 10 - 5) * 5)
        flight_profile['Mach'].append(0.78 + (step % 8 - 4) * 0.001)
        flight_profile['Alpha (deg)'].append(2.5 + (step % 4 - 2) * 0.1)
        flight_profile['CL'].append(0.450 + (step % 3 - 1) * 0.002)
        flight_profile['CD'].append(0.0250 + (step % 2) * 0.0005)
        flight_profile['Throttle (%)'].append(62.0 + (step % 3 - 1) * 0.2)
        flight_profile['Fuel Burn (lb/hr)'].append(3200 + (step % 4 - 2) * 10)
    elif prog < 1.0:
        dp = (prog - 0.75) / 0.25
        flight_profile['Alt (ft)'].append(35000 - (35000 * dp))
        flight_profile['Mach'].append(0.78 - (0.78 * dp))
        flight_profile['Alpha (deg)'].append(0.5 - (step % 2) * 0.1)
        flight_profile['CL'].append(0.300)
        flight_profile['CD'].append(0.0200)
        flight_profile['Throttle (%)'].append(15.0)
        flight_profile['Fuel Burn (lb/hr)'].append(1200)
    else:
        for k in ['Alt (ft)', 'Mach', 'Alpha (deg)', 'CL', 'CD', 'Throttle (%)', 'Fuel Burn (lb/hr)']: flight_profile[k].append(0.0)

for i in range(201):
    c_d = flight_profile['CD'][i]
    flight_profile['L/D'].append(flight_profile['CL'][i] / c_d if c_d > 0 else 0)

plot_variables = ['Alt (ft)', 'Mach', 'Alpha (deg)', 'CL', 'CD', 'L/D', 'Throttle (%)', 'Fuel Burn (lb/hr)']

def create_plot(var_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=flight_profile['Time/Progress'], y=flight_profile[var_name], mode='lines', line=dict(color='#3b82f6', width=2)))
    fig.update_layout(
        margin=dict(l=30, r=20, t=10, b=30),
        xaxis=dict(title='Mission Progress', tickformat='.0%', gridcolor='#e2e8f0'),
        yaxis=dict(gridcolor='#e2e8f0'),
        height=220,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

@ui.page('/')
def index():
    ui.add_head_html('<style>.leaflet-container { background: #0e0e0e !important; }</style>')
    
    # --- Drawer Toggle & Map Recenter Logic ---
    async def toggle_drawer():
        right_drawer.toggle()
        # Wait 300ms for the drawer CSS transition to complete
        await asyncio.sleep(0.3)
        # Force Leaflet to recalculate its internal bounds
        map_view.run_map_method('invalidateSize')
        # Re-trigger the state update to snap the camera exactly onto the plane
        update_state(time_slider.value)

    # --- RIGHT DRAWER (Results Data) ---
    right_drawer = ui.right_drawer(fixed=False).props('width=450').classes('border-l shadow-2xl w-[400px] p-4 z-50').props('bordered')
    
    with right_drawer:
        with ui.row().classes('w-full justify-between items-center mb-4'):
            ui.label('Mission Results').classes('text-xl font-bold text-slate-800')
            # Updated to use the new toggle function
            ui.button(icon='close', on_click=toggle_drawer).props('flat round size=sm').classes('text-slate-500')
        
        ui.label('Select variables to view full simulation traces.').classes('text-sm text-slate-500 mb-6')
        
        ui.select(plot_variables, value='Alt (ft)', label='Plot 1 Variable', on_change=lambda e: plot1.update_figure(create_plot(e.value))).classes('w-full mb-2')
        plot1 = ui.plotly(create_plot('Alt (ft)')).classes('w-full mb-6 border bg-white rounded-lg shadow-sm')
        
        ui.select(plot_variables, value='Mach', label='Plot 2 Variable', on_change=lambda e: plot2.update_figure(create_plot(e.value))).classes('w-full mb-2')
        plot2 = ui.plotly(create_plot('Mach')).classes('w-full mb-6 border bg-white rounded-lg shadow-sm')
        
        ui.select(plot_variables, value='Fuel Burn (lb/hr)', label='Plot 3 Variable', on_change=lambda e: plot3.update_figure(create_plot(e.value))).classes('w-full mb-2')
        plot3 = ui.plotly(create_plot('Fuel Burn (lb/hr)')).classes('w-full border bg-white rounded-lg shadow-sm')
        
    right_drawer.hide()

    with ui.row().classes('w-full h-screen wrap-none m-0 p-0'):
        
        # LEFT PANEL
        with ui.column().classes('w-[350px] p-6 border-r h-full shadow-lg z-20'):
            ui.label('B737 Mission Simulator').classes('text-2xl font-bold mb-6 text-slate-800')
            ui.label('Main Wing Geometry').classes('text-sm font-bold text-slate-500 uppercase')
            ui.number('Span (m)', value=35.9).classes('w-full')
            ui.number('Root Chord (m)', value=7.9).classes('w-full')
            ui.number('Sweep Angle (deg)', value=25.0).classes('w-full')
            ui.separator().classes('my-6')
            ui.label('Flight Profile').classes('text-sm font-bold text-slate-500 uppercase')
            ui.input('Takeoff', value=mission_state['takeoff']).classes('w-full')
            ui.input('Landing', value=mission_state['landing']).classes('w-full')
            ui.number('Cruise Alt', value=mission_state['altitude']).classes('w-full')
            
            play_btn = ui.button('Play Mission', icon='play_arrow', on_click=lambda: toggle_play()).classes('w-full mt-8')
            
        # RIGHT PANEL
        with ui.column().classes('flex-grow h-full relative overflow-hidden bg-white p-8'):
            with ui.element('div').classes('relative w-full h-full border-[16px] border-stone-900 rounded-3xl overflow-hidden shadow-2xl bg-black').style('perspective: 1000px;'):
                
                # MAP LAYER
                map_view = ui.leaflet(center=(JFK_LAT, JFK_LNG), zoom=7) \
                    .classes('absolute inset-0 w-full h-full') \
                    .style('transform-origin: center; transform: rotateX(25deg) scale(1.3);')
                map_view.clear_layers()
                map_view.tile_layer(
                    url_template='https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
                    options={'maxZoom': 10, 'attribution': '&copy; CARTO'}
                )
                map_view.marker(latlng=(JFK_LAT, JFK_LNG)) 
                map_view.marker(latlng=(LHR_LAT, LHR_LNG))
                map_view.generic_layer(name='polyline', args=[route_points, {'color': '#ef4444', 'weight': 4, 'dashArray': '10, 15'}])
            
                # TELEMETRY DASHBOARD
                with ui.row().classes('absolute top-8 left-1/2 -translate-x-1/2 z-10 bg-slate-900/90 text-white p-4 rounded-xl shadow-2xl backdrop-blur-md border border-slate-700 w-11/12 justify-around items-center'):
                    def telemetry_block(label, key, formatter=None, is_phase=False):
                        with ui.column().classes('items-center gap-1'):
                            ui.label(label).classes('text-[10px] text-slate-400 font-bold uppercase tracking-wider')
                            if is_phase: ui.label().bind_text_from(mission_state, key).classes('font-bold text-lg text-green-400')
                            else: ui.label().bind_text_from(mission_state, key, backward=formatter).classes('font-mono text-lg text-slate-100')

                    telemetry_block('Phase', 'segment', is_phase=True)
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
                
                # DRAWER TOGGLE BUTTON (Updated to use the new toggle function)
                ui.button(icon='analytics', on_click=toggle_drawer).classes('absolute top-8 right-8 z-20 bg-blue-600 shadow-xl').props('round')

                # TIMELINE DASHBOARD
                with ui.column().classes('absolute bottom-8 left-1/2 -translate-x-1/2 z-10 bg-slate-900/90 p-4 rounded-xl shadow-2xl backdrop-blur-md border border-slate-700 w-11/12 gap-0'):
                    with ui.row().classes('w-full flex flex-nowrap whitespace-nowrap text-[10px] text-slate-500 font-bold uppercase tracking-wide px-2 mb-[-12px]'):
                        ui.label('CLIMB').classes('w-1/4 text-center border-l border-slate-600')
                        ui.label('CRUISE').classes('w-2/4 text-center border-l border-r border-slate-600')
                        ui.label('DESCEND').classes('w-1/4 text-center border-r border-slate-600')
                    
                    time_slider = ui.slider(
                        min=0, max=1, step=0.005, value=0, 
                        on_change=lambda e: update_state(e.value)
                    ).classes('w-full').props('color="blue-4" track-size="4px" thumb-size="16px"')
            
                # 3D AIRCRAFT OVERLAY
                with ui.column().classes('absolute inset-0 z-10 items-center justify-center pointer-events-none'):
                    plane_icon = ui.icon('flight', size='128px').classes('text-blue-400 transition-all duration-300 drop-shadow-md')

    # --- Centralized State Manager ---
    def update_state(progress):
        step = int(progress * 200)
        
        if progress >= 1.0:
            lat1, lng1 = get_great_circle_point(JFK_LAT, JFK_LNG, LHR_LAT, LHR_LNG, 0.999)
            current_lat, current_lng = get_great_circle_point(JFK_LAT, JFK_LNG, LHR_LAT, LHR_LNG, 1.0)
            bearing = get_bearing(lat1, lng1, current_lat, current_lng)
        else:
            current_lat, current_lng = get_great_circle_point(JFK_LAT, JFK_LNG, LHR_LAT, LHR_LNG, progress)
            lat2, lng2 = get_great_circle_point(JFK_LAT, JFK_LNG, LHR_LAT, LHR_LNG, progress + 0.001)
            bearing = get_bearing(current_lat, current_lng, lat2, lng2)

        mission_state['current_alt'] = flight_profile['Alt (ft)'][step]
        mission_state['current_mach'] = flight_profile['Mach'][step]
        mission_state['alpha'] = flight_profile['Alpha (deg)'][step]
        mission_state['cl'] = flight_profile['CL'][step]
        mission_state['cd'] = flight_profile['CD'][step]
        mission_state['l_d'] = flight_profile['L/D'][step]
        mission_state['throttle'] = flight_profile['Throttle (%)'][step]
        mission_state['fuel_burn'] = flight_profile['Fuel Burn (lb/hr)'][step]

        plane_icon.classes(remove='drop-shadow-[0_35px_35px_rgba(0,0,0,0.5)] drop-shadow-md')
        base_rot = bearing - 45 
        pitch_mod = 0 

        if progress <= 0:
            mission_state['segment'] = 'PRE-FLIGHT'
            plane_icon.classes(add='drop-shadow-md')
        elif progress < 0.25:
            mission_state['segment'] = 'CLIMB'
            plane_icon.classes(add='drop-shadow-[0_35px_35px_rgba(0,0,0,0.5)]')
            pitch_mod = -15
        elif progress < 0.75:
            mission_state['segment'] = 'CRUISE'
            plane_icon.classes(add='drop-shadow-[0_35px_35px_rgba(0,0,0,0.5)]')
            pitch_mod = 0 
        elif progress < 1.0:
            mission_state['segment'] = 'DESCEND'
            plane_icon.classes(add='drop-shadow-md')
            pitch_mod = 10 
        else:
            mission_state['segment'] = 'LANDED'
            plane_icon.classes(add='drop-shadow-md')
            
        plane_icon.style(f'transform: rotate({base_rot + pitch_mod}deg);')

        alt_ratio = mission_state['current_alt'] / max(1, mission_state['altitude'])
        map_view.run_map_method('setView', [current_lat, current_lng], 7 - (3 * alt_ratio), {'animate': False})

    # --- Play/Pause Logic ---
    async def toggle_play():
        if mission_state['is_playing']:
            mission_state['is_playing'] = False
            play_btn.set_text('Play Mission')
            play_btn.props('icon=play_arrow color=primary')
        else:
            if time_slider.value >= 1.0:
                time_slider.value = 0.0 
                await asyncio.sleep(0.2)
                
            mission_state['is_playing'] = True
            play_btn.set_text('Pause Mission')
            play_btn.props('icon=pause color=warning')
            
            while mission_state['is_playing'] and time_slider.value < 1.0:
                time_slider.value = min(1.0, time_slider.value + 0.005) 
                await asyncio.sleep(0.05)
                
            if time_slider.value >= 1.0:
                mission_state['is_playing'] = False
                play_btn.set_text('Play Mission')
                play_btn.props('icon=play_arrow color=primary')

ui.run()