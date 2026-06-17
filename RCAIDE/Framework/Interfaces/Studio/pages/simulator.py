import asyncio
from nicegui import ui
from components.navigation import navigation_header
from utils.flight_math import get_great_circle_point, get_bearing
from utils.mock_data import flight_profile, plot_variables, create_plot
from utils.state import master_state

JFK_LAT, JFK_LNG = 40.6413, -73.7781
LHR_LAT, LHR_LNG = 51.4700, -0.4543
route_points = [get_great_circle_point(JFK_LAT, JFK_LNG, LHR_LAT, LHR_LNG, i/100) for i in range(101)]

@ui.page('/simulator')
def simulator_page():
    navigation_header()
    sim_state = master_state['simulator']
    
    ui.add_head_html('<style>.leaflet-container { background: #0e0e0e !important; }</style>')
    # Nuke the invisible 16px padding on the center container so the map sits flush too
    ui.add_css('.q-page { padding: 0 !important; } .nicegui-content { padding: 0 !important; }')

    # --- 1. LEFT DRAWER (Configuration Controls) ---
    with ui.left_drawer().props('width=350').classes('bg-slate-50 p-6 border-r'):
        ui.label('B737 Mission Simulator').classes('text-2xl font-bold mb-6 text-slate-800')
        ui.label('Main Wing Geometry').classes('text-sm font-bold text-slate-500 uppercase')
        ui.number('Span (m)', value=35.9).classes('w-full')
        ui.number('Root Chord (m)', value=7.9).classes('w-full')
        ui.number('Sweep Angle (deg)', value=25.0).classes('w-full')
        ui.separator().classes('my-6')
        ui.label('Flight Profile').classes('text-sm font-bold text-slate-500 uppercase')
        ui.input('Takeoff', value=sim_state['takeoff']).classes('w-full')
        ui.input('Landing', value=sim_state['landing']).classes('w-full')
        ui.number('Cruise Alt', value=sim_state['altitude']).classes('w-full')
        
        play_btn = ui.button('Play Mission', icon='play_arrow', on_click=lambda: toggle_play()).classes('w-full mt-8')

    # --- 2. RIGHT DRAWER (Mission Results) ---
    with ui.right_drawer().props('width=450').classes('bg-slate-50 p-6 border-l'):
        ui.label('Mission Results').classes('text-xl font-bold text-slate-800 mb-2')
        ui.label('Select variables to view full simulation traces.').classes('text-sm text-slate-500 mb-6')
        
        ui.select(plot_variables, value='Alt (ft)', label='Plot 1 Variable', on_change=lambda e: plot1.update_figure(create_plot(e.value))).classes('w-full mb-2')
        plot1 = ui.plotly(create_plot('Alt (ft)')).classes('w-full mb-6 border bg-white rounded-lg shadow-sm')
        
        ui.select(plot_variables, value='Mach', label='Plot 2 Variable', on_change=lambda e: plot2.update_figure(create_plot(e.value))).classes('w-full mb-2')
        plot2 = ui.plotly(create_plot('Mach')).classes('w-full mb-6 border bg-white rounded-lg shadow-sm')
        
        ui.select(plot_variables, value='Fuel Burn (lb/hr)', label='Plot 3 Variable', on_change=lambda e: plot3.update_figure(create_plot(e.value))).classes('w-full mb-2')
        plot3 = ui.plotly(create_plot('Fuel Burn (lb/hr)')).classes('w-full border bg-white rounded-lg shadow-sm')

    # --- 3. CENTER PANEL (Visualizer/Map HUD) ---
    with ui.element('div').classes('w-full h-[calc(100vh-58px)] p-6 bg-white overflow-hidden'):
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
            map_view.generic_layer(name='polyline', args=[route_points, {'color': "#07afbb", 'weight': 4, 'dashArray': '10, 15'}])
        
            # TELEMETRY DASHBOARD
            with ui.row().classes('absolute top-8 left-1/2 -translate-x-1/2 z-10 bg-slate-900/90 text-white p-4 rounded-xl shadow-2xl backdrop-blur-md border border-slate-700 w-11/12 justify-around items-center'):
                def telemetry_block(label, key, formatter=None, is_phase=False):
                    with ui.column().classes('items-center gap-1'):
                        ui.label(label).classes('text-[10px] text-slate-400 font-bold uppercase tracking-wider')
                        if is_phase: ui.label().bind_text_from(sim_state, key).classes('font-bold text-lg text-green-400')
                        else: ui.label().bind_text_from(sim_state, key, backward=formatter).classes('font-mono text-lg text-slate-100')

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

        sim_state['current_alt'] = flight_profile['Alt (ft)'][step]
        sim_state['current_mach'] = flight_profile['Mach'][step]
        sim_state['alpha'] = flight_profile['Alpha (deg)'][step]
        sim_state['cl'] = flight_profile['CL'][step]
        sim_state['cd'] = flight_profile['CD'][step]
        sim_state['l_d'] = flight_profile['L/D'][step]
        sim_state['throttle'] = flight_profile['Throttle (%)'][step]
        sim_state['fuel_burn'] = flight_profile['Fuel Burn (lb/hr)'][step]

        plane_icon.classes(remove='drop-shadow-[0_35px_35px_rgba(0,0,0,0.5)] drop-shadow-md')
        base_rot = bearing
        pitch_mod = 0 

        if progress <= 0:
            sim_state['segment'] = 'PRE-FLIGHT'
            plane_icon.classes(add='drop-shadow-md')
        elif progress < 0.25:
            sim_state['segment'] = 'CLIMB'
            plane_icon.classes(add='drop-shadow-[0_35px_35px_rgba(0,0,0,0.5)]')
            pitch_mod = -15
        elif progress < 0.75:
            sim_state['segment'] = 'CRUISE'
            plane_icon.classes(add='drop-shadow-[0_35px_35px_rgba(0,0,0,0.5)]')
            pitch_mod = 0 
        elif progress < 1.0:
            sim_state['segment'] = 'DESCEND'
            plane_icon.classes(add='drop-shadow-md')
            pitch_mod = -15
        else:
            sim_state['segment'] = 'LANDED'
            plane_icon.classes(add='drop-shadow-md')
            
        plane_icon.style(f'transform: rotate({base_rot + pitch_mod}deg);')

        alt_ratio = sim_state['current_alt'] / max(1, sim_state['altitude'])
        map_view.run_map_method('setView', [current_lat, current_lng], 7 - (3 * alt_ratio), {'animate': False})

    # --- Play/Pause Logic ---
    async def toggle_play():
        if sim_state.get('is_playing', False):
            sim_state['is_playing'] = False
            play_btn.set_text('Play Mission')
            play_btn.props('icon=play_arrow color=primary')
        else:
            if time_slider.value >= 1.0:
                time_slider.value = 0.0 
                await asyncio.sleep(0.2)
                
            sim_state['is_playing'] = True
            play_btn.set_text('Pause Mission')
            play_btn.props('icon=pause color=warning')
            
            while sim_state['is_playing'] and time_slider.value < 1.0:
                time_slider.value = min(1.0, time_slider.value + 0.005) 
                await asyncio.sleep(0.05)
                
            if time_slider.value >= 1.0:
                sim_state['is_playing'] = False
                play_btn.set_text('Play Mission')
                play_btn.props('icon=play_arrow color=primary')