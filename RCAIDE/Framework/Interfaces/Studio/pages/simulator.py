import asyncio

from nicegui import ui
from plotly import graph_objects as go

from utils.state import app_state, theme_config
from utils.flight_math import get_great_circle_point, get_bearing, generate_procedural_flight_profile, normalize_lng, AIRPORT_DB

def mock_flight_profile():
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

    return flight_profile, plot_variables

flight_profile, plot_variables = mock_flight_profile()

def load_mission_profile():
    """Generates calculations and refreshes global runtime variables."""
    prof, vars_list, meta = generate_procedural_flight_profile(
        app_state['simulator']['takeoff'], app_state['simulator']['landing']
    )
    # Package into our simulator tracking state
    app_state['simulator']['profile'] = prof
    app_state['simulator']['meta'] = meta
    app_state['simulator']['vars'] = vars_list
    
    # Seed index 0 values into the HUD variables
    update_hud_to_step(0)

def update_hud_to_step(step_idx):
    """Binds index elements to NiceGUI labels during slider manipulation."""
    prof = app_state['simulator']['profile']
    # Update individual reactive elements

    if step_idx <= 0:
        app_state['simulator']['segment'] = 'PRE-FLIGHT'
    elif step_idx / 200 < 0.25:
        app_state['simulator']['segment'] = 'CLIMB'
    elif step_idx / 200 < 0.75:
        app_state['simulator']['segment'] = 'CRUISE'
    elif step_idx / 200 < 1.0:
        app_state['simulator']['segment'] = 'DESCEND'
    else:
        app_state['simulator']['segment'] = 'LANDED'

    app_state['simulator']['current_alt'] = prof['Alt (ft)'][step_idx]
    app_state['simulator']['current_mach'] = prof['Mach'][step_idx]
    app_state['simulator']['alpha'] = prof['Alpha (deg)'][step_idx]
    app_state['simulator']['cl'] = prof['CL'][step_idx]
    app_state['simulator']['cd'] = prof['CD'][step_idx]
    app_state['simulator']['l_d'] = prof['L/D'][step_idx]
    app_state['simulator']['throttle'] = prof['Throttle (%)'][step_idx]
    app_state['simulator']['fuel_burn'] = prof['Fuel Burn (lb/hr)'][step_idx]

def create_plot(var_name):

    current_theme = theme_config[app_state.get('is_dark', False)]
    colorway = current_theme['colorway']
    
    var_index = plot_variables.index(var_name)
    assigned_color = colorway[var_index % len(colorway)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=flight_profile['Time/Progress'],
        y=flight_profile[var_name],
        mode='lines',
        line=dict(color=assigned_color,
                  width=2)))
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=0, pad=3),
        xaxis=dict(
            title='Mission Progress',
            tickformat='.0%',
            gridcolor='rgba(128, 128, 128, 0.2)',
            automargin=True),
        yaxis=dict(
            gridcolor='rgba(128, 128, 128, 0.2)',
            automargin=True),
        height=220,
        template=current_theme['plotly_template'],
        font=dict(color=current_theme['text_color']),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig

def simulator_ui():
    # 1. Evaluate True Colors based on state
    is_dark = app_state.get('is_dark', False)
    
    sidebar_bg = 'bg-neutral-900' if is_dark else 'bg-gray-200'
    sidebar_border = 'border-neutral-800' if is_dark else 'border-gray-200'
        
    canvas_bg = 'bg-black text-gray-200' if is_dark else 'bg-white text-gray-800'

    # Fixed missing parenthesis on ui.column() and applied SPA w-full h-full
    with ui.column().classes('w-full h-full p-0 gap-0'):
        
        ui.add_head_html('''
            <style>
                .leaflet-container { background: #0e0e0e !important; }
                .leaflet-bottom { bottom: 10px !important; } 
            </style>
        ''')

        # SPLITTER 1: Left Panel
        with ui.splitter(value=350, limits=(250, 600)).classes('w-full h-full').props('unit="px"') as outer_split:
            
            with outer_split.before:
                # --- 1. LEFT PANEL (Configuration Controls) ---
                with ui.column().classes(f'w-full h-full p-6 border-r overflow-y-auto {sidebar_bg} {sidebar_border}'):
                    ui.label('Mission Simulator').classes('text-2xl font-bold mb-6')
                    ui.label('Main Wing Geometry').classes('text-sm font-bold uppercase')
                    ui.number('Span (m)', value=35.9).classes('w-full')
                    ui.number('Root Chord (m)', value=7.9).classes('w-full')
                    ui.number('Sweep Angle (deg)', value=25.0).classes('w-full')
                    
                    ui.separator().classes('my-6 opacity-50')

                    airport_codes = list(AIRPORT_DB.keys())
                    ui.label('Flight Profile').classes('text-sm font-bold uppercase')

                    ui.label('Takeoff Airport (IATA)').classes('text-xs text-gray-500 font-medium ml-2 -mb-2 z-10 relative')
                    with ui.row().classes('w-full items-center mb-2 flex-nowrap'):
                        orig_select = ui.select(
                            options=airport_codes, value=app_state['simulator']['takeoff'],
                            with_input=True, on_change=lambda e: on_route_change(e)
                        ).classes('w-1/6').props('dense hide-bottom-space')
                        
                        orig_name_display = ui.label(AIRPORT_DB[app_state['simulator']['takeoff']]['name'])\
                            .classes('w-4/6 pl-4 text-sm truncate')

                    ui.label('Destination Airport (IATA)').classes('text-xs text-gray-500 font-medium ml-2 -mb-2 z-10 relative')
                    with ui.row().classes('w-full items-center mb-4 flex-nowrap'):
                        dest_select = ui.select(
                            options=airport_codes, value=app_state['simulator']['landing'],
                            with_input=True, on_change=lambda e: on_route_change(e)
                        ).classes('w-1/6').props('dense hide-bottom-space')
                        
                        dest_name_display = ui.label(AIRPORT_DB[app_state['simulator']['landing']]['name'])\
                            .classes('w-4/6 pl-4 text-sm truncate')

                    def on_route_change(e):
                        if orig_select.value in AIRPORT_DB:
                            orig_name_display.set_text(AIRPORT_DB[orig_select.value]['name'])
                        if dest_select.value in AIRPORT_DB:
                            dest_name_display.set_text(AIRPORT_DB[dest_select.value]['name'])

                        app_state['simulator']['takeoff'] = orig_select.value
                        app_state['simulator']['landing'] = dest_select.value
                        
                        load_mission_profile()
                        refresh_map_layer()

                        meta = app_state['simulator']['meta']
                        map_view.set_center((meta['orig_lat'], meta['orig_lng']))
                    
                    ui.number('Cruise Alt', value=app_state['simulator']['altitude']).classes('w-full')        
                    play_btn = ui.button('Play Mission', icon='play_arrow', on_click=lambda: toggle_play()).classes('w-full mt-8')

            with outer_split.after:
                # SPLITTER 2: Right Panel
                with ui.splitter(value=450, limits=(300, 800), reverse=True).classes('w-full h-full').props('unit="px"') as inner_split:
                    
                    with inner_split.before:
                        # --- 3. CENTER PANEL (Visualizer/Map HUD) ---
                        # Use a relative div so the absolute overlays anchor perfectly
                        with ui.element('div').classes(f'w-full h-full relative overflow-hidden {canvas_bg}'):
                            
                            lockdown_options = {
                                'zoomControl': False, 'dragging': False, 'scrollWheelZoom': False,
                                'doubleClickZoom': False, 'touchZoom': False, 'boxZoom': False, 'keyboard': False
                            }

                            load_mission_profile()
                            orig_latlng = (app_state['simulator']['meta']['orig_lat'], app_state['simulator']['meta']['orig_lng'])

                            map_view = ui.leaflet(center=orig_latlng, zoom=7, options=lockdown_options) \
                                .classes('w-full h-full absolute inset-0') \
                                .style('transform-origin: center; transform: perspective(1000px) rotateX(30deg) scale(1.6); z-index: 1;')
                            
                            def refresh_map_layer():
                                
                                orig_lat = app_state['simulator']['meta']['orig_lat']
                                orig_lng = app_state['simulator']['meta']['orig_lng']
                                
                                # Normalize the destination longitude so the marker doesn't vanish
                                dest_lat = app_state['simulator']['meta']['dest_lat']
                                raw_dest_lng = app_state['simulator']['meta']['dest_lng']
                                dest_lng = normalize_lng(raw_dest_lng, orig_lng)

                                map_view.clear_layers()

                                orig_latlng = (orig_lat, orig_lng)
                                dest_latlng = (dest_lat, dest_lng)

                                map_view.clear_layers()
                                map_view.tile_layer(
                                    url_template=theme_config[app_state['is_dark']]['map_tiles'],
                                    options={'maxZoom': 10, 'attribution': '&copy; CARTO'}
                                )
                                map_view.marker(latlng=orig_latlng) 
                                map_view.marker(latlng=dest_latlng)
                                map_view.generic_layer(
                                    name='polyline',
                                    args=[
                                        app_state['simulator']['meta']['route_points'],
                                        {'color': "#07afbb", 'weight': 4, 'dashArray': '10, 15'}
                                    ]
                                )
                            
                            refresh_map_layer()

                            ui.run_javascript(f'''
                                setTimeout(() => {{
                                    const mapContainer = document.getElementById("c{map_view.id}");
                                    if (mapContainer) {{
                                        new ResizeObserver(() => {{
                                            window.dispatchEvent(new Event('resize'));
                                        }}).observe(mapContainer);
                                    }}
                                }}, 100);
                            ''')

                            # TELEMETRY DASHBOARD (TOP)
                            with ui.row().classes('absolute top-0 left-0 z-40 bg-slate-900/90 text-white p-4 shadow-2xl backdrop-blur-md border-b border-slate-700 w-full justify-around items-center'):
                                def telemetry_block(label, key, formatter=None, is_phase=False):
                                    with ui.column().classes('items-center gap-1'):
                                        ui.label(label).classes('text-[10px] text-slate-400 font-bold uppercase tracking-wider')
                                        if is_phase: 
                                            ui.label().bind_text_from(app_state['simulator'], key).classes('font-bold text-lg text-green-400')
                                        else: 
                                            ui.label().bind_text_from(app_state['simulator'], key, backward=formatter).classes('font-mono text-lg text-slate-100')

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

                            # TIMELINE DASHBOARD (BOTTOM)
                            with ui.column().classes('absolute bottom-0 left-0 z-40 bg-slate-900/90 p-4 shadow-2xl backdrop-blur-md border-t border-slate-700 w-full gap-0'):
                                with ui.row().classes('w-full flex flex-nowrap whitespace-nowrap text-[10px] text-slate-500 font-bold uppercase tracking-wide px-2 mb-[-12px]'):
                                    ui.label('CLIMB').classes('w-1/4 text-center border-l border-slate-600')
                                    ui.label('CRUISE').classes('w-2/4 text-center border-l border-r border-slate-600')
                                    ui.label('DESCEND').classes('w-1/4 text-center border-r border-slate-600')
                                
                                time_slider = ui.slider(
                                    min=0, max=1, step=0.005, value=0, 
                                    on_change=lambda e: update_state(e.value)
                                ).classes('w-full').props('color="blue-4" track-size="4px" thumb-size="16px"')
                        
                            # AIRCRAFT OVERLAY (CENTERED)
                            with ui.column().classes('absolute inset-0 z-40 items-center justify-center pointer-events-none'):
                                plane_icon = ui.icon('flight', size='128px').classes('text-blue-400 transition-all duration-300 drop-shadow-md')

                    with inner_split.after:
                        # --- 2. RIGHT PANEL (Mission Results) ---
                        with ui.column().classes(f'w-full h-full p-6 border-l overflow-y-auto {sidebar_bg} {sidebar_border}'):
                            ui.label('Mission Results').classes('text-xl font-bold mb-2')
                            ui.label('Select variables to view full simulation traces.').classes('text-sm mb-6')
                            
                            plot1_select = ui.select(plot_variables, value='Alt (ft)', label='Plot 1 Variable', on_change=lambda e: plot1.update_figure(create_plot(e.value))).classes('w-full mb-2')
                            plot1 = ui.plotly(create_plot('Alt (ft)')).classes('w-full mb-6 shadow-sm')
                            
                            plot2_select = ui.select(plot_variables, value='Mach', label='Plot 2 Variable', on_change=lambda e: plot2.update_figure(create_plot(e.value))).classes('w-full mb-2')
                            plot2 = ui.plotly(create_plot('Mach')).classes('w-full mb-6 shadow-sm')
                            
                            plot3_select = ui.select(plot_variables, value='Fuel Burn (lb/hr)', label='Plot 3 Variable', on_change=lambda e: plot3.update_figure(create_plot(e.value))).classes('w-full mb-2')
                            plot3 = ui.plotly(create_plot('Fuel Burn (lb/hr)')).classes('w-full shadow-sm')

        # --- Centralized State Manager ---
        def update_state(progress):
            step = int(progress * 200) - 1

            update_hud_to_step(step)
            
            orig_lat = app_state['simulator']['meta']['orig_lat']
            orig_lng = app_state['simulator']['meta']['orig_lng']

            dest_lat = app_state['simulator']['meta']['dest_lat']
            dest_lng = app_state['simulator']['meta']['dest_lng']

            if progress >= 1.0:
                lat1, lng1 = get_great_circle_point(orig_lat, orig_lng, dest_lat, dest_lng, 0.999)
                current_lat, current_lng = get_great_circle_point(orig_lat, orig_lng, dest_lat, dest_lng, 1.0)
                bearing = get_bearing(lat1, lng1, current_lat, current_lng)
            else:
                current_lat, current_lng = get_great_circle_point(orig_lat, orig_lng, dest_lat, dest_lng, progress)
                lat2, lng2 = get_great_circle_point(orig_lat, orig_lng, dest_lat, dest_lng, progress + 0.001)
                bearing = get_bearing(current_lat, current_lng, lat2, lng2)

            plane_icon.classes(remove='drop-shadow-[0_35px_35px_rgba(0,0,0,0.5)] drop-shadow-md')
            base_rot = bearing
            pitch_mod = 0 

            if app_state['simulator']['segment'] == 'CLIMB' or app_state['simulator']['segment'] == 'CRUISE':
                plane_icon.classes(add='drop-shadow-[0_35px_35px_rgba(0,0,0,0.5)]')
            else:
                plane_icon.classes(add='drop-shadow-md')
            
            norm_dest_lng = normalize_lng(dest_lng, orig_lng)
            flying_east = norm_dest_lng > orig_lng
            if app_state['simulator']['segment'] == 'CLIMB' or app_state['simulator']['segment'] == 'DESCEND':
                pitch_mod = 15 * (1 - 2 * flying_east)
                
            elif app_state['simulator']['segment'] == 'PRE-FLIGHT' or app_state['simulator']['segment'] == 'CRUISE':
                pitch_mod = 0 
            
            if flying_east:            
                plane_icon.style(f'transform: rotate({base_rot + pitch_mod}deg);')
            else:
                plane_icon.style(f'transform: rotate(-{360 - (base_rot + pitch_mod)}deg);')

            display_lng = normalize_lng(current_lng, orig_lng)
            alt_ratio = app_state['simulator']['current_alt'] / max(1, app_state['simulator']['altitude'])
            map_view.run_map_method('setView', [current_lat, display_lng], 7 - (3 * alt_ratio), {'animate': True})

        # --- Play/Pause Logic ---
        async def toggle_play():
            if app_state['simulator'].get('is_playing', False):
                app_state['simulator']['is_playing'] = False
                play_btn.set_text('Play Mission')
                play_btn.props('icon=play_arrow color=primary')
            else:
                if time_slider.value >= 1.0:
                    time_slider.value = 0.0 
                    await asyncio.sleep(0.2)
                    
                app_state['simulator']['is_playing'] = True
                play_btn.set_text('Pause Mission')
                play_btn.props('icon=pause color=primary')
                
                while app_state['simulator']['is_playing'] and time_slider.value < 1.0:
                    time_slider.value = min(1.0, time_slider.value + 0.005) 
                    await asyncio.sleep(0.05)
                    
                if time_slider.value >= 1.0:
                    app_state['simulator']['is_playing'] = False
                    play_btn.set_text('Play Mission')
                    play_btn.props('icon=play_arrow color=primary')