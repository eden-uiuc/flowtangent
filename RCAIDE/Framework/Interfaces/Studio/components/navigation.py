import json
from nicegui import ui
# from utils.state import app_state

def navigation_header(app_state: dict, on_nav: callable):
    """Shared header with master state Save/Load capabilities."""
    
    dark = ui.dark_mode()

    if 'is_dark' not in app_state:
        app_state['is_dark'] = True

    def on_theme_toggle(e):
        # Update the master state boolean
        app_state['is_dark'] = e.value
        for callback in app_state.get('on_theme_changed', []):
            callback()
    
    # --- GLOBAL SAVE & LOAD LOGIC ---
    def save_project():
        # Create a shallow copy of the state so we don't delete the callbacks from the live app
        export_state = app_state.copy()
        
        # Remove the non-serializable function list before saving
        export_state.pop('on_theme_changed', None)
        
        project_json = json.dumps(export_state, indent=4).encode('utf-8')
        ui.download(project_json, 'rcaide_master.rcaide')
        ui.notify('Master Project saved!', type='positive', position='top')

    async def load_project(e):
        try:
            content = await e.file.text()
            loaded_state = json.loads(content)
            
            # IN-PLACE UPDATE: This ensures any active UI bindings don't break
            if 'hangar' in loaded_state:
                app_state['hangar'].update(loaded_state['hangar'])
            if 'simulator' in loaded_state:
                app_state['simulator'].update(loaded_state['simulator'])
            
            ui.notify('Project loaded! Refreshing...', type='positive', position='top')
            upload_dialog.close()
            
            # Force a hard reload of the current page to visually reflect the new state
            ui.navigate.to(ui.run_javascript('window.location.pathname'))
            
        except Exception as ex:
            ui.notify(f'Failed to load project: {str(ex)}', type='negative', position='top')

    # The hidden upload dialog (available on every page this header is injected into)
    with ui.dialog() as upload_dialog, ui.card().classes('w-96'):
        ui.label('Load Master Project').classes('text-lg font-bold mb-2')
        ui.upload(on_upload=load_project, auto_upload=True, max_files=1).classes('w-full')
        with ui.row().classes('w-full justify-end mt-4'):
            ui.button('Cancel', on_click=upload_dialog.close, color='gray')

    # --- HEADER UI ---
    with ui.header().classes('bg-slate-900 flex flex-row items-center justify-between px-6 py-2 shadow-md z-50'):
        
        # --- THE PROGRESS BAR ---
        global_progress = ui.linear_progress().classes('absolute bottom-0 left-0 w-full !m-0 !p-0 z-50') \
            .props('color=blue-500 size=2px indeterminate track-color=transparent')
        global_progress.set_visibility(False)

        # 1. LEFT: Logo and Title 
        with ui.row().classes('items-center gap-3 z-10'):
            ui.label('RCAIDE-EDEn STUDIO').classes('text-white text-xl font-bold tracking-widest')

        # 2. CENTER: Nav Buttons 
        # 2. Moved the absolute positioning OUTSIDE the refreshable to kill the ghost pixels!
        with ui.row().classes('absolute left-1/2 -translate-x-1/2 flex items-center justify-center gap-4 z-20 h-full'):
            
            @ui.refreshable
            def nav_buttons():
                nav_items = {
                    '/hangar': {'icon': 'send', 'label': 'Hangar'},
                    '/simulator': {'icon': 'flight_takeoff', 'label': 'Simulator'},
                }
                
                for route, info in nav_items.items():
                    is_active = (app_state['route'] == route)
                    state_classes = 'bg-white !text-slate-900' if is_active else 'bg-transparent !text-white hover:bg-slate-800'
                    
                    with ui.button(icon=info['icon'], on_click=lambda r=route: on_nav(r)) \
                            .props('flat round size=md') \
                            .classes(f'transition-colors duration-200 {state_classes}'):
                        ui.tooltip(info['label']).classes('text-sm bg-slate-700')
            
            # Call the refreshable buttons inside the stable wrapper
            nav_buttons()
        
        # 3. RIGHT: Hamburger Menu
        with ui.row().classes('items-center z-10'):
            with ui.button(icon='menu').props('flat round color=white').classes('hover:bg-slate-800'):
                with ui.menu().style('min-width: 192px; width: 192px;') \
                              .classes('bg-slate-800 text-white border border-slate-700'):
                    
                    ui.menu_item('Save Project', on_click=save_project).classes('hover:bg-slate-700 whitespace-nowrap')
                    ui.menu_item('Load Project', on_click=upload_dialog.open).classes('hover:bg-slate-700 whitespace-nowrap')
                    ui.separator()
                    with ui.row().classes('items-center w-full px-4 py-2'):
                        ui.switch('Dark Mode', value=app_state['is_dark'], on_change=on_theme_toggle) \
                            .bind_value_to(dark, 'value').classes('w-full')

    return nav_buttons.refresh, global_progress