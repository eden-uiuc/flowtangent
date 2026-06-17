import json
from nicegui import ui
from utils.state import master_state, theme_config

def navigation_header():
    """Shared header with master state Save/Load capabilities."""
    
    dark = ui.dark_mode()

    if 'is_dark' not in master_state:
        master_state['is_dark'] = True

    def on_theme_toggle(e):
        # Update the master state boolean
        master_state['is_dark'] = e.value
        for callback in master_state.get('on_theme_changed', []):
            callback()
    
    # --- GLOBAL SAVE & LOAD LOGIC ---
    def save_project():
        # Dumps the entire master_state (both hangar and simulator)
        project_json = json.dumps(master_state, indent=4).encode('utf-8')
        ui.download(project_json, 'rcaide_master.rcaide')
        ui.notify('Master Project saved!', type='positive', position='top')

    async def load_project(e):
        try:
            content = await e.file.text()
            loaded_state = json.loads(content)
            
            # IN-PLACE UPDATE: This ensures any active UI bindings don't break
            if 'hangar' in loaded_state:
                master_state['hangar'].update(loaded_state['hangar'])
            if 'simulator' in loaded_state:
                master_state['simulator'].update(loaded_state['simulator'])
            
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
        
        # 1. LEFT: Logo and Title 
        with ui.row().classes('items-center gap-3 z-10'):
            ui.label('RCAIDE-EDEn STUDIO').classes('text-white text-xl font-bold tracking-widest')
        
        # 2. CENTER: Navigation Icons (Absolutely Centered)
        with ui.row().classes('absolute left-1/2 -translate-x-1/2 flex items-center justify-center gap-4 z-20'):
            with ui.button(icon='send', on_click=lambda: ui.navigate.to('/')) \
                    .props('flat round color=white size=md').classes('hover:bg-slate-800'):
                ui.tooltip('Hangar').classes('text-sm bg-slate-700')
                
            with ui.button(icon='flight_takeoff', on_click=lambda: ui.navigate.to('/simulator')) \
                    .props('flat round color=white size=md').classes('hover:bg-slate-800'):
                ui.tooltip('Simulator').classes('text-sm bg-slate-700')
        
        # 3. RIGHT: Hamburger Menu
        with ui.row().classes('items-center z-10'):
            with ui.button(icon='menu').props('flat round color=white').classes('hover:bg-slate-800'):
                
                # Using .style() applies the width immediately, preventing the Quasar geometry snap
                with ui.menu().style('min-width: 192px; width: 192px;') \
                              .classes('bg-slate-800 text-white border border-slate-700'):
                    
                    ui.menu_item('Save Project', on_click=save_project) \
                        .classes('hover:bg-slate-700 whitespace-nowrap')

                    ui.menu_item('Load Project', on_click=upload_dialog.open) \
                        .classes('hover:bg-slate-700 whitespace-nowrap')
                    
                    ui.separator()

                    with ui.row().classes('items-center w-full px-4 py-2'):
                        ui.switch('Dark Mode', value=master_state['is_dark'], on_change=on_theme_toggle) \
                            .bind_value_to(dark, 'value') \
                            .classes('w-full')

