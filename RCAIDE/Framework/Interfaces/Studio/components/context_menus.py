from nicegui import ui
from utils.state import app_state, perform_undo, perform_redo

def setup_context_menu():
    is_dark = app_state.get('is_dark', False)
    menu_bg = 'bg-neutral-800 text-gray-200 border border-neutral-700' if is_dark else 'bg-white text-gray-800 border border-gray-200'
    hover_color = 'hover:bg-neutral-700' if is_dark else 'hover:bg-gray-100'

    # Quasar q-menu with context-menu flag overrides the native browser right-click menu
    with ui.element('q-menu').props('context-menu').classes(f'p-1 min-w-[160px] shadow-xl rounded-md {menu_bg}') as menu:
        with ui.column().classes('w-full gap-0'):
            
            # --- UNDO ITEM ---
            has_undo = len(app_state.get('history_stack', [])) > 0
            with ui.row().classes(f'w-full items-center justify-between px-3 py-2 text-sm rounded cursor-pointer transition-colors {hover_color if has_undo else "opacity-40 pointer-events-none"}') \
                    .on('click', lambda: [perform_undo(), menu.run_method('hide')]):
                with ui.row().classes('items-center gap-2'):
                    ui.icon('undo', size='xs')
                    ui.label('Undo')
                ui.label('Ctrl+Z').classes('text-xs opacity-40 font-mono')
                
            # --- REDO ITEM ---
            has_redo = len(app_state.get('redo_stack', [])) > 0
            with ui.row().classes(f'w-full items-center justify-between px-3 py-2 text-sm rounded cursor-pointer transition-colors {hover_color if has_redo else "opacity-40 pointer-events-none"}') \
                    .on('click', lambda: [perform_redo(), menu.run_method('hide')]):
                with ui.row().classes('items-center gap-2'):
                    ui.icon('redo', size='xs')
                    ui.label('Redo')
                ui.label('Ctrl+Y').classes('text-xs opacity-40 font-mono')