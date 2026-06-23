import asyncio
from nicegui import ui

from pages.hangar import hangar_ui
from pages.simulator import simulator_ui
from components.navigation import navigation_header
from utils.state import app_state

@ui.refreshable
def router_content():
    """Dynamically renders the active view and automatically destroys the old DOM."""
    if app_state['route'] == '/hangar':
        hangar_ui()
    elif app_state['route'] == '/simulator':
        simulator_ui()
    else:
        ui.label('404 - Tool Not Found').classes('text-red-500 text-xl p-8')

@ui.page('/')
def main_app():
    # Remove default padding to let the header go edge-to-edge
    ui.context.client.content.classes('p-0 gap-0')

    header_refresh = None
    global_progress = None
    
    async def switch_page(new_route):
        if app_state['route'] != new_route:
            
            # 1. Start Transition (Show progress, drop opacity to 40%, disable clicking)
            if global_progress:
                global_progress.set_visibility(True)
            
            content_wrapper.classes(add='opacity-40 pointer-events-none')
            
            # Yield to the browser for 200ms so it can actually draw the fade and progress bar
            await asyncio.sleep(0.2) 
            
            # 2. Swap the Route and Rebuild the UI (This happens instantly while hidden)
            app_state['route'] = new_route
            router_content.refresh()
            if header_refresh:
                header_refresh()
            
            # Yield briefly again to ensure the DOM is fully hydrated before lifting the curtain
            await asyncio.sleep(0.1)
            
            # 3. End Transition (Hide progress, restore full opacity, re-enable clicking)
            content_wrapper.classes(remove='opacity-40 pointer-events-none')
            if global_progress:
                global_progress.set_visibility(False)

    callbacks = app_state.setdefault('on_theme_changed', [])
    if router_content.refresh not in callbacks:
        callbacks.append(router_content.refresh)

    # Capture both returns from the header
    header_refresh, global_progress = navigation_header(app_state=app_state, on_nav=switch_page)

    # Use a neutral background here so the fade-out doesn't reveal a blinding white flash underneath
    is_dark = app_state.get('is_dark', False)
    bg_color = 'bg-neutral-900' if is_dark else 'bg-gray-200'

    with ui.column().classes(f'w-full h-[calc(100vh-64px)] overflow-hidden p-0 gap-0 {bg_color}'):
        
        # --- THE TRANSITION WRAPPER ---
        # Tailwind handles the smooth fading automatically via 'transition-opacity duration-300'
        content_wrapper = ui.element('div').classes('w-full h-full transition-opacity duration-300 ease-in-out')
        
        with content_wrapper:
            router_content()

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="RCAIDE Studio", port=8080)