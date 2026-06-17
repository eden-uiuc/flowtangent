from nicegui import ui

# Importing these modules automatically registers their @ui.page routes
import pages.simulator
import pages.hangar

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="RCAIDE Studio", port=8080)