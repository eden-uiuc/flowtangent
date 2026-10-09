import inspect
import flowtangent as ft

def on_pre_build(config):
    process_classes = [
        cls for name, cls in inspect.getmembers(ft, inspect.isclass)
        if issubclass(cls, ft.Process) and cls is not ft.Process
    ]

    for cls in process_classes:
        doc = cls.__doc__ or ""
        
        if "### Process Architecture" not in doc and hasattr(cls, "to_cytoscape_json"):
            try:
                dummy_instance = cls()
                cyto_json = dummy_instance.to_cytoscape_json()
                
                # Wrap the JSON in a hidden div that our JS will target
                diagram_section = (
                    f"\n\n### Process Architecture\n\n"
                    f"<div class='cyto-graph-container' style='height: 400px; width: 100%; border: 1px solid #ddd; border-radius: 8px;'>\n"
                    f"<script type='application/json' class='cyto-data'>\n{cyto_json}\n</script>\n"
                    f"</div>\n"
                )
                
                cls.__doc__ = doc + diagram_section
            except Exception:
                pass