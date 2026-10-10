import inspect
import flowtangent as ft

def on_pre_build(config):
    """Injects Cytoscape JSON into Process docstrings before HTML generation."""
    process_classes = [
        cls for name, cls in inspect.getmembers(ft, inspect.isclass)
        if issubclass(cls, ft.Process) and cls is not ft.Process
    ]

    for cls in process_classes:
        doc = cls.__doc__ or ""
        
        # Only inject if not already documented and the method exists
        if "### Process Architecture" not in doc and hasattr(cls, "to_cytoscape_json"):
            try:
                # Instantiate a dummy process to extract the DAG
                dummy = cls()
                cyto_json = dummy.to_cytoscape_json(recursive=True)
                
                diagram_section = (
                    f"\n\n### Process Architecture\n\n"
                    f"<div class='cyto-wrapper' style='height: 500px; width: 100%; border: 2px solid #3b4252; border-radius: 8px; overflow: hidden; position: relative; margin-top: 1em;'>\n"
                    f"  <div class='cyto-canvas' style='width: 100%; height: 100%; background-color: #2e3440;'></div>\n"
                    f"  <script type='application/json' class='cyto-data'>\n{cyto_json}\n  </script>\n"
                    f"</div>\n"
                )
                
                cls.__doc__ = doc + diagram_section
            except Exception:
                pass