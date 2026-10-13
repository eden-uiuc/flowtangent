import importlib
import json
import logging

from griffe import Docstring, Extension

import flowtangent as ft


class ProcessDiagramExtension(Extension):
    def on_class(self, *, cls, **kwargs):
        # 1. Skip anything that isn't part of FlowTangent
        if not cls.path.startswith("flowtangent."):
            return

        try:
            # 2. Dynamically import the runtime class since Griffe is static
            module_name = cls.module.path
            module = importlib.import_module(module_name)
            runtime_cls = getattr(module, cls.name, None)

            # 3. Check if it is a valid Process class
            if (
                runtime_cls
                and isinstance(runtime_cls, type)
                and issubclass(runtime_cls, ft.Process)
                and runtime_cls is not ft.Process
            ):
                if hasattr(runtime_cls, "to_cytoscape_json"):
                    # Instantiate the dummy and extract the graph
                    dummy = runtime_cls()
                    cyto_json = dummy.to_cytoscape_json(recursive=True)

                    # Parse the JSON string to verify it isn't empty
                    cyto_data = json.loads(cyto_json) if isinstance(cyto_json, str) else cyto_json
                    has_elements = False

                    if isinstance(cyto_data, list) and len(cyto_data) > 0:
                        has_elements = True
                    elif isinstance(cyto_data, dict):
                        # Handle both {"elements": [...]} and {"nodes": [...]} formats
                        if cyto_data.get("elements") or cyto_data.get("nodes"):
                            has_elements = True

                    # Only inject the diagram if we confirmed there are nodes to render
                    if has_elements:
                        diagram_section = (
                            f"\n\n### Process Architecture\n\n"
                            f"<div class='cyto-wrapper' style='height: 500px; width: 100%; border: 2px solid #3b4252; border-radius: 8px; overflow: hidden; position: relative; margin-top: 1em;'>\n"  # noqa: E501
                            f"  <div class='cyto-canvas' style='width: 100%; height: 100%; background-color: #2e3440;'></div>\n"  # noqa: E501
                            f"  <script type='application/json' class='cyto-data'>\n{cyto_json}\n  </script>\n"
                            f"</div>\n"
                        )

                        # 4. Inject directly into Griffe's documentation model
                        if cls.docstring:
                            cls.docstring.value += diagram_section
                        else:
                            cls.docstring = Docstring(diagram_section, parent=cls)

        except Exception as exc:
            logging.getLogger("mkdocs").warning(f"autodoc: no graph for {cls.path}: {exc}")


class CleanSignaturesExtension(Extension):
    """Removes private parameters (starting with _) from all documented signatures."""

    def _clean_parameters(self, obj):
        # Safely check if the object has a parameters collection
        if obj and hasattr(obj, "parameters") and obj.parameters:
            # Find all private arguments
            to_remove = [p.name for p in obj.parameters if p.name.startswith("_")]

            # Delete them directly from the Griffe Parameters object
            for name in to_remove:
                try:
                    del obj.parameters[name]
                except (KeyError, TypeError, AttributeError):
                    pass

    def on_function(self, **kwargs):
        func = kwargs.get("obj") or kwargs.get("function")
        self._clean_parameters(func)

    def on_class(self, **kwargs):
        cls = kwargs.get("obj") or kwargs.get("cls")
        self._clean_parameters(cls)

        # Also clean the underlying __init__ method if it's explicitly defined
        if cls and "__init__" in cls.members:
            self._clean_parameters(cls.members["__init__"])
