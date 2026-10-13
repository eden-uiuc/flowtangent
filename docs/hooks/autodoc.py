import importlib
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
