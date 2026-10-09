import flowtangent as ft
from pathlib import Path

aero = ft.solve.VORJAX()
cyto_json = aero.to_cytoscape_json(recursive=True)

# Updated template with Dagre extension
html_template = """<!DOCTYPE html>
<html>
<head>
    <title>VORJAX Interactive Graph</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.26.0/cytoscape.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/dagre/0.8.5/dagre.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/cytoscape-dagre@2.5.0/cytoscape-dagre.min.js"></script>
    <style>
        /* Nord Polar Night Background (Dark Theme) */
        body { margin: 0; font-family: Inter, system-ui, sans-serif; background-color: #2e3440; overflow: hidden; }
        #cy { width: 100vw; height: 100vh; display: block; z-index: 1; }
    </style>
</head>
<body>
    <div id="cy"></div>
    <script>
        const graphElements = __JSON_PAYLOAD__;

        const layoutConfig = {
            name: 'dagre',
            rankDir: 'LR',
            nodeSep: 25,
            edgeSep: 15,
            rankSep: 50,
            animate: true,
            animationDuration: 300
        };

        const cy = cytoscape({
            container: document.getElementById('cy'),
            elements: graphElements,
            style: [
                /* 1. Process Nodes (Hexagons) */
                {
                    selector: 'node[node_type="process"]',
                    style: {
                        'shape': 'hexagon',
                        'background-color': '#81a1c1', /* Lighter Nord Frost Blue Interior */
                        'border-color': '#3b4252',     /* Darkest Nord Slate-Blue Border */
                        'border-width': 5,             /* Thick border */
                        'color': '#eceff4',            /* White Text */
                        'label': 'data(label)',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'text-wrap': 'wrap',
                        'text-max-width': '80px',
                        'padding': '36px',
                        'font-size': '12px',
                        'font-weight': 'bold',
                        'line-height': 1.2
                    }
                },
                /* 2. User Inputs - COLLAPSED STATE */
                {
                    selector: 'node[node_type="input"]',
                    style: {
                        'shape': 'ellipse',
                        'background-color': '#eceff4', /* White Interior */
                        'border-color': '#3b4252',     /* Darkest Nord Slate-Blue Border */
                        'border-width': 4,
                        'color': '#2e3440',            /* Dark Text */
                        'label': 'data(label)',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'padding': '24px',
                        'font-size': '12px',
                        'font-weight': 'bold'
                    }
                },
                /* 3. User Inputs - EXPANDED STATE */
                {
                    selector: 'node[node_type="input"].expanded',
                    style: {
                        'shape': 'round-rectangle',
                        'label': 'data(full_tree)',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'text-justification': 'left',
                        'text-wrap': 'wrap',
                        'width': 'label',
                        'height': 'label',
                        'padding': '24px',
                        'font-family': 'monospace',
                        'font-size': '12px',
                        'line-height': 1.25,
                        'font-weight': 'bold',
                        'background-color': '#eceff4',
                        'border-color': '#3b4252',
                        'border-width': 4
                    }
                },
                /* 4. Variable Nodes - COLLAPSED STATE */
                {
                    selector: 'node[node_type="variable"]',
                    style: {
                        'shape': 'ellipse',
                        'background-color': '#eceff4', /* White Interior */
                        'border-color': '#3b4252',     /* Darkest Nord Slate-Blue Border */
                        'border-width': 4,
                        'color': '#2e3440',            /* Dark text */
                        'label': 'data(short_label)',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'width': '28px',
                        'height': '28px',
                        'padding': '0px',
                        'font-size': '13px',
                        'font-weight': 'bold'
                    }
                },
                /* 5. Variable Nodes - EXPANDED STATE */
                {
                    selector: 'node[node_type="variable"].expanded',
                    style: {
                        'shape': 'round-rectangle',
                        'label': 'data(full_tree)',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'text-justification': 'left',
                        'text-wrap': 'wrap',
                        'width': 'label',
                        'height': 'label',
                        'padding': '24px',
                        'font-family': 'monospace',
                        'font-size': '12px',
                        'line-height': 1.25,
                        'font-weight': 'normal',
                        'background-color': '#eceff4',
                        'border-color': '#3b4252',
                        'border-width': 4
                    }
                },
                /* 6. Base Edges */
                {
                    selector: 'edge',
                    style: {
                        'width': 2,
                        'line-color': '#81a1c1',       /* Light blue edge to show against dark background */
                        'curve-style': 'bezier',
                    }
                },
                /* 7. Arrowheads */
                {
                    selector: 'edge[edge_type="outgoing"], edge[edge_type="direct"]',
                    style: {
                        'target-arrow-color': '#81a1c1',
                        'target-arrow-shape': 'triangle',
                    }
                }
            ],
            layout: layoutConfig
        });
        
        // 1. Tap Variable OR Input Node -> Toggle itself
        cy.on('tap', 'node[node_type="variable"], node[node_type="input"]', function(evt){
            evt.target.toggleClass('expanded');
            cy.layout(layoutConfig).run();
        });

        // 2. Tap Process Node -> Expand/Collapse ALL connected variables
        cy.on('tap', 'node[node_type="process"]', function(evt){
            const node = evt.target;
            const connectedVars = node.connectedEdges().connectedNodes('node[node_type="variable"]');
            
            // If any connected variable is collapsed, expand them all. Otherwise, collapse all.
            const anyCollapsed = connectedVars.some(n => !n.hasClass('expanded'));
            
            if (anyCollapsed) {
                connectedVars.addClass('expanded');
            } else {
                connectedVars.removeClass('expanded');
            }
            
            cy.layout(layoutConfig).run();
        });

        // 3. Double-Click Process Node -> MkDocs Navigation
        cy.on('dblclick', 'node[node_type="process"]', function(evt){
            const nodeName = evt.target.data('label');
            
            // Convert "Compute Aerodynamic Coefficients" to "compute-aerodynamic-coefficients"
            const urlSlug = nodeName.toLowerCase().replace(/[^a-z0-9]+/g, '-');
            
            // In the actual MkDocs site, this would execute:
            // window.location.href = `../api/processes/#flowtangent.solve.${urlSlug}`;
            
            console.log(`Navigating to docs for: ${nodeName}`);
            alert(`[MkDocs Routing Placeholder]\n\nSimulating redirect to:\n/api/processes/#${urlSlug}`);
        });
    </script>
</body>
</html>
"""
html_output = html_template.replace("__JSON_PAYLOAD__", cyto_json)

save_path = Path(__file__).resolve().parent / "VORJAX_cytoscape.html"
save_path.write_text(html_output, encoding="utf-8")
print(f"Graph saved to {save_path}. Open this file in your web browser.")