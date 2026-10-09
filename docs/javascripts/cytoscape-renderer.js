document.addEventListener("DOMContentLoaded", function() {
    const containers = document.querySelectorAll('.cyto-graph-container');
    
    containers.forEach(container => {
        // Extract the JSON data from the hidden script tag
        const dataScript = container.querySelector('.cyto-data');
        if (!dataScript) return;
        
        const elements = JSON.parse(dataScript.textContent);
        
        // Initialize Cytoscape
        const cy = cytoscape({
            container: container,
            elements: elements,
            style: [
                {
                    selector: 'node',
                    style: {
                        'background-color': '#3b82f6',
                        'label': 'data(label)',
                        'color': '#0f172a',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'shape': 'round-rectangle',
                        'width': 'label',
                        'height': 'label',
                        'padding': '10px'
                    }
                },
                {
                    selector: 'node[type="input"]',
                    style: {
                        'shape': 'ellipse',
                        'background-color': '#10b981'
                    }
                },
                {
                    selector: 'edge',
                    style: {
                        'width': 2,
                        'line-color': '#94a3b8',
                        'target-arrow-color': '#94a3b8',
                        'target-arrow-shape': 'triangle',
                        'curve-style': 'bezier',
                        'label': 'data(label)',
                        'font-size': '10px',
                        'text-wrap': 'wrap', // Native support for \n in labels!
                        'text-background-color': '#ffffff',
                        'text-background-opacity': 0.8,
                        'text-background-padding': '2px'
                    }
                }
            ],
            layout: {
                name: 'breadthfirst',
                directed: true,
                padding: 10,
                spacingFactor: 1.5
            }
        });
        
        // Interactive feature: Click an edge to highlight it
        cy.on('tap', 'edge', function(evt){
            const edge = evt.target;
            cy.edges().style('line-color', '#94a3b8');
            edge.style('line-color', '#f59e0b');
        });
    });
});