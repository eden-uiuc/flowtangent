document$.subscribe(function() {
    const wrappers = document.querySelectorAll('.cyto-wrapper');
    
    wrappers.forEach(wrapper => {
        const canvas = wrapper.querySelector('.cyto-canvas');
        const dataScript = wrapper.querySelector('.cyto-data');
        
        if (!canvas || !dataScript || canvas.hasAttribute('data-cy-initialized')) return;
        
        const graphElements = JSON.parse(dataScript.textContent);
        canvas.setAttribute('data-cy-initialized', 'true');

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
            container: canvas,
            elements: graphElements,
            style: [
                {
                    selector: 'node[node_type="process"]',
                    style: {
                        'shape': 'hexagon',
                        'background-color': '#81a1c1',
                        'border-color': '#3b4252',
                        'border-width': 5,
                        'color': '#eceff4',
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
                {
                    selector: 'node[node_type="input"]',
                    style: {
                        'shape': 'ellipse',
                        'background-color': '#eceff4',
                        'border-color': '#3b4252',
                        'border-width': 4,
                        'color': '#2e3440',
                        'label': 'data(label)',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'padding': '24px',
                        'font-size': '12px',
                        'font-weight': 'bold'
                    }
                },
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
                {
                    selector: 'node[node_type="variable"]',
                    style: {
                        'shape': 'ellipse',
                        'background-color': '#eceff4',
                        'border-color': '#3b4252',
                        'border-width': 4,
                        'color': '#2e3440',
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
                {
                    selector: 'edge',
                    style: {
                        'width': 2,
                        'line-color': '#81a1c1',
                        'curve-style': 'bezier',
                    }
                },
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
        
        // INTERACTION: Toggle Variable/Input node on tap
        cy.on('tap', 'node[node_type="variable"], node[node_type="input"]', function(evt){
            evt.target.toggleClass('expanded');
            cy.layout(layoutConfig).run();
        });

        // INTERACTION: Tap Process -> Toggle all connected variables
        cy.on('tap', 'node[node_type="process"]', function(evt){
            const node = evt.target;
            const connectedVars = node.connectedEdges().connectedNodes('node[node_type="variable"]');
            
            const anyCollapsed = connectedVars.some(n => !n.hasClass('expanded'));
            if (anyCollapsed) {
                connectedVars.addClass('expanded');
            } else {
                connectedVars.removeClass('expanded');
            }
            cy.layout(layoutConfig).run();
        });

        // INTERACTION: Double-click Process -> Route to specific Docs API page
        cy.on('dblclick', 'node[node_type="process"]', function(evt){
            const nodeName = evt.target.data('label');
            // Assuming your MkDocs creates slugs like 'compute-aerodynamic-coefficients'
            const urlSlug = nodeName.toLowerCase().replace(/[^a-z0-9]+/g, '-');
            
            // NOTE: Adjust this base path depending on your MkDocs structure!
            const basePath = window.location.origin + "/api/processes/"; 
            window.location.href = basePath + "#flowtangent.solve." + urlSlug;
        });
    });
});