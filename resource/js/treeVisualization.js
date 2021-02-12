'use strict';
app.directive('tooltipTree', function() {
    return {
        scope: true,
        restrict: "C",
        templateUrl: "/plugins/model-error-analysis/resource/templates/tooltip.html",
        link: function(scope, element, attr) {
            const node = scope.treeData[attr.node];
            scope.probabilities = node.probabilities;
            scope.samples = node.samples;
            scope.globalError = node.global_error * 100;

            scope.inRightPanel = attr.hasOwnProperty("rightPanel");

            d3.select(element[0].children[0])
            .attr("x", scope.inRightPanel ? 0 : -30)
            .attr("y", scope.inRightPanel ? 0 : -25)
            .attr("height", scope.inRightPanel ? "100%" : 100)
            .attr("width", scope.inRightPanel ? "100%" : 260);

            // Compute the position of each group on the pie
            const pie = d3.layout.pie()
                .value(function(d) {return d[1];});
            const proba = pie(scope.probabilities);

            // Build the pie chart
            d3.select(scope.inRightPanel ? "#tooltip-right-panel" : "#tooltip-" + node.node_id)
            .append("g")
            .attr("transform", scope.inRightPanel ? "translate(50, 50)" : "translate(10, 20)")
            .selectAll('.camembert')
            .data(proba)
            .enter()
            .append('path')
            .attr('d', d3.svg.arc()
                .innerRadius(0)
                .outerRadius(scope.inRightPanel ? 40 : 30)
            )
            .attr('fill', function(d) {
                return scope.colors[d.data[0]];
            });
        }
    };
});

app.service("TreeInteractions", function($timeout, $http, $compile, Format) {
    let svg, tree, currentPath = new Set();
    const side = 40, maxZoom = 3;

    const zoom = function() {
        svg.attr("transform",  "translate(" + d3.event.translate + ")scale(" + d3.event.scale + ")")
    };

    const zoomListener = d3.behavior.zoom()
        .on("zoom", function() {
            const svgArea = svg.node().getBoundingClientRect().width * svg.node().getBoundingClientRect().height;
            zoomListener.scaleExtent([svgArea < 100 ? zoomListener.scale() :  0, maxZoom])
            zoom();
    });

    const nodeValues = function(d) {
        if (d.values) {
            if (d.others) {
                return "Others"
            }
            return Format.ellipsis(d.values.join(", "), 20);
        }
        return ((d.hasOwnProperty("beginning") ? ("]" + Format.ellipsis(d.beginning, 8)) : "]-∞") + " ; "
                + (d.hasOwnProperty("end") ? (Format.ellipsis(d.end, 8) + "]") : "+∞["));
    }

    const decisionRule = function(node) {
        if (node.values) {
            let middle = " is ";
            if (node.others) {
                middle += "not ";
            }
            if (node.values.length > 1) {
                middle += "one of ";
            }
            return {
                left: node.feature,
                middle,
                right: node.values.join(", ")
            }
        }
        return {
            left: (node.hasOwnProperty("beginning") ? (Format.toFixedIfNeeded(node.beginning, 5) + Format.noBreakingSpace + "<") : ""),
            middle: node.feature,
            right: (node.hasOwnProperty("end") ? ("≤" + Format.noBreakingSpace + Format.toFixedIfNeeded(node.end, 5)) : ""),
            numerical: true
        }
    }

    const shift = function(id, scope, classLink, unspread) {
        let nodes = Array.from(scope.treeData[0].children_ids);
        const shiftRight = new Set();
        while (nodes.length) {
            const node = scope.treeData[nodes.shift()];
            if (node.parent_id != id) {
                nodes = nodes.concat(node.children_ids);
            }

            const linkParentToNode = d3.select("#edge-" + node.node_id);
            if (linkParentToNode.classed(classLink)) {
                shiftRight.add(node.parent_id);
            } else {
                if (node.parent_id != id) {
                    let delta;
                    if (shiftRight.has(node.parent_id)) {
                        delta = 120 + 40*node.depth;
                        shiftRight.add(node.node_id);
                    } else {
                        delta = -80 - 30*node.depth;
                    }
                    node.x = node.x + (unspread ? -delta : delta);
                    d3.select("#node-" + node.node_id).attr("transform", function(d) {
                        return "translate(" + d.x + "," + d.y + ")";
                    });
                }
                linkParentToNode.attr("d", function(d) {
                    return d3.svg.diagonal()({source: {x: d.source.x + side/2, y: d.source.y},
                        target: {x: d.target.x + side/2, y: d.target.y}});
                });
            }
        }
    }

    const hideUnselected = function() {
        d3.selectAll(".selected").classed("selected", false);
        d3.select(".node--selected").classed("node--selected", false);
    }

    const showSelected = function(id, scope) {
        let node_id = id;
        currentPath.clear();
        scope.decisionRule = [];
        while (node_id > -1) {
            let node = d3.select("#node-" + node_id);
            node.selectAll(".decision-rule,.feature-children,.node-background").classed("selected", true).classed("hovered", false);
            d3.select("#edge-" + node_id).classed("selected", true).classed("hovered", false);

            if (node_id == id) {
                node.select("rect").classed("node--selected", true);
            } else {
                node.select("#tooltip-"+node_id).classed("selected", true);
            }

            if (node_id > 0) {
                scope.decisionRule.unshift(decisionRule(node.node().__data__));
            }
            currentPath.add(node_id);
            node_id = node.node().__data__.parent_id;
        }
    }

    const showHovered = function(id, scope) {
        let node_id = id;
        while (node_id > -1) {
            let node = d3.select("#node-" + node_id);
            node.selectAll(".decision-rule,.feature-children").classed("hovered", true);
            d3.select("#edge-" + node_id).classed("hovered", true);
            node.select("#tooltip-"+node_id).classed("hovered", true);
            node_id = scope.treeData[node_id].parent_id;
        }
    }

    const hideUnhovered = function() {
        d3.selectAll(".hovered").classed("hovered", false);
    }

    const zoomBack = function(selectedNode) {
        centerOnNode(selectedNode, true);
    }

    const zoomFit = function() {
        const treePanel = d3.select(".tree").node().getBoundingClientRect(),
            svgDim = svg.node().getBBox();
        const leftOffset = 10;
        const scaleX = treePanel.width / (svgDim.width + leftOffset),
            scaleY = treePanel.height / (svgDim.height + 25)
        const scale = Math.min(scaleX, scaleY, maxZoom);

        let leftTranslate;
        if (scale == maxZoom) {
            leftTranslate = treePanel.width / 2;
        } else {
            leftTranslate = (Math.abs(svgDim.x) + leftOffset)*scale;
        }

        const topTranslate = 40 * scale + 20;
        zoomListener.translate([leftTranslate, topTranslate]).scale(scale);
        svg.transition().duration(400).attr("transform", "translate(" + leftTranslate + "," + topTranslate +")scale(" + scale + ")");
    }

    const centerOnNode = function(selectedNode, unzoom) {
        const scale = unzoom ? 1 : zoomListener.scale(),
            treePanel = d3.select(".tree").node().getBoundingClientRect();

        const x = treePanel.width / 2 - selectedNode.x * scale,
            y = treePanel.height / 2 - selectedNode.y * scale;

        svg.transition()
            .duration(400)
            .attr("transform", "translate(" + x + "," + (y - 20) + ")scale(" + scale + ")");
        zoomListener.translate([x, y]).scale(scale);
    }

    const select = function(id, scope, unzoom, noRecenter) {
        if(scope.selectedNode) {
            delete scope.selectedNode.editLabel;
        }
        if (scope.selectedNode) {
            update(scope);
            hideUnselected();
        }
        hideUnhovered();
        showSelected(id, scope);
        shift(id, scope, "selected");
        scope.selectedNode = scope.treeData[id];
        scope.histData = {};
        scope.loadingHistogram = true;
        loadHistograms(scope, id);

        if (!noRecenter) {
            centerOnNode(scope.selectedNode, unzoom);
        }
    }

    const loadHistograms = function(scope, id) {
        $http.get(getWebAppBackendUrl("select-node/"+id))
            .then(function(response) {
                if (id == 0) {
                    scope.histDataWholeSet = response.data;
                } else {
                    scope.histData = response.data;
                }
                scope.loadingHistogram = false;
            }, function(e) {
                scope.loadingHistogram = false;
                scope.createModal.error(e.data);
            });
    }

    const addVizTooltips = function(scope) {
        d3.selectAll(".node-container").append("g")
        .attr("transform", "translate(100, -10)")
        .classed("tooltip", true)
        .classed("tooltip-tree", true)
        .attr("id", d => "tooltip-" + d.node_id)
        .attr("node", d => d.node_id)
        .call(function() {
            $compile(this[0])(scope);
        })
        .on("wheel", function() {
            d3.event.stopPropagation();
        });
    };

    const addHatchMask = function(hatchSize = 5) {
        //Create hatched pattern
        const defs = svg.append("defs");
        defs.append("pattern")
            .attr("id", "hatch-pattern")
            .attr("width", hatchSize)
            .attr("height", hatchSize)
            .attr("patternTransform", "rotate(45)")
            .attr("patternUnits","userSpaceOnUse")
            .append("rect")
            .attr("width", hatchSize/2)
            .attr("height", hatchSize)
            .attr("fill", "white");

        defs.append("mask")
            .attr("id", "hatch-mask")
            .append("rect")
            .attr("width", "100%")
            .attr("height", "100%")
            .attr("x", 0)
            .attr("y", 0)
            .attr("fill", "url(#hatch-pattern");
    }

    const createTree = function(scope) {
        tree = d3.layout.tree()
            .nodeSize([140, 65])
            .children(function(d) {
                return d.children_ids.map(_ => scope.treeData[_]);
            });

        svg = d3.select(".tree").append("svg")
            .attr("width", "100%")
            .attr("height", "100%")
            .call(zoomListener);

        addHatchMask();

        svg = svg.append("g");

        update(scope);
        loadHistograms(scope, 0);
        zoomFit();
        addVizTooltips(scope);
    }

    const update = function(scope) {
        let source = scope.treeData[0];
        const nodes = tree.nodes(source).reverse(),
          links = tree.links(nodes);
        nodes.forEach(function(d) {
          d.y = d.depth * 180;
        });

        const node = svg.selectAll("g.node-container")
        .data(nodes, d => d.node_id);

        // update pre-existing nodes
        node.attr("transform", function(d) {
            return "translate(" + d.x + "," + d.y + ")";
        });

        // add new nodes
        const nodeEnter = node.enter().append("g")
        .classed("node-container", true)
        .attr("id", d => "node-" + d.node_id)
        .attr("transform", function(d) {
            return "translate(" + d.x + "," + d.y + ")";
        });

        nodeEnter.append("rect")
        .classed("node-background", true)
        .attr("height", side)
        .attr("width", side);

        nodeEnter.append("rect")
        .classed("node-content--error", true)
        .attr("x", .5)
        .attr("y", d => .5 + (side - 1) * (1 - d.global_error))
        .attr("height", d => (side-1)*d.global_error)
        .attr("width", side-1);

        nodeEnter.append("text")
        .attr("class", "global-error")
        .attr("text-anchor","middle")
        .attr("x", side / 2)
        .attr("y", side / 2)
        .text(d => Format.toFixedIfNeeded(d.global_error*100, 2, true));

        nodeEnter.on("click", function(d) {
            if (scope.selectedNode && scope.selectedNode.node_id == d.node_id) return;
            $timeout(select(d.node_id, scope));
        }).on("mouseenter", function(d) {
            if (currentPath.has(d.node_id)) return;
            showHovered(d.node_id, scope);
            shift(d.node_id, scope, "hovered");
        }).on("mouseleave", function(d) {
            if (currentPath.has(d.node_id))return;
            shift(d.node_id, scope, "hovered", true);
            hideUnhovered();
        });

        nodeEnter.filter(d => d.node_id > 0)
        .append("text")
        .attr("class", "decision-rule")
        .attr("text-anchor","middle")
        .attr("x", side / 2)
        .attr("y", - 10)
        .text(d => nodeValues(d));

        nodeEnter.filter(d => d.children_ids.length)
        .append("text")
        .attr("class", "feature-children")
        .attr("text-anchor","middle")
        .attr("x", side / 2)
        .attr("y", side + 20)
        .text(d => Format.ellipsis(scope.treeData[d.children_ids[0].toString()].feature, 20));

        const edge = svg.selectAll(".edge")
        .data(links, d => d.target.node_id);

        // update pre-existing links
        edge.attr("d", function(d) {
            return d3.svg.diagonal()({source: {x: d.source.x + side/2, y: d.source.y},
                                    target: {x: d.target.x + side/2, y: d.target.y}});
        })

        // add new links
        edge.enter().insert("path", "g")
        .attr("class", "edge")
        .attr("id", d => "edge-" + d.target.node_id)
        .attr("d", function(d) {
            return d3.svg.diagonal()({source: {x: d.source.x + side/2, y: d.source.y},
                                  target: {x: d.target.x + side/2, y: d.target.y}});
        })
        .attr("stroke-width", d => 1+d.target.samples[1] / 5)
    }

    return {
        createTree,
        decisionRule, // TODO: remove
        zoomFit,
        zoomBack,
        select // TODO: remove
    }
});
