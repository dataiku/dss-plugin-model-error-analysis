'use strict';
app.service("TreeUtils", function(Format) {
    const WRONG_PREDICTION = "Wrong prediction";
    const computeLocalError = function(d) {
        return (d.probabilities.find(_ => _[0] === WRONG_PREDICTION) || ["", 0, 0]).slice(1,3)
    };

    const addNode = function(svgParentElem, radius, getLocalErrorFunc, labelTextFunc, select=false) {
        svgParentElem.append("circle")
        .classed("node__background", true)
        .classed("legend-tree", select)
        .attr("cx", radius)
        .attr("cy", radius)
        .attr("r", radius);

        svgParentElem.append("path")
        .classed("node__gauge", true)
        .classed("legend-tree", select)
        .attr("d", function(d) {
            const localError = getLocalErrorFunc(d);
            const innerRadius = radius - 1;
            const theta = Math.PI*(-localError+.5);
            const start = {
                x: innerRadius*Math.cos(theta) + radius,
                y: innerRadius*Math.sin(theta) + radius
            };
            const end = {
                x: -innerRadius*Math.cos(theta) + radius,
                y: start.y
            };
            if (end.x === start.x) {
                end.x -= .0001;
            }
            const largeArcFlag = theta > 0 ? 0 : 1;
            return `M ${start.x} ${start.y} A ${innerRadius} ${innerRadius} 0 ${largeArcFlag} 1 ${end.x} ${end.y}`;
        });

        svgParentElem.append("text")
        .attr("text-anchor", "middle")
        .attr("x", radius)
        .attr("y", radius)
        .text(d => labelTextFunc(d));

    };

    const nodeValues = function(d) {
        if (d.values) {
            if (d.others) {
                return "Others"
            }
            return Format.ellipsis(d.values.join(", "), 20);
        }
        if (d.hasOwnProperty("beginning")) {
            return ">" + Format.noBreakingSpace + Format.toFixedIfNeeded(d.beginning, 5);
        }
        return "≤" + Format.noBreakingSpace + Format.toFixedIfNeeded(d.end, 5);
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
    return {
        addNode,
        computeLocalError,
        decisionRule,
        nodeValues,
        WRONG_PREDICTION
    }
});

app.service("TreeInteractions", function($timeout, $http, Format, TreeUtils) {
    let svg, tree, currentPath = new Set();
    const radius = 20, maxZoom = 3;

    const zoom = function() {
        svg.attr("transform",  "translate(" + d3.event.translate + ")scale(" + d3.event.scale + ")")
    };

    const zoomListener = d3.behavior.zoom()
        .on("zoom", function() {
            const svgArea = svg.node().getBoundingClientRect().width * svg.node().getBoundingClientRect().height;
            zoomListener.scaleExtent([svgArea < 100 ? zoomListener.scale() :  0, maxZoom])
            zoom();
    });

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
            node.selectAll(".decision-rule,.feature-children,.node__background,.node__gauge").classed("selected", true).classed("hovered", false);
            d3.select("#edge-" + node_id).classed("selected", true).classed("hovered", false);

            if (node_id == id) {
                node.select(".node__background").classed("node--selected", true);
            }

            if (node_id > 0) {
                scope.decisionRule.unshift(TreeUtils.decisionRule(node.node().__data__));
            }
            currentPath.add(node_id);
            node_id = node.node().__data__.parent_id;
        }
    }

    const showHovered = function(id, scope) {
        let node_id = id;
        while (node_id > -1) {
            let node = d3.select("#node-" + node_id);
            node.selectAll(".decision-rule,.feature-children,.node__gauge,.node__background").classed("hovered", true);
            d3.select("#edge-" + node_id).classed("hovered", true);
            node_id = scope.treeData[node_id].parent_id;
        }
    }

    const hideUnhovered = function() {
        d3.selectAll(".hovered").classed("hovered", false);
    }

    const zoomBack = function(selectedNode) {
        centerOnNode(selectedNode, true);
    }

    const zoomFit = function(onLoad) {
        const treePanel = d3.select(".tree").node().getBoundingClientRect(),
            svgDim = svg.node().getBBox();
        const leftOffset = 10;
        const scaleX = treePanel.width / (svgDim.width + leftOffset),
            scaleY = treePanel.height / (svgDim.height + 25)
        const scale = onLoad ? Math.min(Math.max(scaleX, scaleY), maxZoom) : Math.min(scaleX, scaleY, maxZoom);

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
        const scale = unzoom ? .8 : zoomListener.scale(),
            treePanel = d3.select(".tree").node().getBoundingClientRect();

        const x = treePanel.width / 2 - selectedNode.x * scale,
            y = treePanel.height / 2 - selectedNode.y * scale;

        svg.transition()
            .duration(400)
            .attr("transform", "translate(" + x + "," + (y - 20) + ")scale(" + scale + ")");
        zoomListener.translate([x, y]).scale(scale);
    }

    const select = function(id, scope, unzoom) {
        scope.seeGlobalChartData = false;
        if (scope.selectedNode) {
            hideUnselected();
            d3.select("#node--right-panel").remove();
        }
        hideUnhovered();
        showSelected(id, scope);
        scope.selectedNode = scope.treeData[id];
        const node = d3.select(".placeholder-node svg").append("g").attr("id", "node--right-panel");
        TreeUtils.addNode(node, 30, d=>scope.selectedNode.localError[0],  d=> Format.toFixedIfNeeded(scope.selectedNode.localError[0]*100, 2, true), true);

        scope.histData = {};
        //if (id == 0) {
        //    scope.histData = scope.histDataWholeSet;
        //} else {        
            loadHistograms(scope, id);
        //}
        
        centerOnNode(scope.selectedNode, unzoom);
    }

    const loadHistograms = function(scope, id) {
        $http.get(getWebAppBackendUrl("select-node/"+id))
            .then(function(response) {
                Object.assign(scope.histData, response.data);
            }, function(e) {
                scope.createModal.error(e.data);
            });
    }

    const addHatchMask = function(hatchSize = 5) {
        // Create hatched pattern
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
            .classed("tree-svg", true)
            .attr("width", "100%")
            .attr("height", "100%")
            .call(zoomListener);

        addHatchMask();

        svg = svg.append("g");

        update(scope);
        zoomFit(true);
    }

    const update = function(scope) {
        let source = scope.treeData[0];
        const nodeData = tree.nodes(source).reverse(),
          edgeData = tree.links(nodeData);
          nodeData.forEach(function(d) {
          d.y = d.depth * 180;
        });

        // Add new nodes
        const nodeEnter = svg.selectAll("g.node-container").data(nodeData, d => d.node_id)
        .enter().append("g")
        .classed("node-container", true)
        .attr("id", d => "node-" + d.node_id)
        .attr("transform", function(d) {
            return "translate(" + d.x + "," + d.y + ")";
        });

        const nodes = nodeEnter.append("g").classed("node", true)
        .on("click", function(d) {
            if (scope.selectedNode && scope.selectedNode.node_id == d.node_id) return;
            $timeout(select(d.node_id, scope, true));
        }).on("mouseenter", function(d) {
            if (currentPath.has(d.node_id)) return;
            showHovered(d.node_id, scope);
        }).on("mouseleave", function(d) {
            if (currentPath.has(d.node_id))return;
            hideUnhovered();
        });

        TreeUtils.addNode(nodes, radius, d=>d.localError[0], d=> Format.toFixedIfNeeded(d.localError[0]*100, 2, true));

        nodeEnter.filter(d => d.node_id > 0)
        .append("text")
        .attr("class", "decision-rule")
        .attr("text-anchor","middle")
        .attr("x", radius)
        .attr("y", - 10)
        .text(d => TreeUtils.nodeValues(d));

        nodeEnter.filter(d => d.children_ids.length)
        .append("text")
        .attr("class", "feature-children")
        .attr("text-anchor","middle")
        .attr("x", radius)
        .attr("y", radius*2 + 20)
        .text(d => Format.ellipsis(scope.treeData[d.children_ids[0].toString()].feature, 20));

        // Add new edges
        const edgeContainer = svg.selectAll(".edge").data(edgeData, d => d.target.node_id).enter().insert("g", "g");

        edgeContainer.append("path").attr("class", "edge")
        .attr("id", d => "edge-" + d.target.node_id)
        .classed("edge--no-error", d => !d.target.global_error)
        .attr("d", function(d) {
            return d3.svg.diagonal()({
                source: {x: d.source.x + radius, y: d.source.y + radius},
                target: {x: d.target.x + radius, y: d.target.y + radius}
            });
        })
        .attr("stroke-width", d => 1+radius*2*d.target.global_error);

        edgeContainer.append("text").append("textPath")
        .attr("href", d => "#edge-" + d.target.node_id)
        .attr("startOffset", "50%")
        .text(d => Format.toFixedIfNeeded(d.target.global_error*100, 2, true))

        edgeContainer.select("text").filter(d => d.target.x < d.source.x)
        .attr("transform", function(d) {
            const box = this.getBBox();
            const thickness = 1+radius*2*d.target.global_error;
            return `translate(${box.width-thickness} ${box.height-thickness}) rotate(180 ${box.x} ${box.y})`;
        });
    }

    return {
        createTree,
        zoomFit,
        zoomBack
    }
});
