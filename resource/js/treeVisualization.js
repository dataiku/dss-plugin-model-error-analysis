'use strict';
app.service("TreeInteractions", function($timeout, $http, $compile, Format) {
    let svg, tree, currentPath = new Set();
    const side = 30, maxZoom = 3;

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

    const decisionRule = function(node, ellipsis) {
        if (node.values) {
            let middle = " is ";
            if (node.others) {
                middle += "not ";
            }
            if (node.values.length > 1) {
                middle += "one of ";
            }
            if (ellipsis) {
                return Format.ellipsis(node.feature, 20) + middle + Format.ellipsis(node.values, 20);
            }
            return node.feature + middle + node.values;
        }
        if (ellipsis) {
            return (node.hasOwnProperty("beginning") ? (Format.ellipsis(node.beginning, 10) + " < ") : "")
                + Format.ellipsis(node.feature, 20)
                + (node.hasOwnProperty("end") ? (" ≤ " + Format.ellipsis(node.end, 10)) : "");
        }
        return (node.hasOwnProperty("beginning") ? (Format.toFixedIfNeeded(node.beginning, 3) + " < ") : "")
        + node.feature
        + (node.hasOwnProperty("end") ? (" ≤ " + Format.toFixedIfNeeded(node.end, 3)) : "");
    }

    const shift = function(id, scope, classLink, unspread) {
        let nodes = Array.from(scope.treeData[0].children_ids);
        const shiftRight = new Set();
        while (nodes.length) {
            const node = scope.treeData[nodes.shift()];
            if (node.parent_id != id) {
                nodes = nodes.concat(node.children_ids);
            }

            const linkParentToNode = d3.select("#link-" + node.id);
            if (linkParentToNode.classed(classLink)) {
                shiftRight.add(node.parent_id);
            } else {
                if (node.parent_id != id) {
                    let delta;
                    if (shiftRight.has(node.parent_id)) {
                        delta = 120 + 40*node.depth;
                        shiftRight.add(node.id);
                    } else {
                        delta = -80 - 30*node.depth;
                    }
                    node.x = node.x + (unspread ? -delta : delta);
                    d3.select("#node-" + node.id).attr("transform", function(d) {
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

    const hideUnselected = function(id) {
        d3.selectAll("[tooltip='tree']").classed("selected", false);
        d3.selectAll(".selected").classed("selected", false);
        d3.select("#node-" + id).select("rect").style("stroke", null).style("stroke-width", null);
    }

    const showSelected = function(id, scope) {
        let node_id = id;
        currentPath.clear();
        scope.decisionRule = [];
        while (node_id > -1) {
            let node = d3.select("#node-" + node_id);
            node.selectAll(".decision-rule,.feature-children").classed("selected", true).classed("hovered", false);
            d3.select("#link-" + node_id).classed("selected", true).classed("hovered", false);

            if (node_id == id) {
                node.select("rect").style("stroke", "#007eff")
                    .style("stroke-width", "1px");
            } else {
                node.select("#tooltip-"+node_id).classed("selected", true);
            }

            if (node_id > 0) {
                scope.decisionRule.unshift({"full": decisionRule(node.node().__data__), "ellipsed": decisionRule(node.node().__data__, true)});
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
            d3.select("#link-" + node_id).classed("hovered", true);
            if (node_id != id) {
                node.select("#tooltip-"+node_id).classed("hovered", true);
            }
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
            scaleY = treePanel.height / (svgDim.height + 5)
        const scale = Math.min(scaleX, scaleY, maxZoom);

        let leftTranslate;
        if (scale == maxZoom) {
            leftTranslate = treePanel.width / 2;
        } else {
            leftTranslate = (Math.abs(svgDim.x) + leftOffset)*scale;
        }

        const topTranslate = 40 * scale;
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
            hideUnselected(scope.selectedNode.id);
        }
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
        .attr("tooltip", "tree")
        .attr("id", d => "tooltip-" + d.id)
        .attr("node", d => d.id)
        .call(function() {
            $compile(this[0])(scope);
        })
        .on("wheel", function() {
            d3.event.stopPropagation();
        });
    };

    const createTree = function(scope) {
        tree = d3.layout.tree()
                .nodeSize([140, 65])
                .children(function(d) {
                   return d.children_ids.map(_ => scope.treeData[_]);
               });

        svg = d3.select(".tree").append("svg")
            .attr("width", "100%")
            .attr("height", "100%")
            .call(zoomListener).on("dblclick.zoom", null)
            .append("g");

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
        .data(nodes, d => d.id);

        // update pre-existing nodes
        node.attr("transform", function(d) {
        return "translate(" + d.x + "," + d.y + ")";
        });

        node.select("rect")
        .style("fill", function(d) {return scope.colors[d.prediction] || "black"});

        node.select(".decision-rule")
        .text(d => nodeValues(d));

        // add new nodes
        const nodeEnter = node.enter().append("g")
        .classed("node-container", true)
        .attr("id", d => "node-" + d.id)
        .attr("transform", function(d) {
            return "translate(" + d.x + "," + d.y + ")";
        });

        nodeEnter.append("rect")
        .attr("height", side)
        .attr("width", side)
        .attr("fill", function(d) {return scope.colors[d.prediction] || "black"})
        .on("click", function(d) {
            if (scope.selectedNode && scope.selectedNode.id == d.id) {return;}
            $timeout(select(d.id, scope));
        })
        .on("mouseenter", function(d) {
            if (currentPath.has(d.id)) { return;}
            showHovered(d.id, scope);
            shift(d.id, scope, "hovered");
        })
        .on("mouseleave", function(d) {
            if (currentPath.has(d.id)) { return;}
            shift(d.id, scope, "hovered", true);
            hideUnhovered();
        });

        nodeEnter.filter(d => d.id > 0)
        .append("text")
        .attr("class", "decision-rule")
        .attr("text-anchor","middle")
        .attr("x", side / 2)
        .attr("y", - 5)
        .text(d => nodeValues(d));

        nodeEnter.filter(d => d.children_ids.length)
        .append("text")
        .attr("class", "feature-children")
        .attr("text-anchor","middle")
        .attr("x", side / 2)
        .attr("y", side + 15)
        .text(d => Format.ellipsis(scope.treeData[d.children_ids[0]].feature, 20));

        var link = svg.selectAll(".link")
        .data(links, d => d.target.id);

        // update pre-existing links
        link.attr("d", function(d) {
            return d3.svg.diagonal()({source: {x: d.source.x + side/2, y: d.source.y},
                                    target: {x: d.target.x + side/2, y: d.target.y}});
        })
        .attr("stroke", function(d) {return scope.colors[d.target.prediction] || "black"})
        .attr("stroke-width", function(d) {
                return 1+d.target.samples[1] / 5;
        });

        // add new links
        link.enter().insert("path", "g")
        .attr("class", "link")
        .attr("id", d => "link-" + d.target.id)
        .attr("d", function(d) {
            return d3.svg.diagonal()({source: {x: d.source.x + side/2, y: d.source.y},
                                  target: {x: d.target.x + side/2, y: d.target.y}});
        })
        .attr("stroke", function(d) {return scope.colors[d.target.prediction] || "black"})
        .style("fill", "none")
        .attr("stroke-width", function(d) {
                return 1+d.target.samples[1] / 5;
        })
        .attr("stroke-opacity", ".8");
    }

    const updateTooltipColors = function(colors) {
        d3.selectAll("[tooltip='tree']").selectAll("path").attr("fill", d => colors[d.data[0]]);
    }

    return {
        createTree,
        decisionRule,
        zoomFit,
        zoomBack,
        select,
        updateTooltipColors
    }
});
