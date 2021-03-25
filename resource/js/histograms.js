'use strict';
app.directive('tooltip', function(TreeUtils) {
    return {
        scope: true,
        restrict: "C",
        templateUrl: "/plugins/model-error-analysis/resource/templates/tooltip.html",
        link: function(scope, element, attr) {
            const binIndex = parseInt(attr.binIndex);
            scope.globalData = attr.wholeData;
            const histData = scope.globalData ? scope.histDataWholeSet[attr.feature] : scope.histData[attr.feature];
            const nrErrors = histData.target_distrib[TreeUtils.WRONG_PREDICTION] && histData.target_distrib[TreeUtils.WRONG_PREDICTION][binIndex] || 0;
            scope.localError = [
                nrErrors,
                nrErrors*100/histData.count[binIndex],
                nrErrors*100/TreeUtils.computeLocalError(scope.treeData[0])[1]
            ];
            scope.samples = [
                histData.count[binIndex],
                histData.count[binIndex]*100/scope.selectedNode.samples[0],
                histData.count[binIndex]*100/scope.treeData[0].samples[0]
            ];
            if (histData.bin_value) {
                scope.binName = histData.bin_value[binIndex];
            } else {
                scope.binName = `[${histData.bin_edge[binIndex]}, ${histData.bin_edge[binIndex+1]})`;
            }
        }
    };
});

app.directive("histogram", function (Format, TreeUtils, $compile) {
    return {
        scope: true,
        link: function ($scope, elem, attrs) {
            const feature = $scope.rankedFeatures.find(_ => _.name === attrs.histogram);
            let histSvg = d3.select(elem[0].children[0]).append("svg")
                .attr("width", "100%")
                .attr("height", "100%");

            const margin = {top: 15, bottom: 40, left: 40, right: 20},
                width = histSvg.node().getBoundingClientRect().width - margin.left - margin.right,
                height = 195 - margin.top - margin.bottom;
            
            histSvg = histSvg.append("g").attr("transform", "translate(" + margin.left + "," + margin.top + ")");

            // Add x & y axes
            const x = d3.scale.ordinal().rangeRoundBands([0, width], .2);
            const xAxis = d3.svg.axis().scale(x).orient("bottom");
            const xRange = feature.numerical ? $scope.histData[feature.name].mid : $scope.histData[feature.name].bin_value.slice(0,10)
            x.domain(xRange);

            const y = d3.scale.linear().range([height, 0]);
            y.domain([0, 100]);
            const yAxis = d3.svg.axis().scale(y).orient("left");
            yAxis.ticks(5);

            histSvg.append("g")
            .attr("class", "x axis")
            .attr("transform", "translate(0," + height + ")")
            .call(xAxis)
            .selectAll("text")
            .classed("x-axis__text", true)
            .text(xValue => Format.ellipsis(xValue, 10))
            .attr("transform", "translate(-15,10) rotate(-45)")
            .attr("dy", "1em");

            histSvg.append("g")
            .attr("class", "y axis")
            .call(yAxis)
            .selectAll("text")
            .classed("y-axis__text", true)
            .text(percentage => percentage + "%");

            const computeHistData = function(global) {
                const values = global ? $scope.histDataWholeSet[feature.name] : $scope.histData[feature.name];
                const samples = (global ? $scope.treeData[0]: $scope.selectedNode).samples[0]
                return xRange.map(function(x, idx) {
                    idx = (!feature.numerical && global) ? values.bin_value.indexOf(x) : idx;
                    const bar = {data: [], idx};
                    let y0 = 0;
                    for (const pred in values.target_distrib) {
                        const predBinCount = values.target_distrib[pred][idx];
                        if (!predBinCount) continue;
                        const predBinPercentage = predBinCount*100/samples;
                        bar.data.push({x, y: predBinPercentage, y0, pred});
                        y0 += predBinPercentage;
                    }
                    return bar;
                });
            }

            function update(global) {
                const data = computeHistData(global);

                // Create groups for each series, rects for each segment
                const groups = histSvg.selectAll("g.bar")
                .data(data)
                .enter()
                .append("g")
                .attr("class", global ? "histogram__bar_global" : "histogram__bar");

                // Add bars
                groups.selectAll("rect")
                .data(d => d.data)
                .enter()
                .append("rect")
                .attr("class", d => d.pred === TreeUtils.WRONG_PREDICTION ? "rect--error" : "rect--correct")
                .attr("x", d => x(d.x) + (global? x.rangeBand()/2 : 0))
                .attr("y", d => y(d.y0 + d.y))
                .attr("height", d => y(d.y0) - y(d.y0 + d.y))
                .attr("width", global ? 0 : x.rangeBand());

                if (global) {
                    histSvg.selectAll(".histogram__bar").selectAll("rect").transition().duration(200).attr("width", x.rangeBand()/2);
                    histSvg.selectAll(".histogram__bar_global").selectAll("rect").transition().duration(200).attr("width", x.rangeBand()/2);
                }

                // Add tooltips on bar hover
                groups.on("mouseenter", function(d) {
                    d3.select("#container-" + feature.name).append("div")
                    .attr("id", "tooltip-" + feature.name)
                    .attr("feature", feature.name)
                    .attr("bin-index", d.idx)
                    .attr("whole-data", global ? true : null)
                    .classed("tooltip", true)
                    .call(function() {
                        $compile(this[0])($scope);
                    })
                })
                .on('mousemove', function() {
                    const topOffset = histSvg.node().getBoundingClientRect().top + d3.mouse(this)[1] - margin.top - 75;
                    d3.select('#tooltip-' + feature.name).style("top", topOffset).style("left", margin.left + 30 + d3.mouse(this)[0]);
                })
                .on("mouseleave", function() {
                    d3.select("#tooltip-" + feature.name).remove();
                });
            }

            const unregister = $scope.$watch("histDataWholeSet." + feature.name, function(nv) {
                if (nv && $scope.leftPanel.seeGlobalChartData) {
                    update(true);
                    unregister();
                }
            });

            $scope.$watch("leftPanel.seeGlobalChartData", function(nv) {
                if (nv && $scope.histDataWholeSet[feature.name]) {
                    update(true);
                } else {
                    histSvg.selectAll(".histogram__bar_global").remove();
                    histSvg.selectAll(".histogram__bar").selectAll("rect").transition().duration(200).attr("width", x.rangeBand());
                }
            });
            
            update();
            if ($scope.leftPanel.seeGlobalChartData && $scope.histDataWholeSet[feature.name]) {
                update(true);
            }
        }
    }
});