'use strict';
app.directive('tooltipHistogram', function(TreeUtils) {
    return {
        scope: true,
        restrict: "C",
        templateUrl: "/plugins/model-error-analysis/resource/templates/tooltip.html",
        link: function(scope, element, attr) {
            const binIndex = parseInt(attr.binIndex);
            scope.globalData = attr.wholeData;
            const histData = scope.globalData ? scope.histDataWholeSet[attr.feature] : scope.histData[attr.feature];
            const probaError = histData.target_distrib[TreeUtils.WRONG_PREDICTION];
            if (probaError && probaError[binIndex]) {
                scope.localError = probaError[binIndex] * 100;
            } else {
                scope.localError = 0;
            }
            scope.samples = [histData.count[binIndex],
                            histData.count[binIndex]/scope.selectedNode.samples[0]*100];
            if (histData.bin_value) {
                scope.binName = histData.bin_value[binIndex];
            } else {
                scope.binName = `[${histData.bin_edge[binIndex]}, ${histData.bin_edge[binIndex+1]})`;
            }

            d3.select(element[0].children[0])
            .attr("width", 190)
            .attr("height", 80);
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

            const margin = {top: 15, bottom: 40, left: 30, right: 20},
                width = histSvg.node().getBoundingClientRect().width - margin.left - margin.right,
                height = 195 - margin.top - margin.bottom;
            
            histSvg = histSvg.append("g").attr("transform", "translate(" + margin.left + "," + margin.top + ")");

            const x = d3.scale.ordinal().rangeRoundBands([0, width], .2);
            const y = d3.scale.linear().range([height, 0]);
            const xAxis = d3.svg.axis().scale(x).orient("bottom");
            const yAxis = d3.svg.axis().scale(y).orient("left");
            yAxis.ticks(5);
            y.domain([0, 100]);

            const addInteractions = function(groups, onWholeSet) {
                groups.on("mouseenter", function(d) {
                    histSvg.append("g")
                    .classed("tooltip", true)
                    .classed("tooltip-histogram", true)
                    .attr("feature", feature.name)
                    .attr("whole-data", onWholeSet)
                    .attr("bin-index", d.idx)
                    .call(function() {
                        $compile(this[0])($scope);
                    });
                })
                .on("mousemove", function(){
                    let xPosition = d3.mouse(this)[0] + 20;
                    let yPosition = d3.mouse(this)[1];
                    const histogramDim = d3.select(".histogram-svg").node().getBoundingClientRect();
                    const tooltipDim = d3.select(".tooltip-histogram").node().getBoundingClientRect();
                    if (xPosition + 25 + tooltipDim.width > histogramDim.width) {
                        xPosition -= 30 + tooltipDim.width;
                    }
                    if (yPosition + 15 + tooltipDim.height > histogramDim.height) {
                        yPosition -= (yPosition + tooltipDim.height) - histogramDim.height + 15;
                    }
                    d3.select(".tooltip-histogram").attr("transform", "translate(" + xPosition + "," + yPosition + ")");
                })
                .on("mouseleave", function() {
                    d3.select(".tooltip-histogram").remove();
                });
            }

            const addGroupProperties = function(groups, wholeData) {
                groups.selectAll("rect")
                .data(d => d.data)
                .enter()
                .append("rect")
                .attr("class", d => d.pred === TreeUtils.WRONG_PREDICTION ? "rect--error" : "rect--correct")
                .attr("x", d => x(d.x) + (wholeData? x.rangeBand()/2 : 0))
                .attr("y", d => y(d.y0 + d.y))
                .attr("height", d => y(d.y0) - y(d.y0 + d.y))
                .attr("width", wholeData ? 0 : x.rangeBand());

                if (wholeData) {
                    histSvg.selectAll(".histogram__bar").selectAll("rect").transition().duration(400).attr("width", x.rangeBand()/2);
                    histSvg.selectAll(".histogram__bar_global").selectAll("rect").transition().duration(400).attr("width", x.rangeBand()/2);
                }
            }

            function update(global) {
                let predArray = $scope.selectedNode.probabilities.map(_ => _[0]);
                const values = global ? $scope.histDataWholeSet[feature.name] : $scope.histData[feature.name];
                const data = [];
                if (feature.numerical) {
                    values.mid.forEach(function(mid, idx) {
                        const bar = {data: [], idx};
                        let y0 = 0;
                        predArray.forEach(function(prediction) {
                            if (values.target_distrib[prediction][idx]) {
                                const height = values.target_distrib[prediction][idx]*100;
                                bar.data.push({
                                    x: mid,
                                    y: height,
                                    y0: y0,
                                    pred: prediction,
                                    interval: `[${values.bin_edge[idx]}, ${values.bin_edge[idx+1]})`
                                });
                                y0 += height;
                            }
                        });
                        data.push(bar);
                    });
                    x.domain(values.mid);
                } else {
                    $scope.histData[feature.name].bin_value.slice(0,10).forEach(function(bin_value, idx) { // TODO: useless slice?
                        idx = global ? values.bin_value.indexOf(bin_value) : idx;
                        const bar = {data: [], idx};
                        let y0 = 0;
                        predArray.forEach(function(prediction) {
                            if (values.target_distrib[prediction][idx]) {
                                const height = values.target_distrib[prediction][idx]*100;
                                bar.data.push({
                                    x: bin_value,
                                    y: height,
                                    y0: y0,
                                    pred: prediction
                                });
                                y0 += height;
                            }
                        });
                        data.push(bar);
                    });
                    x.domain($scope.histData[feature.name].bin_value.slice(0,10));
                }
                histSvg.append("g")
                    .attr("class", "x axis")
                    .attr("transform", "translate(0," + height + ")")
                    .call(xAxis)
                    .selectAll("text")
                    .classed("x-axis__text", true)
                    .attr("transform", "translate(-15,10) rotate(-45)")
                    .attr("dy", "1em");

                histSvg.selectAll(".tick text").text(function(d) {
                    return Format.ellipsis(d, 10);
                });

                histSvg.append("g")
                    .attr("class", "y axis")
                    .call(yAxis)
                    .append("text")
                    .classed("y-axis__text", true)
                    .attr("transform", "rotate(-90)")
                    .attr("y", 6)
                    .attr("dy", ".71em");

                // Create groups for each series, rects for each segment
                const groups = histSvg.selectAll("g.bar")
                .data(data)
                .enter()
                .append("g")
                .attr("class", global ? "histogram__bar_global" : "histogram__bar");

                addGroupProperties(groups, global);
                addInteractions(groups, global);
            }

            const unregister = $scope.$watch("histDataWholeSet." + feature.name, function(nv) {
                if (nv && $scope.seeGlobalChartData) {
                    update(true);
                    unregister();
                }
            });

            $scope.$watch("seeGlobalChartData", function(nv) {
                if (nv && $scope.histDataWholeSet[feature.name]) {
                    update(true);
                } else {
                    histSvg.selectAll(".histogram__bar_global").remove();
                    histSvg.selectAll(".histogram__bar").selectAll("rect").transition().duration(400).attr("width", x.rangeBand());
                }
            });
            
            $scope.$watch("selectedNode", function(nv) {
                if (nv) {
                    histSvg.selectAll("rect").remove();
                    histSvg.selectAll("g").remove();
                    update();
                    if ($scope.seeGlobalChartData && $scope.histDataWholeSet[feature.name]) {
                        update(true);
                    }
                }
            })
        }
    }
});