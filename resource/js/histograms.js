'use strict';
app.directive("histogram", function (Format, $compile) {
    return {
        scope: true,
        link: function ($scope, elem, attrs) {
            const feature = attrs.histogram;
            const margin = {top: 15, bottom: 40, left: 30, right: 20},
                width = 415 - margin.left - margin.right,
                height = 195 - margin.top - margin.bottom;

                let histSvg = d3.select(elem[0].children[0]).append("svg")
                    .attr("width", "100%")
                    .attr("height", "100%")
                    .append("g")
                    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

            const x = d3.scale.ordinal().rangeRoundBands([0, width], .2);
            const y = d3.scale.linear().range([height, 0]);
            const xAxis = d3.svg.axis().scale(x).orient("bottom");
            const yAxis = d3.svg.axis().scale(y).orient("left");

            const addInteractions = function(groups, onWholeSet) {
                groups.on("mouseenter", function(d, i) {
                    histSvg.append("g")
                    .attr("tooltip", "histogram")
                    .attr("feature", feature)
                    .attr("whole-data", onWholeSet)
                    .attr("bin-index", i)
                    .call(function() {
                        $compile(this[0])($scope);
                    });

                    d3.select(this).style("opacity", .7);
                })
                .on("mousemove", function(d, i){
                    let xPosition = d3.mouse(this)[0] + 20;
                    let yPosition = d3.mouse(this)[1];
                    const histogramDim = d3.select(".histogram-svg").node().getBoundingClientRect();
                    const tooltipDim = d3.select("[tooltip='histogram']").node().getBoundingClientRect();
                    if (xPosition + 25 + tooltipDim.width > histogramDim.width) {
                        xPosition -= 30 + tooltipDim.width;
                    }
                    if (yPosition + 15 + tooltipDim.height > histogramDim.height) {
                        yPosition -= (yPosition + tooltipDim.height) - histogramDim.height + 15;
                    }
                    d3.select("[tooltip='histogram']").attr("transform", "translate(" + xPosition + "," + yPosition + ")");
                })
                .on("mouseleave", function() {
                    d3.select(this).style("opacity", null);
                    d3.select("[tooltip='histogram']").remove();
                });
            }

            const addGroupProperties = function(groups, darkerShade) {
                groups.selectAll("rect")
                .data(d => d)
                .enter()
                .append("rect")
                .style("fill", function(d) {
                    return darkerShade ? d3.rgb(d.color).darker(1) : d.color;
                })
                .attr("y", d => y(d.y0 + d.y))
                .attr("height", d => y(d.y0) - y(d.y0 + d.y))
                .attr("width", x.rangeBand()/2);
            }

            function update() {
                let data;
                let dataWhole;
                let predArray;
                if ($scope.selectedNode.probabilities[0][0] == "Wrong prediction") {
                    predArray = ["Wrong prediction", "Correct prediction"]
                } else {
                    predArray = ["Correct prediction", "Wrong prediction"]
                }

                if (feature in $scope.features) {
                    const values = $scope.histData[feature];
                    data = values.map(function(d) {
                        const bar = [];
                        let y0 = 0;
                        $scope.selectedNode.probabilities.forEach(function(proba, i) {
                            if (d.target_distrib[proba[0]]) {
                                bar.push({x: d.mid,
                                    y: d.target_distrib[proba[0]],
                                    y0: y0,
                                    color: $scope.colors[proba[0]],
                                    interval: d.value
                                });
                                y0 += d.target_distrib[proba[0]];
                            }
                        });
                        return bar;
                    });
                    const valuesWhole = $scope.histDataWholeSet[feature];
                    dataWhole = valuesWhole.map(function(d) {
                        const bar = [];
                        let y0 = 0;
                        predArray.forEach(function(proba, i) {
                            if (d.target_distrib[proba]) {
                                bar.push({x: d.mid,
                                    y: d.target_distrib[proba],
                                    y0: y0,
                                    color: $scope.colors[proba],
                                    interval: d.value
                                });
                                y0 += d.target_distrib[proba];
                            }
                        });
                        return bar;
                    });
                    yAxis.ticks(5)
                    y.domain([0, 1]);
                    x.domain(values.map(_ => _.mid));
                } else {
                    const values = $scope.histData[feature];
                    data = values.filter(_ => _.target_distrib).map(function(d) {
                        const bar = [];
                        let y0 = 0;
                        $scope.selectedNode.probabilities.forEach(function(proba, i) {
                            if (d.target_distrib[proba[0]]) {
                                bar.push({x: d.value,
                                    y: d.target_distrib[proba[0]],
                                    y0: y0,
                                    color: $scope.colors[proba[0]]
                                });
                                y0 += d.target_distrib[proba[0]];
                            }
                        });
                        return bar;
                    });
                    let predArray;
                    if ($scope.selectedNode.probabilities[0][0] == "Wrong prediction") {
                        predArray = ["Wrong prediction", "Correct prediction"];
                    } else {
                        predArray = ["Correct prediction", "Wrong prediction"];
                    }
                    dataWhole = $scope.histDataWholeSet[feature].filter(_ => _.target_distrib).map(function(d) {
                        const bar = [];
                        let y0 = 0;
                        predArray.forEach(function(proba, i) {
                            if (d.target_distrib[proba]) {
                                bar.push({x: d.value,
                                    y: d.target_distrib[proba],
                                    y0: y0,
                                    color: $scope.colors[proba]
                                });
                                y0 += d.target_distrib[proba];
                            }
                        });
                        return bar;
                    });
                    yAxis.ticks(5)
                    y.domain([0, 1]);
                    x.domain(values.map(_ => _.value));
                }

                histSvg.append("g")
                    .attr("class", "x axis")
                    .attr("transform", "translate(0," + height + ")")
                    .call(xAxis)
                    .selectAll("text")
                    .style("text-anchor", "middle")
                    .attr("transform", "translate(-15,10) rotate(-45)")
                    .attr("dy", "1em");

                histSvg.selectAll(".tick text").text(function(d) {
                    return Format.ellipsis(d, 10);
                });

                histSvg.append("g")
                    .attr("class", "y axis")
                    .call(yAxis)
                    .append("text")
                    .attr("transform", "rotate(-90)")
                    .attr("y", 6)
                    .attr("dy", ".71em")
                    .style("text-anchor", "end")

                // Create groups for each series, rects for each segment
                let groups = histSvg.selectAll("g.bar")
                .data(data)
                .enter()
                .append("g");

                let groupsWhole = histSvg.selectAll("g.bar")
                .data(dataWhole)
                .enter()
                .append("g");

                addGroupProperties(groups);
                addGroupProperties(groupsWhole, true);

                groups.selectAll("rect").attr("x", d => x(d.x));
                groupsWhole.selectAll("rect").attr("x", d => x(d.x) + x.rangeBand()/2);

                addInteractions(groups);
                addInteractions(groupsWhole, true);
            }

            $scope.$watch("selectedNode", function(nv) {
                if (nv) {
                    histSvg.selectAll("rect").remove();
                    histSvg.selectAll("g").remove();
                    update();
                }
            })
        }
    }
});