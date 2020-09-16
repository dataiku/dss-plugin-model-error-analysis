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
                .attr("height", "100%");
  
            //Create hatched pattern
            const hatchSize = 5;
            const defs = histSvg.append("defs");
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

            
            histSvg = histSvg.append("g").attr("transform", "translate(" + margin.left + "," + margin.top + ")");

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
                })
                .on("mousemove", function(){
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
                    d3.select("[tooltip='histogram']").remove();
                });
            }

            const addGroupProperties = function(groups, wholeData) {
                groups.selectAll("rect")
                .data(d => d)
                .enter()
                .append("rect")
                .attr("fill", d => d.color)
                .attr("x", d => x(d.x) + (wholeData? x.rangeBand()/2 : 0))
                .attr("y", d => y(d.y0 + d.y))
                .attr("height", d => y(d.y0) - y(d.y0 + d.y))
                .attr("width", x.rangeBand()/2);
            }

            function update() {
                const data = [];
                const dataWhole = [];
                let predArray;
                if ($scope.selectedNode.probabilities[0][0] == "Wrong prediction") {
                    predArray = ["Wrong prediction", "Correct prediction"]
                } else {
                    predArray = ["Correct prediction", "Wrong prediction"]
                }

                const values = $scope.histData[feature];
                const valuesWhole = $scope.histDataWholeSet[feature];
                if (feature in $scope.features) {
                    const values = $scope.histData[feature];
                    values.mid.forEach(function(mid, idx) {
                        const bar = [];
                        let y0 = 0;
                        predArray.forEach(function(prediction) {
                            if (values.target_distrib[prediction][idx]) {
                                bar.push({x: mid,
                                    y: values.target_distrib[prediction][idx],
                                    y0: y0,
                                    color: $scope.colors[prediction],
                                    interval: `[${values.bin_edge[idx]}, ${values.bin_edge[idx+1]})`
                                });
                                y0 += values.target_distrib[prediction][idx];
                            }
                        });
                        data.push(bar);
                    });
                    valuesWhole.mid.map(function(mid, idx) {
                        const bar = [];
                        let y0 = 0;
                        predArray.forEach(function(prediction) {
                            if (valuesWhole.target_distrib[prediction][idx]) {
                                bar.push({x: mid,
                                    y: valuesWhole.target_distrib[prediction][idx],
                                    y0: y0,
                                    color: $scope.colors[prediction],
                                    interval: `[${valuesWhole.bin_edge[idx]}, ${valuesWhole.bin_edge[idx+1]})`
                                });
                                y0 += valuesWhole.target_distrib[prediction][idx];
                            }
                        });
                        dataWhole.push(bar);
                    });
                    yAxis.ticks(5)
                    y.domain([0, 1]);
                    x.domain(values.mid);
                } else {
                    values.bin_value.forEach(function(bin_value, idx) {
                        const bar = [];
                        let y0 = 0;
                        predArray.forEach(function(prediction) {
                            if (values.target_distrib[prediction][idx]) {
                                bar.push({x: bin_value,
                                    y: values.target_distrib[prediction][idx],
                                    y0: y0,
                                    color: $scope.colors[prediction]
                                });
                                y0 += values.target_distrib[prediction][idx];
                            }
                        });
                        data.push(bar);
                    });
                    values.bin_value.forEach(function(bin_value, idx) {
                        const bar = [];
                        let y0 = 0;
                        predArray.forEach(function(prediction) {
                            if (valuesWhole.target_distrib[prediction][idx]) {
                                bar.push({x: bin_value,
                                    y: valuesWhole.target_distrib[prediction][idx],
                                    y0: y0,
                                    color: $scope.colors[prediction]
                                });
                                y0 += valuesWhole.target_distrib[prediction][idx];
                            }
                        });
                        dataWhole.push(bar);
                    });
                    yAxis.ticks(5)
                    y.domain([0, 1]);
                    x.domain(values.bin_value);
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
                let groups = histSvg.selectAll("g.bar")
                .data(data)
                .enter()
                .append("g")
                .classed("histogram__bar", true);

                let groupsWhole = histSvg.selectAll("g.bar")
                .data(dataWhole)
                .enter()
                .append("g")
                .classed("histogram__bar histogram__bar_global", true);

                addGroupProperties(groups);
                addGroupProperties(groupsWhole, true);

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