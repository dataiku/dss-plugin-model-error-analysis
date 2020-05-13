'use strict';
app.service("Format", function() {
    return {
        ellipsis: function(text, length) {
            text = text.toString();
            if (text.length > length) {
                return (text.substr(0, length-3) + "...");
            }
            return text;
        },
        toFixedIfNeeded: function(number, decimals) {
            if(Math.round(number) !== number) {
                return number.toFixed(decimals);
            }
            return number;
        }
    };
});

app.directive("spinner", function () {
    return {
        template: "<div class='spinner-container'></div>",
        link: function (scope, element) {
            var opts = {
                lines: 6,
                length: 0,
                width: 10,
                radius: 10,
                corners: 1,
                rotate: 0,
                color: '#fff',
                speed: 2,
                trail: 60,
                shadow: false,
                hwaccel: false,
                className: 'spinner',
                zIndex: 2e9,
                top: '10px',
                left: '10px'
             };
             const spinner = new Spinner(opts);
             spinner.spin(element[0].childNodes[0]);
        }
    }
});

app.service("ModalService", function() {
    const remove = function(config) {
        return function(event) {
            if (event && !event.target.className.includes("modal-background")) return false;
            for (const key in config) {
                delete config[key];
            }
            return true;
        }
    }
    return {
        create: function(config) {
            return {
                confirm: function(msg, title, confirmAction) {
                    Object.assign(config, {
                        type: "confirm",
                        msg: msg,
                        title: title,
                        confirmAction: confirmAction
                    });
                },
                error: function(msg) {
                    Object.assign(config, {
                        type: "error",
                        msg: msg,
                        title: "Backend error"
                    });
                },
                alert: function(msg, title) {
                    Object.assign(config, {
                        type: "alert",
                        msg: msg,
                        title: title
                    });
                },
                prompt: function(inputLabel, confirmAction, res, title, msg, attrs) {
                    Object.assign(config, {
                        type: "prompt",
                        inputLabel: inputLabel,
                        promptResult: res,
                        title: title,
                        msg: msg,
                        conditions: attrs,
                        confirmAction: function() {
                            confirmAction(config.promptResult);
                        }
                    });
                }
            };
        },
        remove: remove
    }
});

app.directive("modalBackground", function($compile) {
    return {
        scope: true,
        restrict: "C",
        templateUrl: "/plugins/error-analysis/resource/templates/modal.html",
        link: function(scope, element) {
            if (scope.modal.conditions) {
                const inputField = element.find("input");
                for (const attr in scope.modal.conditions) {
                    inputField.attr(attr, scope.modal.conditions[attr]);
                }
                $compile(inputField)(scope);
            }
        }
    }
});

app.directive('tooltip', function() {
    return {
        scope: true,
        templateUrl: "/plugins/error-analysis/resource/templates/tooltip.html",
        link: function($scope, element, attr) {
            if(attr.tooltip == "tree") {
                const node = $scope.treeData[attr.node];
                $scope.probabilities = node.probabilities;
                $scope.samples = node.samples;
                $scope.globalError = node.global_error;

                d3.select(element[0].children[0])
                .attr("x", -30)
                .attr("y", -25)
                .attr("height", 120)
                .attr("width", 240)
                .select(".tooltip-info")
                .classed("tooltip-info-tree", true);

                // Compute the position of each group on the pie
                var pie = d3.layout.pie()
                    .value(function(d) {return d[1];});
                var proba = pie($scope.probabilities);

                // Build the pie chart
                d3.select("#tooltip-" + node.id)
                .append("g")
                .attr("transform", "translate(5, 10)")
                .selectAll('.camembert')
                .data(proba)
                .enter()
                .append('path')
                .attr('d', d3.svg.arc()
                    .innerRadius(0)
                    .outerRadius(30)
                )
                .attr('fill', function(d) {
                    return $scope.colors[d.data[0]];
                });
            }

            if (attr.tooltip == "histogram") {
                let histData;
                if (attr.wholeData) {
                    $scope.dataOrigin = "Whole data";
                    histData = $scope.histDataWholeSet[attr.feature][attr.binIndex];
                } else {
                    $scope.dataOrigin = "Subset data (at this node)"
                    histData = $scope.histData[attr.feature][attr.binIndex];
                }
                $scope.probabilities = Object.entries(histData.target_distrib);
                $scope.probabilities.sort(function(a, b) {
                    return b[1] - a[1];
                });
                $scope.probabilities = $scope.probabilities.slice(0, 5).map(_ => [_[0], _[1] / histData.count]);
                $scope.samples = [$scope.toFixedIfNeeded(histData.count, 2, true)];
                $scope.binName = histData.value;

                d3.select(element[0].children[0])
                .attr("width", 190)
                .attr("height", 80 + $scope.probabilities.length * 22);
            }
        }
    };
});

app.directive('focusHere', function ($timeout) {
    return {
        restrict: 'A',
        link: function (scope, element) {
            $timeout(function() {
                element[0].focus();
            });
        }
    };
});
