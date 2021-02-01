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
        templateUrl: "/plugins/model-error-analysis/resource/templates/modal.html",
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

app.directive('tooltipTree', function() {
    return {
        scope: true,
        restrict: "C",
        templateUrl: "/plugins/model-error-analysis/resource/templates/tooltip.html",
        link: function(scope, element, attr) {
            const node = scope.treeData[attr.node];
            scope.probabilities = node.probabilities;
            scope.samples = node.samples;
            scope.globalError = node.global_error;

            scope.inRightPanel = attr.hasOwnProperty("rightPanel");
            d3.select(element[0].children[0])
            .attr("x", scope.inRightPanel ? 0 : -30)
            .attr("y", scope.inRightPanel ? 0 : -25)
            .attr("height", scope.inRightPanel ? "100%" : 120)
            .attr("width", scope.inRightPanel ? "100%" : 260)
            .select(".tooltip-container")
            .classed("tooltip-container-tree", !scope.inRightPanel)
            .classed("tooltip-container-rp", scope.inRightPanel);

            // Compute the position of each group on the pie
            const pie = d3.layout.pie()
                .value(function(d) {return d[1];});
            const proba = pie(scope.probabilities);

            // Build the pie chart
            d3.select(scope.inRightPanel ? "#tooltip-right-panel" : "#tooltip-" + node.node_id)
            .append("g")
            .attr("transform", scope.inRightPanel ? "translate(50, 50)" : "translate(10, 25)")
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

app.directive('tooltipHistogram', function() {
    return {
        scope: true,
        restrict: "C",
        templateUrl: "/plugins/model-error-analysis/resource/templates/tooltip.html",
        link: function(scope, element, attr) {
            scope.inHistogram = true;
            const binIndex = parseInt(attr.binIndex);
            const histData = attr.wholeData ? scope.histDataWholeSet[attr.feature] : scope.histData[attr.feature];
            scope.probabilities = Object.entries(histData.target_distrib).map(_ => [_[0], _[1][binIndex]]);
            scope.probabilities.sort(function(a, b) {
                return b[1] - a[1];
            });
            scope.probabilities = scope.probabilities.slice(0, 5).map(_ => [_[0], _[1]]);
            scope.samples = [histData.count[binIndex],
                            histData.count[binIndex]/scope.selectedNode.samples[0]];
            if (histData.bin_value) {
                scope.binName = histData.bin_value[binIndex];
            } else {
                scope.binName = `[${histData.bin_edge[binIndex]}, ${histData.bin_edge[binIndex+1]})`;
            }

            d3.select(element[0].children[0])
            .attr("width", 190)
            .attr("height", 60 + scope.probabilities.length * 22);
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
