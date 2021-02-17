(function() {
    'use strict';

    app.controller("MeaController", function($scope, ModalService) {
        $scope.config = {};
        $scope.modal = {};
        $scope.removeModal = function(event) {
            if (ModalService.remove($scope.modal)(event)) {
                angular.element(".template").focus();
            }
        };
        $scope.createModal = ModalService.create($scope.modal);
    });

    app.controller("EditController", function($scope, $http, TreeInteractions, Format) {
        $scope.loadingHistogram = true;
        let targetValues;

        $scope.colors = { // TODO
            "Wrong prediction": "#CE1228",
            "Correct prediction": "#CCC"
        };

        const radius = 16;
        const node = d3.select(".tree-legend_pie svg")
        .append("g");

        node.append("circle")
        .attr("class", "node__background selected")
        .attr("cx", radius)
        .attr("cy", radius)
        .attr("r", radius);

        node.append("path")
        .attr("class", "node__gauge selected")
        .attr("d", function(d) {
            const localError = .4;
            const innerRadius = radius - 2;
            const theta = Math.PI*(-localError+.5);
            const start = {
                x: innerRadius*Math.cos(theta) + radius,
                y: innerRadius*Math.sin(theta) + radius
            };
            const end = {
                x: -innerRadius*Math.cos(theta) + radius,
                y: start.y
            };
            const largeArcFlag = theta > 0 ? 0 : 1;
            return `M ${start.x} ${start.y} A ${innerRadius} ${innerRadius} 0 ${largeArcFlag} 1 ${end.x} ${end.y}`;
        });

        node.append("text")
        .attr("text-anchor","middle")
        .attr("x", radius)
        .attr("y", radius)
        .text("X");

        const create = function(data) {
            $scope.treeData = data.nodes;
            $scope.features = data.features;
            $scope.rankedFeatures = data.rankedFeatures;
            $scope.metrics = {
                actual: data.actualAccuracy,
                estimated: data.estimatedAccuracy
            }
            targetValues = data.target_values;
            TreeInteractions.createTree($scope);
            $scope.loadingTree = false;
        }

        $scope.load = function() {
            $scope.loadingTree = true;
            $http.get(getWebAppBackendUrl("load"))
            .then(function(response) {
                create(response.data);
            }, function(e) {
                $scope.loadingTree = false;
                $scope.createModal.error(e.data);
            });
        }

        $scope.load();

        $scope.zoomFit = function() {
            TreeInteractions.zoomFit();
        }

        $scope.zoomBack = function() {
            TreeInteractions.zoomBack($scope.selectedNode);
        }

        $scope.toFixedIfNeeded = function(number, decimals, precision) {
            if (number === undefined) return;
            const lowerBound = 5 * Math.pow(10, -decimals-1);
            if (number && Math.abs(number) < lowerBound) {
                if (precision) { // indicates that number is very small instead of rounding to 0 (given that number is positive)
                    return "<" + lowerBound;
                }
                return 0;
            }
            return Format.toFixedIfNeeded(number, decimals);
        }
    });
})();
