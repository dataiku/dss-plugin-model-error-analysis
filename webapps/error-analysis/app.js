(function() {
    'use strict';

    app.controller("MeaController", function($scope, ModalService) {
        $scope.modelId = dataiku.getWebAppConfig().modelId;
        $scope.modal = {};
        $scope.removeModal = function(event) {
            if (ModalService.remove($scope.modal)(event)) {
                angular.element(".template").focus();
            }
        };
        $scope.createModal = ModalService.create($scope.modal);
    });

    app.controller("EditController", function($scope, $http, TreeInteractions, TreeUtils, Format) {
        $scope.loadingHistogram = true;
        let targetValues;

        $scope.colors = { // TODO
            "Wrong prediction": "#CE1228",
            "Correct prediction": "#CCC"
        };

        const radius = 16;
        const node = d3.select(".tree-legend_pie svg").append("g");
        TreeUtils.addNode(node, radius, d=>.4, d=>"X", true);

        const create = function(data) {
            $scope.treeData = data.nodes;
            angular.forEach($scope.treeData, function(node) {
                node.localError = TreeUtils.computeLocalError(node);
            });
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
