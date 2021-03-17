(function() {
    'use strict';

    app.controller("MeaController", function($scope, $http, ModalService, TreeInteractions, TreeUtils, Format) {
        $scope.modal = {};
        $scope.removeModal = function(event) {
            if (ModalService.remove($scope.modal)(event)) {
                angular.element(".template").focus();
            }
        };
        $scope.createModal = ModalService.create($scope.modal);
        
        $scope.modelId = dataiku.getWebAppConfig().modelId;
        
        const radius = 16;
        const node = d3.select(".tree-legend_pie svg").append("g");
        TreeUtils.addNode(node, radius, d=>.4, d=>"X", true);
        
        const DEFAULT_MAX_NR_FEATURES = 5;
        const create = function(data) {
            $scope.treeData = data.nodes;
            angular.forEach($scope.treeData, function(node) {
                node.localError = TreeUtils.computeLocalError(node);
            });
            $scope.rankedFeatures = data.rankedFeatures;
            $scope.rankedFeatures.forEach(function(rankedFeature, idx) {
                rankedFeature.$selected = idx < DEFAULT_MAX_NR_FEATURES;
            });
            $scope.metrics = {
                actual: 1 - data.actualAccuracy,
                estimated: 1 - data.estimatedAccuracy
            }
            TreeInteractions.createTree($scope);
            $scope.loadingTree = false;
        }

        const load = function() {
            $scope.loadingTree = true;
            $http.get(getWebAppBackendUrl("load"))
            .then(function(response) {
                create(response.data);
                $scope.histDataWholeSet = {};
                $scope.histData = {};
                selectFeatures();
            }, function(e) {
                $scope.loadingTree = false;
                $scope.createModal.error(e.data);
            });
        }

        const selectFeatures = function() {
            const selectedFeatures = $scope.rankedFeatures.filter(_ => _.$selected);
            if (!selectedFeatures.length) return;
            $http.post(getWebAppBackendUrl("select-features"), {"feature_ids": selectedFeatures.map(_ => _.rank)})
            .then(function(response) {
                Object.assign($scope.histDataWholeSet, response.data);
                if ($scope.selectedNode && selectedFeatures.filter(_ => !$scope.histData[_.name]).length) {
                    loadHistograms();
                }
            }, function(e) {
                $scope.loadingTree = false;
                $scope.createModal.error(e.data);
            });
        }

        $scope.interactWithFeatureSelector = function(openedSelector, event) {
            if (event && event.keyCode != 27) return;
            if (openedSelector) {
                selectFeatures();
            }
            $scope.featureSelectorShown = !openedSelector;
        }

        const loadHistograms = function() {
            $http.get(getWebAppBackendUrl("select-node/" + $scope.selectedNode.node_id))
                .then(function(response) {
                    Object.assign($scope.histData, response.data);
                }, function(e) {
                    $scope.createModal.error(e.data);
                });
        }

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

        load();
    });
})();
