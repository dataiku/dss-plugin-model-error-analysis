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

        $scope.leftPanel = {};
        const node = d3.select(".tree-legend_pie svg").append("g");
        TreeUtils.addNode(node, 16, d=>.4, false, "X");
        
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
            $scope.actualErrorRate =  1 - data.actualAccuracy;
            TreeInteractions.createTree($scope.treeData, $scope.selectNode);
            $scope.loadingTree = false;
        }

        const load = function() {
            $scope.loadingTree = true;
            $http.get(getWebAppBackendUrl("load"))
            .then(function(response) {
                create(response.data);
                $scope.histDataWholeSet = {};
                $scope.histData = {};
            }, function(e) {
                $scope.loadingTree = false;
                $scope.createModal.error(e.data);
            });
        }

        const selectFeatures = function() {
            const selectedFeatures = $scope.rankedFeatures.filter(_ => _.$selected);
            if (!selectedFeatures.length) return;
            $http.post(getWebAppBackendUrl("select-features"), {"feature_ids": selectedFeatures.map(_ => _.rank)})
            .then(function() {
                loadHistograms(selectedFeatures);
                $scope.leftPanel.seeGlobalChartData && fetchGlobalChartData(selectedFeatures);
            }, function(e) {
                $scope.createModal.error(e.data);
            });
        }

        $scope.interactWithFeatureSelector = function() {
            if ($scope.featureSelectorShown) {
                selectFeatures();
            }
            $scope.featureSelectorShown = !$scope.featureSelectorShown;
        }

        $scope.displayOrHideGlobalData = function() {
            $scope.leftPanel.seeGlobalChartData = !$scope.leftPanel.seeGlobalChartData;
            if ($scope.leftPanel.seeGlobalChartData) {
                fetchGlobalChartData($scope.rankedFeatures.filter(_ => _.$selected));
            }
        }

        const fetchGlobalChartData = function(selectedFeatures) {
            if (!selectedFeatures.filter(_ => !$scope.histDataWholeSet[_.name]).length) return;
            $http.get(getWebAppBackendUrl("global-chart-data"))
            .then(function(response) {
                Object.assign($scope.histDataWholeSet, response.data);
            }, function(e) {
                $scope.createModal.error(e.data);
            });
        }

        const loadHistograms = function(selectedFeatures) {
            if (selectedFeatures && !selectedFeatures.filter(_ => !$scope.histData[_.name]).length) return;
            $http.get(getWebAppBackendUrl("select-node/" + $scope.selectedNode.node_id))
                .then(function(response) {
                    Object.assign($scope.histData, response.data);
                }, function(e) {
                    $scope.createModal.error(e.data);
                });
        }

        $scope.zoomFit = TreeInteractions.zoomFit;

        $scope.zoomBack = () => TreeInteractions.zoomBack($scope.selectedNode);

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

        $http.get(getWebAppBackendUrl("original-model-info")).then(function(response) {
            $scope.isRegression = response.data.isRegression;
            $scope.originalModelName = response.data.modelName;
            load();
        }, function(e) {
            $scope.createModal.error(e.data);
        });

        $scope.selectNode = function(nodeId) {
            if (nodeId === ($scope.selectedNode && $scope.selectedNode.node_id)) return;
            $scope.selectedNode = $scope.treeData[nodeId];
            TreeInteractions.selectNode($scope.selectedNode, $scope.treeData);
            $scope.leftPanel = {};
            $scope.leftPanel.decisionRule = TreeUtils.getPath($scope.selectedNode.node_id, $scope.treeData, true).map(_ => TreeUtils.getDecisionRule($scope.treeData[_]));
            $scope.histData = {};
            loadHistograms();
        };
    });
})();
