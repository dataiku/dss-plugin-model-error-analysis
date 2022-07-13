(function() {
    'use strict';

    app.controller("MeaController", function($scope, $http, ModalService, TreeInteractions, TreeUtils, Format) {
        $scope.$on("closeModal", function() {
            angular.element(".template").focus();
        });

        $scope.leftPanel = {};
        const node = d3.select(".tree-legend_pie svg").append("g");
        TreeUtils.addNode(node, 16, d=>.4, false, "X");

        let chartFeatures = new Set();
        const DEFAULT_MAX_NR_FEATURES = 5;
        const create = function(data) {
            $scope.treeData = data.nodes;
            $scope.rankedFeatures = data.rankedFeatures;
            for (let idx = 0; idx < Math.min(DEFAULT_MAX_NR_FEATURES, $scope.rankedFeatures.length); idx++) {
                $scope.rankedFeatures[idx].$selected = true;
                chartFeatures.add($scope.rankedFeatures[idx].name);
            }
            $scope.epsilon = data.epsilon;
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
                ModalService.createBackendErrorModal($scope, e.data);
            });
        }

        const selectFeatures = function() {
            const selectedFeatures = $scope.rankedFeatures.filter(_ => _.$selected);
            if ((selectedFeatures.length === chartFeatures.size)
                && selectedFeatures.every(_ => chartFeatures.has(_.name))) return;
            $http.post(getWebAppBackendUrl("select-features"), {"feature_ids": selectedFeatures.map(_ => _.rank)})
            .then(function() {
                chartFeatures = new Set(selectedFeatures.map(_ => _.name));
                loadHistograms(selectedFeatures);
                fetchGlobalChartData(selectedFeatures);
            }, function(e) {
                ModalService.createBackendErrorModal($scope, e.data);
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
            fetchGlobalChartData($scope.rankedFeatures);
        }

        const fetchGlobalChartData = function(selectedFeatures) {
            if (!$scope.leftPanel.seeGlobalChartData || !chartFeatures.size) return;
            if (!selectedFeatures.filter(_ => !$scope.histDataWholeSet[_.name] && _.$selected).length) return;
            $http.get(getWebAppBackendUrl("global-chart-data"))
            .then(function(response) {
                Object.assign($scope.histDataWholeSet, response.data);
            }, function(e) {
                ModalService.createBackendErrorModal($scope, e.data);
            });
        }

        const loadHistograms = function(selectedFeatures) {
            if (!chartFeatures.size) return;
            if (selectedFeatures && !selectedFeatures.filter(_ => !$scope.histData[_.name]).length) return;
            $http.get(getWebAppBackendUrl("select-node/" + $scope.selectedNode.node_id))
                .then(function(response) {
                    Object.assign($scope.histData, response.data);
                }, function(e) {
                    ModalService.createBackendErrorModal($scope, e.data);
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
            ModalService.createBackendErrorModal($scope, e.data);
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
