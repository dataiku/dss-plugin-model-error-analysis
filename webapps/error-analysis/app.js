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
        $scope.scales = {
            "Default": d3.scale.category20().range().concat(d3.scale.category20b().range()),
            "DSS Next": ["#D11141", "#FFC425", "#00AEDB", "#8CC63F", "#F37735", "#91268F", "#194BA3", "#00B159"],
            "Pastel": ["#EC6547", "#538BC8", "#FDC665", "#95C37B", "#75C2CC", "#694A82", "#65B890", "#A874A0"],
            "Corporate": ["#EF3C39", "#0075B2", "#818991", "#EA9423", "#A4C2DB", "#009D4B", "#CFD6D3", "#231F20"],
            "Deuteranopia": ["#D6C222",  "#193C81",  "#211924", "#757A8D", "#776A37", "#AE963A", "#655E5D"],
            "Tritanopia": ["#CA0849", "#0B4D61", "#E4B2BF", "#3F6279", "#F24576", "#7D8E98", "#9C4259", "#2B2A2E"],
            "Pastel 2": ["#f06548", "#7bc9a6", "#fdc766",  "#4ec5da", "#548ecb", "#97668f", "#5e2974"]
        };
        let targetValues;

        $scope.displayScale = function(scale) {
            if (!scale) return [];
            return scale.slice(0,5);
        }

        $scope.setScale = function(scaleName) {
            setScale(scaleName);
            TreeInteractions.select($scope.selectedNode.node_id, $scope, false, true);
            TreeInteractions.updateTooltipColors($scope.colors);
        }

        const setScale = function(scaleName) {
            $scope.selectedScale = $scope.scales[scaleName];
            $scope.colors = {};
            $scope.colors["Wrong prediction"] = $scope.selectedScale[0];
            $scope.colors["Correct prediction"] = $scope.selectedScale[1];
        }

        $scope.closeColorPicker = function(event) {
            if (event.target.matches('.color-picker') || event.target.matches('.icon-tint')) return;
            $scope.displayColorPicker = false;
        }

        const create = function(data) {
            $scope.treeData = data.nodes;
            $scope.features = data.features;
            $scope.rankedFeatures = data.rankedFeatures;
            targetValues = data.target_values;
            setScale("Pastel");
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
