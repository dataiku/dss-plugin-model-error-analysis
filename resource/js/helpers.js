'use strict';
app.service("Format", function() {
    return {
        noBreakingSpace: "\xa0",
        ellipsis: function(text, length) {
            text = text.toString();
            if (text.length > length) {
                return (text.substr(0, length-3) + "...");
            }
            return text;
        },
        toFixedIfNeeded: function(number, decimals) {
            if(Math.round(number) !== number) {
                return parseFloat(number.toFixed(decimals));
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

app.service("ModalService", function($compile, $http) {
    const DEFAULT_MODAL_TEMPLATE = "/plugins/model-error-analysis/resource/templates/modal.html";

    function create(scope, config, templateUrl=DEFAULT_MODAL_TEMPLATE) {
        $http.get(templateUrl).then(function(response) {
            const template = response.data;
            const newScope = scope.$new();
            const element = $compile(template)(newScope);

            angular.extend(newScope, config);

            newScope.close = function(event) {
                if (event && !event.target.className.includes("modal-background")) return;
                element.remove();
                newScope.$emit("closeModal");
            };

            if (newScope.promptConfig && newScope.promptConfig.conditions) {
                const inputField = element.find("input");
                for (const attr in newScope.promptConfig.conditions) {
                    inputField.attr(attr, newScope.promptConfig.conditions[attr]);
                }
                $compile(inputField)(newScope);
            }

            angular.element("body").append(element);
            element.focus();
        });
    };
    return {
        createBackendErrorModal: function(scope, errorMsg) {
            create(scope, {
                title: 'Backend error',
                msgConfig: { error: true, msg: errorMsg }
            }, DEFAULT_MODAL_TEMPLATE);
        },
        create
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

app.filter('pluralize', function() {
    return function(word, number) {
        return number === 1 ? word : (word + 's');
    };
});
