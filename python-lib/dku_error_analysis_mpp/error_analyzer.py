# -*- coding: utf-8 -*-
import numpy as np
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.base import is_regressor
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from dku_error_analysis_mpp.kneed import KneeLocator
from dku_error_analysis_utils import not_enough_data
from dku_error_analysis_mpp.error_config import ErrorAnalyzerConstants
from dku_error_analysis_mpp.metrics import mpp_report

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='Error Analysis Plugin | %(levelname)s - %(message)s')


class ErrorAnalyzer(object):
    """
    ErrorAnalyzer analyzes the errors of a prediction model on a test set.
    It uses model predictions and ground truth target to compute the model errors on the test set.
    It then trains a Decision Tree on the same test set by using the model error as target.
    The nodes of the decision tree are different segments of errors to be studied individually.
    """

    def __init__(self, predictor, seed=65537, feature_names=None):

        try:
            # TODO need to set the right attributes to check for each type of estimator
            check_is_fitted(predictor, ["coef_", "estimator_", "estimators_"], all_or_any=np.any)
        except NotFittedError:
            raise NotFittedError('you need to input a fitted model.')

        self._predictor = predictor
        self._is_regression = is_regressor(self._predictor)

        self._error_clf = None
        self._error_train_x = None
        self._error_train_y = None
        self._error_test_x = None
        self._error_test_y = None
        self._error_test_y_pred = None
        self._test_y = None
        self._features_in_model_performance_predictor = feature_names
        self._mpp_accuracy_score = None
        self._primary_model_predicted_accuracy = None
        self._primary_model_true_accuracy = None
        self._confidence_decision = None
        self._report = None

        self._tree = None
        self._tree_parser = None

        self._error_train_leaf_id = None
        self._ranked_error_nodes = None

        self._seed = seed

    @property
    def tree(self):
        return self._tree

    @property
    def error_train_x(self):
        return self._error_train_x

    @property
    def error_train_y(self):
        return self._error_train_y

    @property
    def model_performance_predictor_features(self):
        return self._features_in_model_performance_predictor

    @property
    def model_performance_predictor(self):
        return self._error_clf

    @property
    def leaf_ids(self):
        if self._error_train_leaf_id is None:
            self._error_train_leaf_id = self._compute_leaf_ids()
        return self._error_train_leaf_id

    @property
    def ranked_error_nodes(self):
        if self._ranked_error_nodes is None:
            self._ranked_error_nodes = self._compute_ranked_error_nodes()
        return self._ranked_error_nodes

    @property
    def mpp_accuracy_score(self):
        return self._mpp_accuracy_score

    @property
    def primary_model_predicted_accuracy(self):
        return self._primary_model_predicted_accuracy

    @property
    def primary_model_true_accuracy(self):
        return self._primary_model_true_accuracy

    @property
    def confidence_decision(self):
        return self._confidence_decision

    def fit(self, x, y):
        """
        Trains a Decision Tree to discriminate between samples that are correctly predicted or wrongly predicted
        (errors) by a primary model.
        """
        logger.info("Preparing the model performance predictor...")

        np.random.seed(self._seed)

        self._error_train_x = x
        self._error_train_y = self._compute_primary_model_error(x, y)

        logger.info("Fitting the model performance predictor...")

        # entropy/mutual information is used to split nodes in Microsoft Pandora system
        criterion = ErrorAnalyzerConstants.CRITERION

        dt_clf = tree.DecisionTreeClassifier(criterion=criterion, min_samples_leaf=1, random_state=self._seed)
        parameters = {'max_depth': ErrorAnalyzerConstants.MAX_DEPTH_GRID}
        gs_clf = GridSearchCV(dt_clf, parameters, cv=5)
        gs_clf.fit(self._error_train_x, self._error_train_y)
        self._error_clf = gs_clf.best_estimator_

        logger.info('Grid search selected max_depth = {}'.format(gs_clf.best_params_['max_depth']))

    def _compute_primary_model_error(self, x, y):
        """
        Computes the errors of the primary model predictions and samples with max n = ErrorAnalyzerConstants.MAX_NUM_ROW
        :return: an array with error target (correctly predicted vs wrongly predicted)
        """

        logger.info('Prepare data with model for model performance predictor')

        if not_enough_data(x, min_len=ErrorAnalyzerConstants.MIN_NUM_ROWS):
            raise ValueError(
                'The original dataset is too small ({} rows) to have stable result, it needs to have at least '
                '{} rows'.format(len(x), ErrorAnalyzerConstants.MIN_NUM_ROWS))

        logger.info("Rebalancing data:")
        number_of_rows = min(x.shape[0], ErrorAnalyzerConstants.MAX_NUM_ROW)
        logger.info(" - original dataset had %s rows. Selecting the first %s." % (x.shape[0], number_of_rows))

        x = x[:number_of_rows, :]

        y_pred = self._predictor.predict(x)

        error_y = self._get_errors(y, y_pred)

        return error_y

    def predict(self, x):
        """ Predict model performance on samples """
        return self._error_clf.predict(x)

    def _compute_model_performance_predictor_metrics(self, x_test, y_test):
        """
        Compute MPP ability to predict primary model performance.
        Ideally primary_model_predicted_accuracy should be equal
        to primary_model_true_accuracy
        """

        if (y_test == self._test_y).all() and (x_test == self._error_test_x).all():
            # performances already computed on this test set
            return

        self._error_test_x = x_test
        self._test_y = y_test

        self._error_test_y = self._compute_primary_model_error(x_test, y_test)

        y_true = self._error_test_y
        self._error_test_y_pred = self._error_clf.predict(self._error_test_x)
        y_pred = self._error_test_y_pred

        report_dict = mpp_report(y_true, y_pred, output_dict=True)

        self._mpp_accuracy_score = report_dict[ErrorAnalyzerConstants.MPP_ACCURACY]
        self._primary_model_true_accuracy = report_dict[ErrorAnalyzerConstants.PRIMARY_MODEL_TRUE_ACCURACY]
        self._primary_model_predicted_accuracy = report_dict[ErrorAnalyzerConstants.PRIMARY_MODEL_PREDICTED_ACCURACY]

        self._confidence_decision = report_dict[ErrorAnalyzerConstants.CONFIDENCE_DECISION]

    @staticmethod
    def _get_epsilon(difference, mode='rec'):
        """ Compute epsilon to define errors in regression task """
        assert (mode in ['std', 'rec'])
        if mode == 'std':
            std_diff = np.std(difference)
            mean_diff = np.mean(difference)
            epsilon = mean_diff + std_diff
        elif mode == 'rec':
            n_points = ErrorAnalyzerConstants.NUMBER_EPSILON_VALUES
            epsilon_range = np.linspace(min(difference), max(difference), num=n_points)
            cdf_error = np.zeros_like(epsilon_range)
            n_samples = difference.shape[0]
            for i, epsilon in enumerate(epsilon_range):
                correct = difference <= epsilon
                cdf_error[i] = float(np.count_nonzero(correct)) / n_samples
            kneedle = KneeLocator(epsilon_range, cdf_error)
            epsilon = kneedle.knee
        return epsilon

    def _get_errors(self, y, y_pred):
        """ Compute errors of the primary model on the test set """
        if self._is_regression:

            difference = np.abs(y - y_pred)

            epsilon = ErrorAnalyzer._get_epsilon(difference, mode='rec')

            error = difference > epsilon
        else:

            error = (y != y_pred)

        error = list(error)
        transdict = {True: ErrorAnalyzerConstants.WRONG_PREDICTION, False: ErrorAnalyzerConstants.CORRECT_PREDICTION}
        error = np.array([transdict[elem] for elem in error], dtype=object)

        return error

    def _compute_leaf_ids(self):
        """ Compute indices of leaf nodes for the train set """
        return self._error_clf.apply(self._error_train_x)

    def _compute_ranked_error_nodes(self):
        """ Select error nodes and rank them by importance, defined as: (n_errors - n_correct) - impurity.
            Specifically error nodes with the same difference (n_errors - n_correct) are clustered together and
            sorted by increasing impurity. For instance nodes [n_correct, n_errors]=[0, 7] is ranked before [0, 6] and
            [0, 5], while [0, 5] is ranked after [0, 6] and [1, 7]: [0, 7], [0, 6], [1, 7], [0, 5]."""
        error_leaf_nodes = []
        error_leaf_nodes_importance = []
        leaf_ids = self._error_train_leaf_id
        leaf_nodes = np.unique(leaf_ids)
        error_class_idx = np.where(self._error_clf.classes_ == ErrorAnalyzerConstants.WRONG_PREDICTION)[0]
        correct_class_idx = np.where(self._error_clf.classes_ == ErrorAnalyzerConstants.CORRECT_PREDICTION)[0]
        for leaf in leaf_nodes:
            decision = self._error_clf.tree_.value[leaf, :].argmax()
            if self._error_clf.classes_[decision] == ErrorAnalyzerConstants.WRONG_PREDICTION:
                error_leaf_nodes.append(leaf)
                values = self._error_clf.tree_.value[leaf, :]
                n_errors = values[0, error_class_idx]
                n_corrects = values[0, correct_class_idx]
                leaf_impurity = float(n_corrects) / (n_errors + n_corrects)
                n_difference = n_corrects - n_errors  # always negative
                error_leaf_nodes_importance.append(n_difference + leaf_impurity)
        ranked_error_nodes = [x for _, x in sorted(zip(error_leaf_nodes_importance, error_leaf_nodes))]

        return ranked_error_nodes

    def get_list_of_nodes(self, nodes):
        """ Parse input string and provide the desired nodes indices """
        if not (isinstance(nodes, list) or isinstance(nodes, int)):
            if nodes not in ['all', 'all_errors']:
                raise ValueError('The input nodes value is invalid. '
                                 'Nodes should be a node index, a list of node indices, '
                                 '"all" to show all nodes or "all_errors" to show all error nodes.')

        if isinstance(nodes, int):
            nodes = [nodes]

        leaf_ids = self.leaf_ids
        leaf_nodes = np.unique(leaf_ids)
        if nodes is not 'all':
            if nodes is 'all_errors':
                leaf_nodes = self.ranked_error_nodes
            else:
                leaf_nodes = set(nodes) & set(leaf_nodes)
                if not bool(leaf_nodes):
                    print("Selected nodes are not leaf nodes.")
                    return

        return leaf_nodes

    def get_path_to_node(self, node_id):
        """ Return path to node as a list of split steps from the nodes of the sklearn Tree object """
        feature_names = self._features_in_model_performance_predictor

        children_left = self._error_clf.tree_.children_left
        children_right = self._error_clf.tree_.children_right
        feature = self._error_clf.tree_.feature
        threshold = self._error_clf.tree_.threshold

        cur_node_id = node_id
        path_to_node = []
        while cur_node_id > 0:
            if cur_node_id in children_left:
                sign = ' <= '
                parent_id = list(children_left).index(cur_node_id)
            else:
                sign = " > "
                parent_id = list(children_right).index(cur_node_id)

            feat = feature[parent_id]
            thresh = threshold[parent_id]

            step = str(feature_names[feat]) + sign + ("%.2f" % thresh)
            path_to_node.append(step)
            cur_node_id = parent_id
        path_to_node = path_to_node[::-1]

        return path_to_node

    def error_node_summary(self, nodes='all_errors', print_path_to_node=True):
        """ Return summary information regarding input nodes """

        leaf_nodes = self.get_list_of_nodes(nodes)

        y = self._error_train_y
        n_total_errors = y[y == ErrorAnalyzerConstants.WRONG_PREDICTION].shape[0]
        error_class_idx = np.where(self._error_clf.classes_ == ErrorAnalyzerConstants.WRONG_PREDICTION)[0]
        correct_class_idx = np.where(self._error_clf.classes_ == ErrorAnalyzerConstants.CORRECT_PREDICTION)[0]
        for leaf in leaf_nodes:
            values = self._error_clf.tree_.value[leaf, :]
            n_errors = values[0, error_class_idx]
            n_corrects = values[0, correct_class_idx]
            print('Node %d: (%d correct predictions, %d wrong predictions)' % (leaf, n_corrects, n_errors))
            print(' Local error (Purity): %.2f' % (float(n_errors) / (n_corrects + n_errors)))
            print(' Global error: %.2f' % (float(n_errors) / n_total_errors))
            if print_path_to_node:
                print(' Path to node:')
                path_to_node = self.get_path_to_node(leaf)
                for step in path_to_node:
                    print('     ' + step)

    def mpp_summary(self, x_test, y_test, output_dict=False):
        """ Print ErrorAnalyzer summary metrics """

        self._compute_model_performance_predictor_metrics(x_test, y_test)

        y_true = self._error_test_y
        y_pred = self._error_test_y_pred

        return mpp_report(y_true, y_pred, output_dict)






