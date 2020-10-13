# -*- coding: utf-8 -*-
import numpy as np
import collections
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.base import is_regressor
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from dku_error_analysis_mpp.kneed import KneeLocator
from dku_error_analysis_utils import check_enough_data, ErrorAnalyzerConstants
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

    def __init__(self, predictor, feature_names=None, seed=65537):
        self._predictor = predictor
        self._is_regression = is_regressor(self._predictor)

        self._error_clf = None
        self._error_train_x = None
        self._error_train_y = None
        self._features_in_model_performance_predictor = feature_names

        self._error_train_leaf_id = None
        self._leaf_ids = None

        self._impurity = None
        self._quantized_impurity = None
        self._difference = None

        self._seed = seed

    @property
    def error_train_x(self):
        return self._error_train_x

    @property
    def error_train_y(self):
        return self._error_train_y

    @property
    def model_performance_predictor_features(self):
        if self._features_in_model_performance_predictor is None:
            self._features_in_model_performance_predictor = ["feature#%s" % feature_index
                                                             for feature_index in range(self._error_clf.n_features_)]

        return self._features_in_model_performance_predictor

    @property
    def model_performance_predictor(self):
        if self._error_clf is None:
            raise NotFittedError("You should fit a model performance predictor first")
        return self._error_clf

    @property
    def train_leaf_ids(self):
        if self._error_train_leaf_id is None:
            self._compute_train_leaf_ids()
        return self._error_train_leaf_id

    @property
    def impurity(self):
        if self._impurity is None:
            self._compute_ranking_arrays()
        return self._impurity

    @property
    def quantized_impurity(self):
        if self._quantized_impurity is None:
            self._compute_ranking_arrays()
        return self._quantized_impurity

    @property
    def difference(self):
        if self._difference is None:
            self._compute_ranking_arrays()
        return self._difference

    @property
    def leaf_ids(self):
        if self._leaf_ids is None:
            self._compute_leaf_ids()
        return self._leaf_ids

    def fit(self, x, y, max_nr_rows=ErrorAnalyzerConstants.MAX_NUM_ROW):
        """
        Trains a Decision Tree to discriminate between samples that are correctly predicted or wrongly predicted
        (errors) by a primary model.

        x must be a numpy array, ie. df[features] won't work, it must be df[features].values
        """
        logger.info("Preparing the model performance predictor...")

        np.random.seed(self._seed)

        self._error_train_x, self._error_train_y = self._compute_primary_model_error(x, y, max_nr_rows)

        logger.info("Fitting the model performance predictor...")

        # entropy/mutual information is used to split nodes in Microsoft Pandora system
        criterion = ErrorAnalyzerConstants.CRITERION

        dt_clf = tree.DecisionTreeClassifier(criterion=criterion, min_samples_leaf=1, random_state=self._seed)
        parameters = {'max_depth': ErrorAnalyzerConstants.MAX_DEPTH_GRID}
        gs_clf = GridSearchCV(dt_clf, parameters, cv=5)
        gs_clf.fit(self._error_train_x, self._error_train_y)
        self._error_clf = gs_clf.best_estimator_

        logger.info('Grid search selected max_depth = {}'.format(gs_clf.best_params_['max_depth']))

    def _compute_primary_model_error(self, x, y, max_nr_rows):
        """
        Computes the errors of the primary model predictions and samples
        :return: an array with error target (correctly predicted vs wrongly predicted)
        """

        logger.info('Prepare data with model for model performance predictor')

        check_enough_data(x, min_len=ErrorAnalyzerConstants.MIN_NUM_ROWS)

        if x.shape[0] > max_nr_rows:
            logger.info("Rebalancing data: original dataset had {} rows, selecting the first {}.".format(x.shape[0], max_nr_rows))

            x = x[:max_nr_rows, :]
            y = y[:max_nr_rows]

        y_pred = self._predictor.predict(x)

        error_y = self._get_errors(y, y_pred)

        return x, error_y

    def predict(self, x):
        """ Predict model performance on samples """
        return self.model_performance_predictor.predict(x)

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

    def _compute_train_leaf_ids(self):
        """ Compute indices of leaf nodes for the train set """
        self._error_train_leaf_id = self.model_performance_predictor.apply(self._error_train_x)

    def _compute_leaf_ids(self):
        """ Compute indices of leaf nodes """
        self._leaf_ids = np.where(self.model_performance_predictor.tree_.feature < 0)[0]

    def _get_error_leaves(self):
        error_class_idx = np.where(self.model_performance_predictor.classes_ == ErrorAnalyzerConstants.WRONG_PREDICTION)[0][0]
        error_node_ids = np.where(self.model_performance_predictor.tree_.value[:, 0, :].argmax(axis=1) == error_class_idx)[0]
        return np.in1d(self.leaf_ids, error_node_ids)

    def _compute_ranking_arrays(self, n_purity_levels=ErrorAnalyzerConstants.NUMBER_PURITY_LEVELS):
        """ Compute ranking array """
        error_class_idx = np.where(self.model_performance_predictor.classes_ == ErrorAnalyzerConstants.WRONG_PREDICTION)[0][0]
        correct_class_idx = 1 - error_class_idx

        wrongly_predicted_samples = self.model_performance_predictor.tree_.value[self.leaf_ids, 0, error_class_idx]
        correctly_predicted_samples = self.model_performance_predictor.tree_.value[self.leaf_ids, 0, correct_class_idx]

        self._impurity = correctly_predicted_samples / (wrongly_predicted_samples + correctly_predicted_samples)

        purity_bins = np.linspace(0, 1., n_purity_levels)
        self._quantized_impurity = np.digitize(self._impurity, purity_bins)
        self._difference = correctly_predicted_samples - wrongly_predicted_samples  # only negative numbers

    def get_ranked_leaf_ids(self, leaf_selector, rank_by='purity'):
        """ Select error nodes and rank them by importance."""
        apply_leaf_selector = self._get_leaf_selector(leaf_selector)
        selected_leaves = apply_leaf_selector(self.leaf_ids)
        if selected_leaves.size == 0:
            return selected_leaves
        if rank_by == 'purity':
            sorted_ids = np.lexsort((apply_leaf_selector(self.difference), apply_leaf_selector(self.quantized_impurity)))
        elif rank_by == 'class_difference':
            sorted_ids = np.lexsort((apply_leaf_selector(self.impurity), apply_leaf_selector(self.difference)))
        else:
            raise NotImplementedError("Input argument 'rank_by' is invalid. Should be 'purity' or 'class_difference'")
        return selected_leaves.take(sorted_ids)

    def _get_leaf_selector(self, leaf_selector):
        """
            Return a function that select rows of provided arrays. Arrays must be of shape (1, number of leaves)
            Args:
                leaf_selector: int, str, or array-like
                How to select the rows of the array
                  * int: Only keep the row corresponding to this leaf id
                  * array-like: Only keep the rows corresponding to these leaf ids
                  * str:
                    - "all": Keep the whole array
                    - "all_errors": Keep the rows with indices corresponding to the leaf ids classifying the primary model prediction as wrong

            Return:
                A function with one argument array as a selector of leaf ids
                Args:
                    array: numpy array of shape (1, number of leaves)
                    An array of which we only want to keep some rows
        """
        if isinstance(leaf_selector, str):
            if leaf_selector == "all":
                return lambda array: array
            if leaf_selector == "all_errors":
                return lambda array: array[self._get_error_leaves()]

        leaf_selector_as_array = np.array(leaf_selector)
        leaf_selector = np.in1d(self.leaf_ids, leaf_selector_as_array)
        nr_kept_leaves = np.count_nonzero(leaf_selector)
        if nr_kept_leaves == 0:
            print("None of the ids provided correspond to a leaf id.")
        elif nr_kept_leaves < leaf_selector_as_array.size:
            print("Some of the ids provided do not belong to leaves. Only leaf ids are kept.")
        return lambda array: array[leaf_selector]

    def _get_path_to_node(self, node_id):
        """ Return path to node as a list of split steps from the nodes of the sklearn Tree object """
        feature_names = self.model_performance_predictor_features

        children_left = self.model_performance_predictor.tree_.children_left
        children_right = self.model_performance_predictor.tree_.children_right
        feature = self.model_performance_predictor.tree_.feature
        threshold = self.model_performance_predictor.tree_.threshold

        cur_node_id = node_id
        path_to_node = collections.deque()
        while cur_node_id > 0:
            decision_rule = ''
            if cur_node_id in children_left:
                decision_rule += ' <= '
                parent_id = list(children_left).index(cur_node_id)
            else:
                decision_rule += " > "
                parent_id = list(children_right).index(cur_node_id)

            feat = feature[parent_id]
            thresh = threshold[parent_id]

            decision_rule = str(feature_names[feat]) + decision_rule + ("%.2f" % thresh)
            path_to_node.appendleft(decision_rule)
            cur_node_id = parent_id

        return path_to_node

    #TODO: rewrite this method using the ranking arrays
    def error_node_summary(self, leaf_selector='all_errors', add_path_to_leaves=False, print_summary=False):
        """ Return summary information regarding input nodes """

        leaf_nodes = self.get_ranked_leaf_ids(leaf_selector=leaf_selector)

        y = self._error_train_y
        n_total_errors = y[y == ErrorAnalyzerConstants.WRONG_PREDICTION].shape[0]
        error_class_idx = np.where(self.model_performance_predictor.classes_ == ErrorAnalyzerConstants.WRONG_PREDICTION)[0][0]
        correct_class_idx = 1 - error_class_idx

        leaves_summary = []
        for leaf_id in leaf_nodes:
            values = self.model_performance_predictor.tree_.value[leaf_id, :]
            n_errors = values[0, error_class_idx]
            n_corrects = values[0, correct_class_idx]
            local_error = float(n_errors) / (n_corrects + n_errors)
            global_error = float(n_errors) / n_total_errors

            leaf_dict = {
                "id": leaf_id,
                "n_corrects": n_corrects,
                "n_errors": n_errors,
                "local_error": local_error,
                "global_error": global_error
            }

            leaves_summary.append(leaf_dict)

            if add_path_to_leaves:
                path_to_node = self._get_path_to_node(leaf_id)
                leaf_dict["path_to_leaf"] = path_to_node

            if print_summary:
                print("LEAF %d:" % leaf_id)
                print("     Correct predictions: %d | Wrong predictions: %d | "
                      "Local error (purity): %.2f | Global error: %.2f" %
                      (n_corrects, n_errors, local_error, global_error))

                if add_path_to_leaves:
                    print('     Path to leaf:')
                    for (step_idx, step) in enumerate(path_to_node):
                        print('     ' + '   ' * step_idx + step)

        return leaves_summary

    def mpp_summary(self, x_test, y_test, nr_max_rows=ErrorAnalyzerConstants.MAX_NUM_ROW, output_dict=False):
        """ Print ErrorAnalyzer summary metrics """
        x_test, y_true = self._compute_primary_model_error(x_test, y_test, nr_max_rows)
        y_pred = self.model_performance_predictor.predict(x_test)
        return mpp_report(y_true, y_pred, output_dict)
