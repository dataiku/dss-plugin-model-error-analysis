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
from dku_error_analysis_mpp.error_visualizer import ErrorVisualizer
from dku_error_analysis_mpp.metrics import mpp_report

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='Error Analysis Plugin | %(levelname)s - %(message)s')


class ErrorAnalyzer:
    """
    ErrorAnalyzer analyzes the errors of a prediction model on a test set.
    It uses model predictions and ground truth target to compute the model errors on the test set.
    It then trains a Decision Tree on the same test set by using the model error as target.
    The nodes of the decision tree are different segments of errors to be studied individually.
    """

    def __init__(self, predictor, seed=65537):

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
        self._features_in_model_performance_predictor = None
        self._mpp_accuracy_score = None
        self._primary_model_predicted_accuracy = None
        self._primary_model_true_accuracy = None
        self._confidence_decision = None
        self._report = None

        self._tree = None
        self._tree_parser = None

        self._error_visualizer = None

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

    def mpp_accuracy_score(self, x_test, y_test):
        self.check_computed_metrics(self._mpp_accuracy_score, x_test, y_test)
        return self._mpp_accuracy_score

    def primary_model_predicted_accuracy(self, x_test, y_test):
        self.check_computed_metrics(self._primary_model_predicted_accuracy, x_test, y_test)
        return self._primary_model_predicted_accuracy

    def primary_model_true_accuracy(self, x_test, y_test):
        self.check_computed_metrics(self._primary_model_true_accuracy, x_test, y_test)
        return self._primary_model_true_accuracy

    def confidence_decision(self, x_test, y_test):
        self.check_computed_metrics(self._confidence_decision, x_test, y_test)
        return self._confidence_decision

    def check_computed_metrics(self, attribute, x_test, y_test):
        if attribute is None:
            self.compute_model_performance_predictor_metrics(x_test, y_test)

    def fit(self, x, y):
        """
        Trains a Decision Tree to discriminate between samples that are correctly predicted or wrongly predicted
        (errors) by a primary model.
        """
        logger.info("Preparing the model performance predictor...")

        np.random.seed(self._seed)

        self._error_train_x = x
        self._error_train_y = self.prepare_data_for_model_performance_predictor(x, y)

        logger.info("Fitting the model performance predictor...")

        # entropy/mutual information is used to split nodes in Microsoft Pandora system
        criterion = ErrorAnalyzerConstants.CRITERION

        dt_clf = tree.DecisionTreeClassifier(criterion=criterion, min_samples_leaf=1, random_state=self._seed)
        parameters = {'max_depth': ErrorAnalyzerConstants.MAX_DEPTH_GRID}
        gs_clf = GridSearchCV(dt_clf, parameters, cv=5)
        gs_clf.fit(self._error_train_x, self._error_train_y)
        self._error_clf = gs_clf.best_estimator_

        logger.info('Grid search selected max_depth = {}'.format(gs_clf.best_params_['max_depth']))

    def prepare_data_for_model_performance_predictor(self, x, y):
        """
        Computes the errors of the primary model predictions and samples with max n = ErrorAnalyzerConstants.MAX_NUM_ROW
        :return: a dataframe with error target (correctly predicted vs wrongly predicted)
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
        return self._error_clf.predict(x)

    def compute_model_performance_predictor_metrics(self, x_test, y_test):
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

        self._error_test_y = self.prepare_data_for_model_performance_predictor(x_test, y_test)

        y_true = self._error_test_y
        self._error_test_y_pred = self._error_clf.predict(self._error_test_x)
        y_pred = self._error_test_y_pred

        report_dict = mpp_report(y_true, y_pred, output_dict=True)

        self._mpp_accuracy_score = report_dict['mpp_accuracy_score']
        self._primary_model_true_accuracy = report_dict['primary_model_true_accuracy']
        self._primary_model_predicted_accuracy = report_dict['primary_model_predicted_accuracy']

        self._confidence_decision = report_dict['confidence_decision']

    def _get_epsilon(self, difference, mode='rec'):
        """ compute epsilon to define errors in regression task """
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
        """ compute errors of the primary model on the test set """
        if self._is_regression:

            difference = np.abs(y - y_pred)

            epsilon = self._get_epsilon(difference, mode='rec')

            error = difference > epsilon
        else:

            error = (y != y_pred)

        error = list(error)
        transdict = {True: ErrorAnalyzerConstants.WRONG_PREDICTION, False: ErrorAnalyzerConstants.CORRECT_PREDICTION}
        error = np.array([transdict[elem] for elem in error], dtype=object)

        return error

    def plot_error_tree(self, size=None, feature_names=None):

        if self._error_visualizer is None:
            self._error_visualizer = ErrorVisualizer(self._error_clf, self._error_train_x, self._error_train_y)

        return self._error_visualizer.plot_error_tree(size, feature_names)

    def plot_error_node_feature_distribution(self, nodes='all_errors', top_k_features=3, compare_to_global=True,
                                             show_class=False, figsize=(10, 5), feature_names=None):
        """ return plot of error node feature distribution and compare to global baseline """

        if self._error_visualizer is None:
            self._error_visualizer = ErrorVisualizer(self._error_clf, self._error_train_x, self._error_train_y)

        self._error_visualizer.plot_error_node_feature_distribution(nodes, top_k_features, compare_to_global,
                                                                    show_class, figsize, feature_names)

    def error_node_summary(self, nodes='all_errors', feature_names=None):
        """ return summary information regarding input nodes """
        if self._error_visualizer is None:
            self._error_visualizer = ErrorVisualizer(self._error_clf, self._error_train_x, self._error_train_y)

        self._error_visualizer.error_node_summary(nodes, feature_names=feature_names)

    def mpp_summary(self, x_test, y_test, output_dict=False):
        """ print ErrorAnalyzer summary metrics """

        self.check_computed_metrics(self._error_test_y_pred, x_test, y_test)
        y_true = self._error_test_y
        y_pred = self._error_test_y_pred

        return mpp_report(y_true, y_pred, output_dict)






