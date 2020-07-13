# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from dku_error_analysis_mpp.error_config import ErrorAnalyzerConstants
from dku_error_analysis_tree_parsing.tree_parser import TreeParser
from dku_error_analysis_mpp.error_analyzer import ErrorAnalyzer
from dku_error_analysis_mpp.dku_error_visualizer import DkuErrorVisualizer
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='Error Analysis Plugin | %(levelname)s - %(message)s')


class DkuErrorAnalyzer(object):
    """
    DkuErrorAnalyzer analyzes the errors of a DSS prediction model on its test set.
    It uses model predictions and ground truth target to compute the model errors on the test set.
    It then trains a Decision Tree on the same test set by using the model error as target.
    The nodes of the decision tree are different segments of errors to be studied individually.
    """

    def __init__(self, model_accessor, seed=65537):

        if model_accessor is None:
            raise NotImplementedError('you need to define a model accessor.')

        self._model_accessor = model_accessor
        self._target = self._model_accessor.get_target_variable()
        self._is_regression = self._model_accessor.is_regression()
        self._predictor = self._model_accessor.get_predictor()

        self._error_df = None

        self._seed = seed

        self._error_analyzer = ErrorAnalyzer(self._model_accessor.get_clf(), self._seed)

        self._train_x = None
        self._test_x = None
        self._train_y = None
        self._test_y = None

        self._train_x_df = None
        self._test_x_df = None
        self._train_y_df = None
        self._test_y_df = None

        self._error_df = None
        self._error_clf = None
        self._features_in_model_performance_predictor = None
        self._tree = None
        self._tree_parser = None

        self._dku_error_visualizer = None

    @property
    def tree(self):
        if self._tree_parser is None or self._tree is None:
            self.parse_tree()
        return self._tree

    @property
    def model_performance_predictor_features(self):
        return self._features_in_model_performance_predictor

    @property
    def model_performance_predictor(self):
        return self._error_analyzer.model_performance_predictor

    @property
    def mpp_accuracy_score(self):
        return self._error_analyzer.mpp_accuracy_score(self._test_x, self._test_y)

    @property
    def primary_model_predicted_accuracy(self):
        return self._error_analyzer.primary_model_predicted_accuracy(self._test_x, self._test_y)

    @property
    def primary_model_true_accuracy(self):
        return self._error_analyzer.primary_model_true_accuracy(self._test_x, self._test_y)

    @property
    def confidence_decision(self):
        return self._error_analyzer.confidence_decision(self._test_x, self._test_y)

    def fit(self):
        """
        Trains a Decision Tree to discriminate between samples that are correctly predicted or wrongly predicted
        (errors) by a primary model.
        """

        np.random.seed(self._seed)

        original_df = self._model_accessor.get_original_test_df(ErrorAnalyzerConstants.MAX_NUM_ROW)

        if self._target not in original_df:
            raise ValueError('The original dataset does not contain target "{}".'.format(self._target))

        x_df = original_df.drop(self._target, axis=1)
        y_df = original_df[self._target]

        self._train_x_df, self._test_x_df, self._train_y_df, self._test_y_df = train_test_split(
            x_df, y_df, test_size=ErrorAnalyzerConstants.TEST_SIZE
        )

        train_df = pd.concat([self._train_x_df, self._train_y_df], axis=1)
        test_df = pd.concat([self._test_x_df, self._test_y_df], axis=1)

        self._train_x, _, _, _, _ = self._predictor.preprocessing.preprocess(
            train_df,
            with_target=True,
            with_sample_weights=True)

        self._test_x, _, _, _, _ = self._predictor.preprocessing.preprocess(
            test_df,
            with_target=True,
            with_sample_weights=True)

        self._train_y = np.array(self._train_y_df)
        self._test_y = np.array(self._test_y_df)

        self._error_analyzer.fit(self._train_x, self._train_y)

        self._error_clf = self._error_analyzer.model_performance_predictor

        self._features_in_model_performance_predictor = self._predictor.get_features()

    def parse_tree(self):
        modified_length = len(self._error_analyzer.error_train_x)
        self._error_df = self._train_x_df.head(modified_length)
        self._error_df.loc[:, ErrorAnalyzerConstants.ERROR_COLUMN] = self._error_analyzer.error_train_y

        self._tree_parser = TreeParser(self._model_accessor.model_handler, self._error_clf)
        self._tree = self._tree_parser.build_tree(self._error_df, self._features_in_model_performance_predictor)
        self._tree.parse_nodes(self._tree_parser,
                               self._features_in_model_performance_predictor,
                               self._error_analyzer.error_train_x)

    def prepare_error_visualizer(self):

        if self._tree_parser is None:
            self.parse_tree()

        self._dku_error_visualizer = DkuErrorVisualizer(error_clf=self._error_clf,
                                                        error_train_x=self._error_analyzer.error_train_x,
                                                        error_train_y=self._error_analyzer.error_train_y,
                                                        features_in_mpp=self._features_in_model_performance_predictor,
                                                        tree=self._tree,
                                                        tree_parser=self._tree_parser,
                                                        features_dict=self._model_accessor.get_per_feature())

    def plot_error_tree(self, size=None):

        if self._dku_error_visualizer is None:
            self.prepare_error_visualizer()

        return self._dku_error_visualizer.plot_error_tree(size)

    def plot_error_node_feature_distribution(self, nodes='all_errors', top_k_features=3, compare_to_global=True,
                                             show_class=False, figsize=(10, 5)):
        """ return plot of error node feature distribution and compare to global baseline """

        if self._dku_error_visualizer is None:
            self.prepare_error_visualizer()

        self._dku_error_visualizer.plot_error_node_feature_distribution(nodes, top_k_features, compare_to_global,
                                                                        show_class, figsize)

    def error_node_summary(self, nodes='all_errors'):
        """ return summary information regarding input nodes """
        if self._dku_error_visualizer is None:
            self.prepare_error_visualizer()

        self._dku_error_visualizer.error_node_summary(nodes)

    def mpp_summary(self, output_dict=False):
        """ print ErrorAnalyzer summary metrics """

        return self._error_analyzer.mpp_summary(self._test_x, self._test_y, output_dict)
