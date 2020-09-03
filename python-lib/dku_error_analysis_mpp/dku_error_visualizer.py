# -*- coding: utf-8 -*-
import numpy as np
import pydotplus
import graphviz as gv
from sklearn import tree
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from dku_error_analysis_utils import ErrorAnalyzerConstants
from dku_error_analysis_mpp.error_analyzer import ErrorAnalyzer
from dku_error_analysis_mpp.dku_error_analyzer import DkuErrorAnalyzer
from dku_error_analysis_decision_tree.node import Node

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='Error Analysis Plugin | %(levelname)s - %(message)s')


class DkuErrorVisualizer(object):
    """
    ErrorVisualizer provides visual utilities to analyze the error classifier in ErrorAnalyzer and DkuErrorAnalyzer.
    """

    def __init__(self, error_analyzer):

        if not isinstance(error_analyzer, ErrorAnalyzer):
            raise NotImplementedError('you need to input an ErrorAnalyzer object.')

        self._error_analyzer = error_analyzer
        self._error_clf = self._error_analyzer.model_performance_predictor
        self._error_train_x = self._error_analyzer.error_train_x
        self._error_train_y = self._error_analyzer.error_train_y

        self._features_in_model_performance_predictor = self._error_analyzer.model_performance_predictor_features

        if self._features_in_model_performance_predictor is None:
            self._features_in_model_performance_predictor = list(range(self._error_clf.max_features_))

        if isinstance(error_analyzer, DkuErrorAnalyzer):
            if self._error_analyzer.tree is None:
                self._error_analyzer.parse_tree()

            self._tree = self._error_analyzer.tree
            self._tree_parser = self._error_analyzer.tree_parser

            self._features_dict = self._error_analyzer.features_dict

    def plot_error_tree(self, size=None):
        """ Plot the graph of the decision tree """

        return gv.Source(self._tree.to_dot_string())

    def read_feature(self, preprocessed_feature):
        """ Undo the preprocessing of feature names for categorical variables """
        if preprocessed_feature in self._tree_parser.preprocessed_feature_mapping:
            split_param = self._tree_parser.preprocessed_feature_mapping[preprocessed_feature]
            return split_param.feature, split_param.value
        else:
            return preprocessed_feature, None

    @staticmethod
    def plot_hist(data, bins, labels, colors, alpha, histtype='bar'):
        #n_samples = 0
        #for x in data:
        #    n_samples += x.shape[0]
        #weights = [np.ones_like(x, dtype=np.float) / n_samples for x in data]

        plt.hist(data, bins, label=labels, stacked=True, density=False,
                 alpha=alpha, color=colors, histtype=histtype)

    def rank_features_by_error_correlation(self, feature_names=None,
                                           top_k_features=ErrorAnalyzerConstants.TOP_K_FEATURES,
                                           include_non_split_features=False):
        """ Rank features by correlation with error """

        if feature_names is None:
            feature_names = list(range(self._error_clf.max_features_))

        sorted_feature_indices = np.argsort(-self._error_clf.feature_importances_)

        ranked_features = []
        for feature_idx in sorted_feature_indices:
            feature_importance = - self._error_clf.feature_importances_[feature_idx]
            if feature_importance != 0 or include_non_split_features:
                feature = feature_names[feature_idx]
                if feature not in ranked_features:
                    ranked_features.append(feature)
                    if len(ranked_features) == top_k_features:
                        return ranked_features
        return ranked_features

    def plot_error_node_feature_distribution(self, leaf_selector='all_errors', top_k_features=3, compare_to_global=True,
                                             show_class=False, figsize=(10, 5)):
        """ Return plot of error node feature distribution and compare to global baseline """

        leaf_nodes = self._error_analyzer.get_ranked_leaf_ids(leaf_selector=leaf_selector)
        for leaf in leaf_nodes:
            pass
        
        for unprocessed_name in feature_valid_values:
            feature_valid_values[unprocessed_name].append(ErrorAnalyzerConstants.CATEGORICAL_OTHERS)

        x_unprocessed_df = self._error_analyzer.error_df.loc[:,
                                                                self._error_analyzer.error_df.columns !=
                                                                ErrorAnalyzerConstants.ERROR_COLUMN]

        x, y = x_unprocessed_df[feature_names].values, self._error_train_y


        feature_idx_by_importance = [feature_names.index(feat_name) for feat_name in ranked_features]

        x_error_global = x[y == ErrorAnalyzerConstants.WRONG_PREDICTION, :]
        x_correct_global = x[y == ErrorAnalyzerConstants.CORRECT_PREDICTION, :]

        class_colors = [
            ErrorAnalyzerConstants.ERROR_TREE_COLORS[ErrorAnalyzerConstants.CORRECT_PREDICTION],
            ErrorAnalyzerConstants.ERROR_TREE_COLORS[ErrorAnalyzerConstants.WRONG_PREDICTION]]

        for leaf in leaf_nodes:
            values = self._error_clf.tree_.value[leaf, :]
            n_errors = values[0, error_class_idx]
            n_corrects = values[0, correct_class_idx]
            print('Node %d: (%d correct predictions, %d wrong predictions)' % (leaf, n_corrects, n_errors))
            node_indices = self._error_analyzer.train_leaf_ids == leaf
            y_node = y[node_indices]
            x_node = x[node_indices, :]
            x_error_node = x_node[y_node == ErrorAnalyzerConstants.WRONG_PREDICTION, :]
            x_correct_node = x_node[y_node == ErrorAnalyzerConstants.CORRECT_PREDICTION, :]

            for f_id in feature_idx_by_importance:
                plt.figure(figsize=figsize)
                if compare_to_global:
                    if show_class:
                        x_hist = [f_correct_global, f_error_global]
                        labels = ['global correct', 'global error']
                        colors = class_colors
                    else:
                        x_hist = [f_global]
                        labels = ['global']
                        # global is mainly correct: take color of correct prediction class
                        colors = [ErrorAnalyzerConstants.ERROR_TREE_COLORS[ErrorAnalyzerConstants.CORRECT_PREDICTION]]

                    DkuErrorVisualizer.plot_hist(x_hist, bins, labels, colors, alpha=0.5)

                if show_class:
                    x_hist = [f_correct_node, f_error_node]
                    labels = ['node correct', 'node error']
                    colors = class_colors
                else:
                    x_hist = [f_node]
                    labels = ['node']
                    decision = self._error_clf.tree_.value[leaf, :].argmax()
                    colors = [ErrorAnalyzerConstants.ERROR_TREE_COLORS[self._error_clf.classes_[decision]]]

                DkuErrorVisualizer.plot_hist(x_hist, bins, labels, colors, alpha=1.0)

                plt.xlabel(f_name)
                plt.ylabel('Proportion of samples')
                plt.legend()
                plt.title('Distribution of %s in Node %d: (%d, %d)' % (f_name, leaf, n_corrects, n_errors))
                plt.pause(0.05)


        plt.show()
