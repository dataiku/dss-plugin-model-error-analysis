# -*- coding: utf-8 -*-
import numpy as np
import pydotplus
import graphviz as gv
from sklearn import tree
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from dku_error_analysis_mpp.error_config import ErrorAnalyzerConstants
from dku_error_analysis_mpp.error_analyzer import ErrorAnalyzer
from dku_error_analysis_mpp.dku_error_analyzer import DkuErrorAnalyzer
from dku_error_analysis_decision_tree.node import Node
from dku_error_analysis_tree_parsing.depreprocessor import _denormalize_feature_value
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='Error Analysis Plugin | %(levelname)s - %(message)s')


class ErrorVisualizer(object):
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

        digraph_tree = tree.export_graphviz(self._error_clf,
                                            feature_names=self._features_in_model_performance_predictor,
                                            class_names=self._error_clf.classes_,
                                            node_ids=True,
                                            proportion=False,
                                            rotate=False,
                                            out_file=None,
                                            filled=True,
                                            rounded=True)

        pydot_graph = pydotplus.graph_from_dot_data(str(digraph_tree))

        nodes = pydot_graph.get_node_list()
        for node in nodes:
            if node.get_label():
                node_label = node.get_label()
                alpha = 0.0
                node_class = ErrorAnalyzerConstants.CORRECT_PREDICTION
                if 'value = [' in node_label:
                    values = [int(ii) for ii in node_label.split('value = [')[1].split(']')[0].split(',')]
                    values = [float(v) / sum(values) for v in values]
                    node_arg_class = np.argmax(values)
                    node_class = self._error_clf.classes_[node_arg_class]
                    # transparency as the entropy value
                    alpha = values[node_arg_class]

                class_color = ErrorAnalyzerConstants.ERROR_TREE_COLORS[node_class].strip('#')
                class_color_rgb = tuple(int(class_color[i:i + 2], 16) for i in (0, 2, 4))
                # compute the color as alpha against white
                color_rgb = [int(round(alpha * c + (1 - alpha) * 255, 0)) for c in class_color_rgb]
                color = '#{:02x}{:02x}{:02x}'.format(color_rgb[0], color_rgb[1], color_rgb[2])
                node.set_fillcolor(color)

                if isinstance(self._error_analyzer, DkuErrorAnalyzer):
                    # descale threshold value
                    if ' <= ' in node_label:
                        idx = int(node_label.split('node #')[1].split('\\n')[0])
                        lte_split = node_label.split(' <= ')
                        entropy_split = lte_split[1].split('\\nentropy')
                        left_child = self._tree.nodes[self._tree.nodes[idx].children_ids[0]]
                        if left_child.get_type() == Node.TYPES.NUM:
                            descaled_value = left_child.end
                            descaled_value = '%.2f' % descaled_value
                            lte_modified = ' <= '.join([lte_split[0], descaled_value])
                        else:
                            descaled_value = left_child.values[0]
                            lte_split_without_feature = lte_split[0].split('\\n')[0]
                            new_feature = left_child.feature
                            lte_split_with_new_feature = lte_split_without_feature + '\\n' + new_feature
                            lte_modified = ' != '.join([lte_split_with_new_feature, descaled_value])
                        new_label = '\\nentropy'.join([lte_modified, entropy_split[1]])
                        node.set_label(new_label)

        if size is not None:
            pydot_graph.set_size('"%d,%d!"' % (size[0], size[1]))
        gvz_graph = gv.Source(pydot_graph.to_string())

        return gvz_graph

    def read_feature(self, preprocessed_feature):
        """ Undo the preprocessing of feature names for categorical variables """
        if preprocessed_feature in self._tree_parser.preprocessed_feature_mapping:
            split_param = self._tree_parser.preprocessed_feature_mapping[preprocessed_feature]
            return split_param.feature, split_param.value
        else:
            return preprocessed_feature, None

    @staticmethod
    def plot_hist(data, bins, labels, colors, alpha, histtype='bar'):
        n_samples = 0
        for x in data:
            n_samples += x.shape[0]

        weights = [np.ones_like(x, dtype=np.float) / n_samples for x in data]
        plt.hist(data, bins, label=labels, stacked=True, density=False,
                 alpha=alpha, color=colors, weights=weights, histtype=histtype)

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

    def plot_error_node_feature_distribution(self, nodes='all_errors', top_k_features=3, compare_to_global=True,
                                             show_class=False, figsize=(10, 5)):
        """ Return plot of error node feature distribution and compare to global baseline """

        leaf_nodes = self._error_analyzer.get_ranked_leaf_ids(input_leaf_ids=nodes)

        error_class_idx = np.where(self._error_clf.classes_ == ErrorAnalyzerConstants.WRONG_PREDICTION)[0]
        correct_class_idx = np.where(self._error_clf.classes_ == ErrorAnalyzerConstants.CORRECT_PREDICTION)[0]

        feature_names = self._features_in_model_performance_predictor

        if isinstance(self._error_analyzer, DkuErrorAnalyzer):
            # cannot use self._tree.ranked_features as their number is always MAX_MOST_IMPORTANT_FEATURES(3)
            # while we want to use the input top_k_features
            ranked_features = self._tree_parser.rank_features_by_error_correlation(feature_names,
                                                                                   max_number_features=top_k_features,
                                                                                   include_non_split_features=True)

            feature_list_unprocessed = [self.read_feature(feature_name)[0] for feature_name in feature_names]
            seen = set()
            feature_names = [x for x in feature_list_unprocessed if not (x in seen or seen.add(x))]

            feature_valid_values = dict()
            for feature_name, param in self._tree_parser.preprocessed_feature_mapping.iteritems():
                unprocessed_name = self.read_feature(feature_name)[0]
                if unprocessed_name not in feature_valid_values:
                    feature_valid_values[unprocessed_name] = list()
                if len(param.value) == 1 and not isinstance(param.value[0], float):
                    feature_valid_values[unprocessed_name].append(param.value[0])

            for unprocessed_name in feature_valid_values:
                feature_valid_values[unprocessed_name].append(ErrorAnalyzerConstants.CATEGORICAL_OTHERS)

            x_unprocessed_df = self._error_analyzer.error_df.loc[:,
                                                                 self._error_analyzer.error_df.columns !=
                                                                 ErrorAnalyzerConstants.ERROR_COLUMN]

            x, y = x_unprocessed_df[feature_names].values, self._error_train_y

        else:
            ranked_features = self.rank_features_by_error_correlation(feature_names,
                                                                      top_k_features=top_k_features,
                                                                      include_non_split_features=True)

            x, y = self._error_train_x, self._error_train_y

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

                f_name = feature_names[f_id]

                f_global = x[:, f_id]
                f_node = x_node[:, f_id]

                f_correct_global = x_correct_global[:, f_id]
                f_error_global = x_error_global[:, f_id]
                f_correct_node = x_correct_node[:, f_id]
                f_error_node = x_error_node[:, f_id]

                f_values = np.unique(x[:, f_id])

                if isinstance(self._error_analyzer, DkuErrorAnalyzer):
                    if self._features_dict.get(f_name).get("type") != "NUMERIC":
                        labels = feature_valid_values[f_name]
                        n_labels = len(labels)
                        bins = np.linspace(0, n_labels - 1, n_labels)
                        plt.xticks(rotation=90)

                        map_invalid_values = np.vectorize(
                            lambda value: ErrorAnalyzerConstants.CATEGORICAL_OTHERS if value not in labels else value)
                        f_global = map_invalid_values(f_global)
                        f_node = map_invalid_values(f_node)
                        f_correct_global = map_invalid_values(f_correct_global)
                        f_error_global = map_invalid_values(f_error_global)
                        f_correct_node = map_invalid_values(f_correct_node)
                        f_error_node = map_invalid_values(f_error_node)
                    else:
                        bins = np.linspace(np.min(f_values), np.max(f_values))
                else:
                    bins = np.linspace(np.min(f_values), np.max(f_values))

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

                    ErrorVisualizer.plot_hist(x_hist, bins, labels, colors, alpha=0.5)

                if show_class:
                    x_hist = [f_correct_node, f_error_node]
                    labels = ['node correct', 'node error']
                    colors = class_colors
                else:
                    x_hist = [f_node]
                    labels = ['node']
                    decision = self._error_clf.tree_.value[leaf, :].argmax()
                    colors = [ErrorAnalyzerConstants.ERROR_TREE_COLORS[self._error_clf.classes_[decision]]]

                ErrorVisualizer.plot_hist(x_hist, bins, labels, colors, alpha=1.0)

                plt.xlabel(f_name)
                plt.ylabel('Proportion of samples')
                plt.legend()
                plt.title('Distribution of %s in Node %d: (%d, %d)' % (f_name, leaf, n_corrects, n_errors))
                plt.pause(0.05)

        plt.show()
