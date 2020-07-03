# -*- coding: utf-8 -*-
import numpy as np
import pydotplus
import graphviz as gv
from sklearn import tree
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from dku_error_analysis_mpp.error_config import ErrorAnalyzerConstants
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='Error Analysis Plugin | %(levelname)s - %(message)s')


class ErrorVisualizer:
    """
    ErrorVisualizer provides visual and command line utilities to analyze the error classifier in ErrorAnalyzer and
    DkuErrorAnalyzer.
    """

    def __init__(self, error_clf, error_train_x, error_train_y):

        self._error_clf = error_clf
        self._error_train_x = error_train_x
        self._error_train_y = error_train_y
        self._error_train_leaf_id = self.compute_leaf_ids()
        self._ranked_error_nodes = self.compute_ranked_error_nodes()

    @property
    def leaf_ids(self):
        return self._error_train_leaf_id

    @property
    def ranked_error_nodes(self):
        return self._ranked_error_nodes

    def plot_error_tree(self, size=None, feature_names=None):

        digraph_tree = tree.export_graphviz(self._error_clf,
                                            feature_names=feature_names,
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

        if size is not None:
            pydot_graph.set_size('"%d,%d!"' % (size[0], size[1]))
        gvz_graph = gv.Source(pydot_graph.to_string())

        return gvz_graph

    def compute_leaf_ids(self):
        return self._error_clf.apply(self._error_train_x)

    def compute_ranked_error_nodes(self):
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

        if not (isinstance(nodes, list) or isinstance(nodes, int)):
            assert (nodes in ['all', 'all_errors'])

        if isinstance(nodes, int):
            nodes = [nodes]

        leaf_ids = self._error_train_leaf_id
        leaf_nodes = np.unique(leaf_ids)
        if nodes is not 'all':
            if nodes is 'all_errors':
                leaf_nodes = self._ranked_error_nodes
            else:
                leaf_nodes = set(nodes) & set(leaf_nodes)
                if not bool(leaf_nodes):
                    print("Selected nodes are not leaf nodes.")
                    return

        return leaf_nodes

    def plot_hist(self, data, bins, labels, colors, alpha, histtype='bar'):
        n_samples = 0
        for x in data:
            n_samples += x.shape[0]

        weights = [np.ones_like(x, dtype=np.float) / n_samples for x in data]
        plt.hist(data, bins, label=labels, stacked=True, density=False,
                 alpha=alpha, color=colors, weights=weights, histtype=histtype)

    def rank_features_by_error_correlation(self, feature_names=None,
                                           top_k_features=ErrorAnalyzerConstants.TOP_K_FEATURES,
                                           include_non_split_features=False):

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
                                             show_class=False, figsize=(10, 5), feature_names=None):
        """ return plot of error node feature distribution and compare to global baseline """

        if feature_names is None:
            feature_names = list(range(self._error_clf.max_features_))

        leaf_ids = self._error_train_leaf_id
        leaf_nodes = self.get_list_of_nodes(nodes)

        error_class_idx = np.where(self._error_clf.classes_ == ErrorAnalyzerConstants.WRONG_PREDICTION)[0]
        correct_class_idx = np.where(self._error_clf.classes_ == ErrorAnalyzerConstants.CORRECT_PREDICTION)[0]

        ranked_features = self.rank_features_by_error_correlation(feature_names,
                                                                  top_k_features=top_k_features,
                                                                  include_non_split_features=True)

        feature_idx_by_importance = [feature_names.index(feat_name) for feat_name in ranked_features]

        x, y = self._error_train_x, self._error_train_y

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
            node_indices = leaf_ids == leaf
            y_node = y[node_indices]
            x_node = x[node_indices, :]
            x_error_node = x_node[y_node == ErrorAnalyzerConstants.WRONG_PREDICTION, :]
            x_correct_node = x_node[y_node == ErrorAnalyzerConstants.CORRECT_PREDICTION, :]

            for f_id in feature_idx_by_importance:

                plt.figure(figsize=figsize)

                f_name = feature_names[f_id]

                print(f_name)

                f_global = x[:, f_id]
                f_node = x_node[:, f_id]

                f_correct_global = x_correct_global[:, f_id]
                f_error_global = x_error_global[:, f_id]
                f_correct_node = x_correct_node[:, f_id]
                f_error_node = x_error_node[:, f_id]

                f_values = np.unique(x[:, f_id])
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

                    self.plot_hist(x_hist, bins, labels, colors, alpha=0.5)

                if show_class:
                    x_hist = [f_correct_node, f_error_node]
                    labels = ['node correct', 'node error']
                    colors = class_colors
                else:
                    x_hist = [f_node]
                    labels = ['node']
                    decision = self._error_clf.tree_.value[leaf, :].argmax()
                    colors = [ErrorAnalyzerConstants.ERROR_TREE_COLORS[self._error_clf.classes_[decision]]]

                self.plot_hist(x_hist, bins, labels, colors, alpha=1.0)

                plt.xlabel(f_name)
                plt.ylabel('Proportion of samples')
                plt.legend()
                plt.title('Distribution of %s in Node %d: (%d, %d)' % (f_name, leaf, n_corrects, n_errors))
                plt.pause(0.05)

        plt.show()

    def get_path_to_node(self, node_id, feature_names=None):
        """ return path to node as a list of split steps """
        if feature_names is None:
            feature_names = list(range(self._error_clf.max_features_))

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

    def error_node_summary(self, nodes='all_errors', print_path_to_node=True, feature_names=None):
        """ return summary information regarding input nodes """

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
                path_to_node = self.get_path_to_node(leaf, feature_names)
                for step in path_to_node:
                    print('     ' + step)

