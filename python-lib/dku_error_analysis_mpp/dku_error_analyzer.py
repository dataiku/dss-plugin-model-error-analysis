# -*- coding: utf-8 -*-
import numpy as np
import pydotplus
import graphviz as gv
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn import tree
from dku_error_analysis_mpp.error_config import ErrorAnalyzerConstants
from dku_error_analysis_tree_parsing.tree_parser import TreeParser
from dku_error_analysis_decision_tree.node import Node
from dku_error_analysis_tree_parsing.depreprocessor import _denormalize_feature_value
from dku_error_analysis_mpp.error_analyzer import ErrorAnalyzer
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

    def __init__(self, model_accessor, seed=None):

        if model_accessor is None:
            raise NotImplementedError('you need to define a model accessor.')

        self._model_accessor = model_accessor
        self._target = self._model_accessor.get_target_variable()
        self._is_regression = self._model_accessor.is_regression()
        self._predictor = self._model_accessor.get_predictor()

        self._error_df = None

        if seed:
            self._seed = seed
        else:
            self._seed = 65537

        self._error_analyzer = ErrorAnalyzer(self._model_accessor.get_clf(), self._seed)

        self._preprocessed_x = None
        self._x_deprocessed = None
        self._error_df = None
        self._error_clf = None
        self._feature_names_deprocessed = None
        self._features_in_model_performance_predictor = None
        self._value_mapping = None
        self._tree = None
        self._tree_parser = None


    @property
    def tree(self):
        return self._tree

    @property
    def model_performance_predictor_features(self):
        return self._features_in_model_performance_predictor

    @property
    def model_performance_predictor(self):
        return self._error_analyzer.model_performance_predictor

    @property
    def mpp_accuracy_score(self):
        return self._error_analyzer.mpp_accuracy_score

    @property
    def primary_model_predicted_accuracy(self):
        return self._error_analyzer.primary_model_predicted_accuracy

    @property
    def primary_model_true_accuracy(self):
        return self._error_analyzer.primary_model_true_accuracy

    def fit(self):
        """
        Trains a Decision Tree to discriminate between samples that are correctly predicted or wrongly predicted
        (errors) by a primary model.
        """

        np.random.seed(self._seed)

        original_df = self._model_accessor.get_original_test_df(ErrorAnalyzerConstants.MAX_NUM_ROW)

        if self._target not in original_df:
            raise ValueError('The original dataset does not contain target "{}".'.format(self._target))

        self._preprocessed_x, _, _, _, _ = self._predictor.preprocessing.preprocess(
            original_df,
            with_target=True,
            with_sample_weights=True)

        x = self._preprocessed_x

        y = original_df[self._target]
        y = np.array(y)

        self._error_analyzer.fit(x, y)

        self._error_clf = self._error_analyzer.model_performance_predictor

        self.prepare_features_for_plot()

    def read_feature(self, preprocessed_feature):
        if preprocessed_feature in self._tree_parser.preprocessed_feature_mapping:
            split_param = self._tree_parser.preprocessed_feature_mapping[preprocessed_feature]
            return split_param.feature, split_param.value
        else:
            return preprocessed_feature, None

    def prepare_features_for_plot(self):

        self._features_in_model_performance_predictor = self._predictor.get_features()

        original_df = self._model_accessor.get_original_test_df(ErrorAnalyzerConstants.MAX_NUM_ROW)

        modified_length = len(self._error_analyzer.error)
        original_df = original_df.head(modified_length)

        self._error_df = original_df.drop(self._target, axis=1)
        self._error_df[ErrorAnalyzerConstants.ERROR_COLUMN] = self._error_analyzer.error

        self._tree_parser = TreeParser(self._model_accessor.model_handler, self._error_clf)
        self._tree = self._tree_parser.build_tree(self._error_df, self._features_in_model_performance_predictor)
        self._tree.parse_nodes(self._tree_parser, self._features_in_model_performance_predictor, self._preprocessed_x)

        rescalers = self._tree_parser.rescalers
        scalings = {rescaler.in_col: rescaler for rescaler in rescalers}

        feature_list = self._features_in_model_performance_predictor

        x = self._error_analyzer.error_train_x

        feature_list_undo = [self.read_feature(feature_name)[0] for feature_name in feature_list]
        feature_list_undo = list(dict.fromkeys(feature_list_undo))

        n_features_undo = len(feature_list_undo)
        x_deprocessed = np.zeros((x.shape[0], n_features_undo))
        value_mapping = dict.fromkeys(list(range(n_features_undo)), [])

        for f_id, feature_name in enumerate(feature_list):
            feature_name_undo, feature_value = self.read_feature(feature_name)
            f_id_undo = feature_list_undo.index(feature_name_undo)

            if self._model_accessor.get_per_feature().get(feature_name_undo).get("type") == "NUMERIC":
                x_deprocessed[:, f_id_undo] = _denormalize_feature_value(scalings, feature_name, x[:, f_id])
            else:
                samples_indices = np.where(x[:, f_id] == 1)
                if samples_indices is not None:
                    if len(feature_value) > 1:
                        feature_value = 'Others'
                    else:
                        feature_value = feature_value[0]
                    value_mapping[f_id_undo].append(feature_value)
                    samples_indices = samples_indices[0]
                    x_deprocessed[samples_indices, f_id_undo] = len(value_mapping[f_id_undo]) - 1

        self._x_deprocessed = x_deprocessed
        self._feature_names_deprocessed = feature_list_undo
        self._value_mapping = value_mapping

    def plot_error_tree(self, size=None):

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

    def plot_error_node_feature_distribution(self, nodes='all_errors', top_k_features=3, compare_to_global=True,
                                             show_class=False, figsize=(10, 5)):
        """ return plot of error node feature distribution and compare to global baseline """

        if not (isinstance(nodes, list) or isinstance(nodes, int)):
            assert (nodes in ['all', 'all_errors'])

        if isinstance(nodes, int):
            nodes = [nodes]

        error_class_idx = np.where(self._error_clf.classes_ == ErrorAnalyzerConstants.WRONG_PREDICTION)[0]
        correct_class_idx = np.where(self._error_clf.classes_ == ErrorAnalyzerConstants.CORRECT_PREDICTION)[0]

        feature_list = self._features_in_model_performance_predictor
        ranked_features = self._tree_parser._rank_features_by_error_correlation(feature_list,
                                                                                max_number_features=top_k_features,
                                                                                include_non_split_features=True)

        feature_names = self._feature_names_deprocessed

        feature_idx_by_importance = [feature_names.index(feat_name) for feat_name in ranked_features]

        x, y = self._x_deprocessed, self._error_analyzer.error_train_y

        leaf_nodes = self._error_analyzer.get_list_of_nodes(nodes)
        leaf_ids = self._error_analyzer.get_leaf_ids()

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

                if self._model_accessor.get_per_feature().get(f_name).get("type") != "NUMERIC":
                    labels = self._value_mapping[f_id]
                    n_labels = len(labels)
                    bins = np.linspace(0, n_labels - 1, n_labels)
                    ax = plt.gca()
                    ax.set_xticks(bins)
                    ax.set_xticklabels(labels)
                    plt.xticks(rotation=90)
                else:
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

                    self._error_analyzer.plot_hist(x_hist, bins, labels, colors, alpha=0.5)

                if show_class:
                    x_hist = [f_correct_node, f_error_node]
                    labels = ['node correct', 'node error']
                    colors = class_colors
                else:
                    x_hist = [f_node]
                    labels = ['node']
                    decision = self._error_clf.tree_.value[leaf, :].argmax()
                    colors = [ErrorAnalyzerConstants.ERROR_TREE_COLORS[self._error_clf.classes_[decision]]]

                self._error_analyzer.plot_hist(x_hist, bins, labels, colors, alpha=1.0)

                plt.xlabel(f_name)
                plt.ylabel('Proportion of samples')
                plt.legend()
                plt.title('Distribution of %s in Node %d: (%d, %d)' % (f_name, leaf, n_corrects, n_errors))
                plt.pause(0.05)

        plt.show()

    def get_path_to_node(self, node_idx):
        run_node_idx = node_idx
        path_to_node = []
        while self._tree.nodes[run_node_idx].feature:
            cur_node = self._tree.nodes[run_node_idx]
            feature = cur_node.feature
            if cur_node.get_type() == Node.TYPES.NUM:
                if cur_node.beginning:
                    sign = ' > '
                    value = "%.2f" % cur_node.beginning
                else:
                    sign = ' <= '
                    value = "%.2f" % cur_node.end
            else:
                if cur_node.others:
                    sign = ' != '
                else:
                    sign = ' == '
                value = cur_node.values[0]
            path_to_node.append(feature + sign + value)
            run_node_idx = self._tree.nodes[run_node_idx].parent_id
        path_to_node = path_to_node[::-1]

        return path_to_node

    def error_node_summary(self, nodes='all_errors'):
        """ return summary information regarding input nodes """
        leaf_nodes = self._error_analyzer.get_list_of_nodes(nodes)

        for leaf in leaf_nodes:
            self._error_analyzer.error_node_summary(nodes=[leaf], print_path_to_node=False)
            print(' Path to node:')
            path_to_node = self.get_path_to_node(leaf)
            for step in path_to_node:
                print('     ' + step)

    def mpp_summary(self):
        """ print ErrorAnalyzer summary metrics """

        self._error_analyzer.mpp_summary()
