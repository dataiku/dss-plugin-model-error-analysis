# -*- coding: utf-8 -*-
import numpy as np
from sklearn import tree
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.base import is_regressor
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from dku_error_analysis_mpp.kneed import KneeLocator
from dku_error_analysis_utils import not_enough_data
import pydotplus
import graphviz as gv
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from dku_error_analysis_mpp.error_config import ErrorAnalyzerConstants
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

    def __init__(self, predictor, seed=None):

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
        self._preprocessed_x = None
        self._features_in_model_performance_predictor = None
        self._mpp_accuracy_score = None
        self._primary_model_predicted_accuracy = None
        self._primary_model_true_accuracy = None

        self._error_train_leaf_id = None
        self._ranked_error_nodes = None

        self._error_y = None

        self._tree = None
        self._tree_parser = None

        if seed:
            self._seed = seed
        else:
            self._seed = 65537

    @property
    def tree(self):
        return self._tree

    @property
    def error(self):
        return self._error_y

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
    def mpp_accuracy_score(self):
        return self._mpp_accuracy_score

    @property
    def primary_model_predicted_accuracy(self):
        return self._primary_model_predicted_accuracy

    @property
    def primary_model_true_accuracy(self):
        return self._primary_model_true_accuracy

    def fit(self, x, y):
        """
        Trains a Decision Tree to discriminate between samples that are correctly predicted or wrongly predicted
        (errors) by a primary model.
        """
        logger.info("Preparing the model performance predictor...")

        np.random.seed(self._seed)

        self._preprocessed_x = x
        self.prepare_data_for_model_performance_predictor(x, y)

        error_train_x, error_test_x, error_train_y, error_test_y = train_test_split(
            self._preprocessed_x,
            self._error_y,
            test_size=0.2
        )

        self._error_train_x = error_train_x  # we will use them later when plotting
        self._error_train_y = error_train_y

        self._error_test_x = error_test_x  # we will use them later when compute metrics
        self._error_test_y = error_test_y

        logger.info("Fitting the model performance predictor...")

        # entropy/mutual information is used to split nodes in Microsoft Pandora system
        criterion = ErrorAnalyzerConstants.CRITERION

        dt_clf = tree.DecisionTreeClassifier(criterion=criterion, min_samples_leaf=1, random_state=1337)
        parameters = {'max_depth': ErrorAnalyzerConstants.MAX_DEPTH_GRID}
        gs_clf = GridSearchCV(dt_clf, parameters, cv=5)
        gs_clf.fit(error_train_x, error_train_y)
        self._error_clf = gs_clf.best_estimator_

        logger.info('Grid search selected max_depth = {}'.format(gs_clf.best_params_['max_depth']))

        self.compute_model_performance_predictor_metrics()

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

        self._error_y = self._get_errors(y, y_pred)

        return

    def compute_model_performance_predictor_metrics(self):
        """
        MPP ability to predict primary model performance. Ideally primary_model_predicted_accuracy should be equal
        to primary_model_true_accuracy
        """

        y_true = self._error_test_y
        y_pred = self._error_clf.predict(self._error_test_x)
        n_test_samples = y_pred.shape[0]
        self._mpp_accuracy_score = accuracy_score(y_true, y_pred)
        logger.info('Model Performance Predictor accuracy: {}'.format(self._mpp_accuracy_score))

        self._primary_model_predicted_accuracy = float(
            np.count_nonzero(y_pred == ErrorAnalyzerConstants.CORRECT_PREDICTION)) / n_test_samples
        self._primary_model_true_accuracy = float(
            np.count_nonzero(y_true == ErrorAnalyzerConstants.CORRECT_PREDICTION)) / n_test_samples

        logger.info('Primary model accuracy: {}'.format(self._primary_model_true_accuracy))
        logger.info('MPP predicted accuracy: {}'.format(self._primary_model_predicted_accuracy))

        difference_true_pred_accuracy = np.abs(
            self._primary_model_true_accuracy - self._primary_model_predicted_accuracy)
        if difference_true_pred_accuracy > ErrorAnalyzerConstants.MPP_ACCURACY_TOLERANCE:  # TODO: add message in UI?
            logger.warning("Warning: the built MPP might not be representative of the primary model performances.")

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

    def get_leaf_ids(self):
        if self._error_train_leaf_id is None:
            self._error_train_leaf_id = self._error_clf.apply(self._error_train_x)

        return self._error_train_leaf_id

    def get_ranked_error_nodes(self):
        if self._ranked_error_nodes is None:
            error_leaf_nodes = []
            error_leaf_nodes_importance = []
            leaf_ids = self.get_leaf_ids()
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
            self._ranked_error_nodes = [x for _, x in sorted(zip(error_leaf_nodes_importance, error_leaf_nodes))]

        return self._ranked_error_nodes

    def get_list_of_nodes(self, nodes):

        if not (isinstance(nodes, list) or isinstance(nodes, int)):
            assert (nodes in ['all', 'all_errors'])

        if isinstance(nodes, int):
            nodes = [nodes]

        leaf_ids = self.get_leaf_ids()
        leaf_nodes = np.unique(leaf_ids)
        if nodes is not 'all':
            if nodes is 'all_errors':
                leaf_nodes = self.get_ranked_error_nodes()
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

        leaf_ids = self.get_leaf_ids()
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

    def mpp_summary(self):
        """ print ErrorAnalyzer summary metrics """
        print('The ErrorAnalyzer Decision Tree was trained with accuracy %.2f%%.' %
              (self._mpp_accuracy_score * 100))
        print('The Decision Tree estimated the primary model''s accuracy to %.2f%%.' %
              (self._primary_model_predicted_accuracy * 100))
        print('The true accuracy of the primary model is %.2f.%%' %
              (self._primary_model_true_accuracy * 100))
        inv_fidelity = np.abs(self._primary_model_predicted_accuracy - self._primary_model_true_accuracy)
        fidelity = 1.-inv_fidelity

        if inv_fidelity <= ErrorAnalyzerConstants.MPP_ACCURACY_TOLERANCE:
            print('The Fidelity of the ErrorAnalyzer is %.2f%%, which is sufficient to trust its results.' %
                  (fidelity * 100))
        else:
            print('The Fidelity of the ErrorAnalyzer is %.2f%%, which might invalidate its results.' %
                  (fidelity * 100))





