# -*- coding: utf-8 -*-
import numpy as np
from sklearn import tree
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from dku_error_analysis_mpp.kneed import KneeLocator
from dku_error_analysis_utils import not_enough_data
import pydotplus
import graphviz as gv
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from dku_error_analysis_mpp.error_config import *
from dku_error_analysis_tree_parsing.tree_parser import TreeParser
from dku_error_analysis_tree_parsing.depreprocessor import _denormalize_feature_value
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='Error Analysis Plugin | %(levelname)s - %(message)s')


class ErrorAnalyzer:
    """
    ErrorAnalyzer analyzes the errors of a prediction models on a test set.
    It uses model predictions and ground truth target to compute the model errors on the test set.
    It then trains a Decision Tree on the same test set by using the model error as target.
    The nodes of the decision tree are different segments of errors to be studied individually.
    """

    def __init__(self, model_accessor, seed=None):

        if model_accessor is None:
            raise NotImplementedError('You need to define a model accessor.')

        self._model_accessor = model_accessor
        self.target = self._model_accessor.get_target_variable()
        self.is_regression = self._model_accessor.is_regression()

        self._error_df = None
        self.error_clf = None
        self._error_train_X = None
        self._error_train_Y = None
        self._error_test_X = None
        self._error_test_Y = None
        self.preprocessed_x = None
        self.features_in_model_performance_predictor = None
        self.mpp_accuracy_score = None
        self.primary_model_predicted_accuracy = None
        self.primary_model_true_accuracy = None

        self.X_deprocessed = None
        self.feature_names_deprocessed = None
        self.value_mapping = None

        if seed:
            self.seed = seed
        else:
            self.seed = 65537

    def get_preprocessed_array(self):
        return self.preprocessed_x

    def get_model_performance_predictor_test_df(self):
        return self._error_df

    def get_model_performance_predictor_features(self):
        return self.features_in_model_performance_predictor

    def get_model_performance_predictor(self):
        return self.error_clf

    def get_mpp_accuracy_score(self):
        return self.mpp_accuracy_score

    def get_primary_model_predicted_accuracy(self):
        return self.primary_model_predicted_accuracy

    def get_primary_model_true_accuracy(self):
        return self.primary_model_true_accuracy

    def fit(self):
        """
        Trains a Decision Tree to discriminate between samples that are correctly predicted or wrongly predicted
        (errors) by a primary model.
        """
        logger.info("Preparing the model performance predictor...")

        np.random.seed(self.seed)

        original_df = self._model_accessor.get_original_test_df(MAX_NUM_ROW)

        predictor = self._model_accessor.get_predictor()
        self.preprocessed_x, _, _, _, _ = predictor.preprocessing.preprocess(
            original_df,
            with_target=True,
            with_sample_weights=True)

        self._error_df = self.prepare_data_for_model_performance_predictor(original_df)

        error_y = np.array(self._error_df[ERROR_COLUMN])
        error_train_X, error_test_X, error_train_Y, error_test_Y = train_test_split(
            self.preprocessed_x,
            error_y,
            test_size=0.2
        )

        self._error_train_X = error_train_X  # we will use them later when plotting
        self._error_train_Y = error_train_Y

        self._error_test_X = error_test_X  # we will use them later when compute metrics
        self._error_test_Y = error_test_Y
        self.features_in_model_performance_predictor = predictor.get_features()

        logger.info("Fitting the model performance predictor...")

        # entropy/mutual information is used to split nodes in Microsoft Pandora system
        criterion = CRITERION

        dt_clf = tree.DecisionTreeClassifier(criterion=criterion, min_samples_leaf=1, random_state=1337)
        parameters = {'max_depth': MAX_DEPTH_GRID}
        gs_clf = GridSearchCV(dt_clf, parameters, cv=5)
        gs_clf.fit(error_train_X, error_train_Y)
        self.error_clf = gs_clf.best_estimator_

        logger.info('Grid search selected max_depth = {}'.format(gs_clf.best_params_['max_depth']))

        self.compute_model_performance_predictor_metrics()

        self.prepare_features_for_plot()

    def prepare_data_for_model_performance_predictor(self, original_df):
        """
        Computes the errors of the primary model predictions and samples with max n = MAX_NUM_ROW
        :return: a dataframe with error target (correctly predicted vs wrongly predicted)
        """

        logger.info('Prepare data with model for model performance predictor')

        if self.target not in original_df:
            raise ValueError('The original dataset does not contain target "{}".'.format(self.target))

        if not_enough_data(original_df, min_len=MIN_NUM_ROWS):
            raise ValueError(
                'The original dataset is too small ({} rows) to have stable result, it needs to have at least {} rows'.format(
                    len(original_df), MIN_NUM_ROWS))

        logger.info("Rebalancing data:")
        number_of_rows = min(original_df.shape[0], MAX_NUM_ROW)
        logger.info(" - original dataset had %s rows. Selecting the first %s." % (original_df.shape[0], number_of_rows))

        df = original_df.head(number_of_rows)

        # get prediction, is there a way to get predictions from the model handler?
        if PREDICTION_COLUMN not in df.columns:
            df[PREDICTION_COLUMN] = self._model_accessor.predict(df)

        df[ERROR_COLUMN] = self._get_errors(df, prediction_column=PREDICTION_COLUMN)
        df[ERROR_COLUMN] = df[ERROR_COLUMN].replace({True: WRONG_PREDICTION, False: CORRECT_PREDICTION})

        selected_features = [ERROR_COLUMN] + self._model_accessor.get_selected_features()

        return df.loc[:, selected_features]

    def compute_model_performance_predictor_metrics(self):
        """
        MPP ability to predict primary model performance. Ideally primary_model_predicted_accuracy should be equal
        to primary_model_true_accuracy
        """

        y_true = self._error_test_Y
        y_pred = self.error_clf.predict(self._error_test_X)
        n_test_samples = y_pred.shape[0]
        self.mpp_accuracy_score = accuracy_score(y_true, y_pred)
        logger.info('Model Performance Predictor accuracy: {}'.format(self.mpp_accuracy_score))

        self.primary_model_predicted_accuracy = float(np.count_nonzero(y_pred == CORRECT_PREDICTION)) / n_test_samples
        self.primary_model_true_accuracy = float(np.count_nonzero(y_true == CORRECT_PREDICTION)) / n_test_samples

        logger.info('Primary model accuracy: {}'.format(self.primary_model_true_accuracy))
        logger.info('MPP predicted accuracy: {}'.format(self.primary_model_predicted_accuracy))

        difference_true_pred_accuracy = np.abs(self.primary_model_true_accuracy - self.primary_model_predicted_accuracy)
        if difference_true_pred_accuracy > MPP_ACCURACY_TOLERANCE:  # TODO: add message in UI?
            logger.warning("Warning: the built MPP might not be representative of the primary model performances.")

    def _get_epsilon(self, difference, mode='rec'):
        """ compute epsilon to define errors in regression task """
        assert (mode in ['std', 'rec'])
        if mode == 'std':
            std_diff = np.std(difference)
            mean_diff = np.mean(difference)
            epsilon = mean_diff + std_diff
        elif mode == 'rec':
            n_points = NUMBER_EPSILON_VALUES
            epsilon_range = np.linspace(min(difference), max(difference), num=n_points)
            cdf_error = np.zeros_like(epsilon_range)
            n_samples = difference.shape[0]
            for i, epsilon in enumerate(epsilon_range):
                correct = difference <= epsilon
                cdf_error[i] = float(np.count_nonzero(correct)) / n_samples
            kneedle = KneeLocator(epsilon_range, cdf_error)
            epsilon = kneedle.knee
        return epsilon

    def _get_errors(self, test_df, prediction_column):
        """ compute errors of the primary model on the test set """
        if self.is_regression:
            target = test_df[self.target]
            predictions = test_df[prediction_column]
            difference = np.abs(target - predictions)

            epsilon = self._get_epsilon(difference, mode='rec')

            error = difference > epsilon
            return error

        error = (test_df[self.target] != test_df[prediction_column])
        return np.array(error)

    def plot_error_tree(self, size=None):

        digraph_tree = tree.export_graphviz(self.error_clf,
                                            feature_names=self.features_in_model_performance_predictor,
                                            class_names=self.error_clf.classes_,
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
                values = [int(ii) for ii in node.get_label().split('value = [')[1].split(']')[0].split(',')]
                values = [float(v) / sum(values) for v in values]
                node_arg_class = np.argmax(values)
                node_class = self.error_clf.classes_[node_arg_class]
                # transparency as the entropy value
                alpha = values[node_arg_class]
                class_color = ERROR_TREE_COLORS[node_class].strip('#')
                class_color_rgb = tuple(int(class_color[i:i + 2], 16) for i in (0, 2, 4))
                # compute the color as alpha against white
                color_rgb = [int(round(alpha * c + (1 - alpha) * 255, 0)) for c in class_color_rgb]
                color = '#{:02x}{:02x}{:02x}'.format(color_rgb[0], color_rgb[1], color_rgb[2])
                node.set_fillcolor(color)

        if size is not None:
            pydot_graph.set_size('"%d,%d!"' % (size[0], size[1]))
        gvz_graph = gv.Source(pydot_graph.to_string())

        return gvz_graph

    def read_feature(self, preprocessed_feature):
        if preprocessed_feature in self.tree_parser.preprocessed_feature_mapping:
            split_param = self.tree_parser.preprocessed_feature_mapping[preprocessed_feature]
            return split_param.feature, split_param.value
        else:
            return preprocessed_feature, None

    def prepare_features_for_plot(self):

        self.tree_parser = TreeParser(self._model_accessor.model_handler, self.error_clf)
        self.tree = self.tree_parser.build_tree(self._error_df, self.features_in_model_performance_predictor)
        self.tree.parse_nodes(self.tree_parser, self.features_in_model_performance_predictor, self.preprocessed_x)

        rescalers = self.tree_parser.rescalers
        scalings = {rescaler.in_col: rescaler for rescaler in rescalers}

        feature_list = self.features_in_model_performance_predictor

        X = self._error_train_X

        feature_list_undo = [self.read_feature(feature_name)[0] for feature_name in feature_list]
        feature_list_undo = list(dict.fromkeys(feature_list_undo))

        n_features_undo = len(feature_list_undo)
        X_deprocessed = np.zeros((X.shape[0], n_features_undo))
        value_mapping = dict.fromkeys(list(range(n_features_undo)), [])

        for f_id, feature_name in enumerate(feature_list):
            feature_name_undo, feature_value = self.read_feature(feature_name)
            f_id_undo = feature_list_undo.index(feature_name_undo)

            if self._model_accessor.get_per_feature().get(feature_name_undo).get("type") == "NUMERIC":
                x_deprocessed = _denormalize_feature_value(scalings, feature_name, X[:, f_id])
                X_deprocessed[:, f_id_undo] = x_deprocessed
            else:
                samples_indices = np.where(X[:, f_id] == 1)
                if samples_indices is not None:
                    if len(feature_value) > 1:
                        feature_value = 'Others'
                    else:
                        feature_value = feature_value[0]
                    value_mapping[f_id_undo].append(feature_value)
                    samples_indices = samples_indices[0]
                    X_deprocessed[samples_indices, f_id_undo] = len(value_mapping[f_id_undo]) - 1

        self.X_deprocessed = X_deprocessed
        self.feature_names_deprocessed = feature_list_undo
        self.value_mapping = value_mapping

    def plot_hist(self, data, bins, labels, colors, alpha, histtype='bar'):
        n_samples = 0
        for x in data:
            n_samples += x.shape[0]

        weights = [np.ones_like(x, dtype=np.float) / n_samples for x in data]
        plt.hist(data, bins, label=labels, stacked=True, density=False,
                 alpha=alpha, color=colors, weights=weights, histtype=histtype)

    def plot_error_node_feature_distribution(self, nodes='all_errors', top_k_features=3, compare_to_global=True,
                                             show_class=False, figsize=(10, 5)):
        """ return plot of error node feature distribution and compare to global baseline """

        if not (isinstance(nodes, list) or isinstance(nodes, int)):
            assert (nodes in ['all', 'all_errors'])

        if isinstance(nodes, int):
            nodes = [nodes]

        error_class_idx = np.where(self.error_clf.classes_ == WRONG_PREDICTION)[0]
        correct_class_idx = np.where(self.error_clf.classes_ == CORRECT_PREDICTION)[0]

        feature_list = self.features_in_model_performance_predictor
        ranked_features = self.tree_parser._rank_features_by_error_correlation(feature_list,
                                                                               max_number_features=top_k_features,
                                                                               include_non_split_features=True)

        feature_names = self.feature_names_deprocessed

        feature_idx_by_importance = [feature_names.index(feat_name) for feat_name in ranked_features]

        X, Y = self.X_deprocessed, self._error_train_Y

        leave_id = self.error_clf.apply(self._error_train_X)
        leaf_nodes = np.unique(leave_id)
        if nodes is not 'all':
            if nodes is 'all_errors':
                error_leaf_nodes = []
                error_leaf_nodes_importance = []
                for leaf in leaf_nodes:
                    decision = self.error_clf.tree_.value[leaf, :].argmax()
                    if self.error_clf.classes_[decision] == WRONG_PREDICTION:
                        error_leaf_nodes.append(leaf)
                        values = self.error_clf.tree_.value[leaf, :]
                        n_errors = values[0, error_class_idx]
                        n_corrects = values[0, correct_class_idx]
                        leaf_impurity = float(n_corrects) / (n_errors + n_corrects)
                        n_difference = n_corrects - n_errors  # always negative
                        error_leaf_nodes_importance.append(n_difference + leaf_impurity)
                leaf_nodes = [x for _, x in sorted(zip(error_leaf_nodes_importance, error_leaf_nodes))]
            else:
                leaf_nodes = set(nodes) & set(leaf_nodes)
                if not bool(leaf_nodes):
                    print("Selected nodes are not leaf nodes.")
                    return

        X_error_global = X[Y == WRONG_PREDICTION, :]
        X_correct_global = X[Y == CORRECT_PREDICTION, :]

        class_colors = [ERROR_TREE_COLORS[CORRECT_PREDICTION], ERROR_TREE_COLORS[WRONG_PREDICTION]]

        for leaf in leaf_nodes:
            values = self.error_clf.tree_.value[leaf, :]
            n_errors = values[0, error_class_idx]
            n_corrects = values[0, correct_class_idx]
            print('Node %d: (%d correct predictions, %d wrong predictions)' % (leaf, n_corrects, n_errors))
            node_indices = leave_id == leaf
            Y_node = Y[node_indices]
            X_node = X[node_indices, :]
            X_error_node = X_node[Y_node == WRONG_PREDICTION, :]
            X_correct_node = X_node[Y_node == CORRECT_PREDICTION, :]

            for f_id in feature_idx_by_importance:

                plt.figure(figsize=figsize)

                f_name = feature_names[f_id]

                print(f_name)

                f_global = X[:, f_id]
                f_node = X_node[:, f_id]

                f_correct_global = X_correct_global[:, f_id]
                f_error_global = X_error_global[:, f_id]
                f_correct_node = X_correct_node[:, f_id]
                f_error_node = X_error_node[:, f_id]

                if self._model_accessor.get_per_feature().get(f_name).get("type") != "NUMERIC":
                    labels = self.value_mapping[f_id]
                    n_labels = len(labels)
                    bins = np.linspace(0, n_labels - 1, n_labels)
                    ax = plt.gca()
                    ax.set_xticks(bins)
                    ax.set_xticklabels(labels)
                    plt.xticks(rotation=90)
                else:
                    f_values = np.unique(X[:, f_id])
                    bins = np.linspace(np.min(f_values), np.max(f_values))

                if compare_to_global:
                    if show_class:
                        x = [f_correct_global, f_error_global]
                        labels = ['global correct', 'global error']
                        colors = class_colors
                    else:
                        x = [f_global]
                        labels = ['global']
                        colors = [ERROR_TREE_COLORS[CORRECT_PREDICTION]] # global is mainly correct

                    self.plot_hist(x, bins, labels, colors, alpha=0.5)

                if show_class:
                    x = [f_correct_node, f_error_node]
                    labels = ['node correct', 'node error']
                    colors = class_colors
                else:
                    x = [f_node]
                    labels = ['node']
                    decision = self.error_clf.tree_.value[leaf, :].argmax()
                    colors = [ERROR_TREE_COLORS[self.error_clf.classes_[decision]]]

                self.plot_hist(x, bins, labels, colors, alpha=1.0)

                plt.xlabel(f_name)
                plt.ylabel('Proportion of samples')
                plt.legend()
                plt.title('Feature distribution of %s in Node %d: (%d, %d)' % (f_name, leaf, n_corrects, n_errors))
                plt.pause(0.05)

        plt.show()

