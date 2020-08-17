# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from dku_error_analysis_mpp.error_config import ErrorAnalyzerConstants
from dku_error_analysis_tree_parsing.tree_parser import TreeParser
from dku_error_analysis_decision_tree.node import Node
from dku_error_analysis_mpp.error_analyzer import ErrorAnalyzer
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='Error Analysis Plugin | %(levelname)s - %(message)s')


class DkuErrorAnalyzer(ErrorAnalyzer):
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
        self._model_predictor = self._model_accessor.get_predictor()
        feature_names = self._model_predictor.get_features()

        super(DkuErrorAnalyzer, self).__init__(self._model_accessor.get_clf(), feature_names, seed)

        self._train_x = None
        self._test_x = None
        self._train_y = None
        self._test_y = None

        self._train_x_df = None
        self._test_x_df = None
        self._train_y_df = None
        self._test_y_df = None

        self._error_df = None
        self._tree = None
        self._tree_parser = None
        self._features_dict = None

        self._dku_error_visualizer = None

    @property
    def tree(self):
        return self._tree

    @property
    def tree_parser(self):
        return self._tree_parser

    @property
    def features_dict(self):
        return self._features_dict

    def fit(self):
        """
        Trains a Decision Tree to discriminate between samples that are correctly predicted or wrongly predicted
        (errors) by a primary model.
        """

        self._prepare_data_from_dku_saved_model()

        super(DkuErrorAnalyzer, self).fit(self._train_x, self._train_y)

    def _preprocess_dataframe(self, df):
        """ Preprocess input DataFrame with primary model preprocessor """
        x, _, _, _, _ = self._model_predictor.preprocessing.preprocess(
            df,
            with_target=True,
            with_sample_weights=True)

        y_raw = np.array(df[self._target])
        y = map(lambda y_raw: self._model_accessor.get_target_map().get(y_raw), y_raw)
        return x, y

    def _prepare_data_from_dku_saved_model(self):
        """ Split original test set from Dku saved model into train and test set for the error analyzer """
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

        self._train_x, self._train_y = self._preprocess_dataframe(train_df)
        self._test_x, self._test_y = self._preprocess_dataframe(test_df)

    def parse_tree(self):
        """ Parse Decision Tree and get features information used to display distributions """
        modified_length = len(self.error_train_x)
        self._error_df = self._train_x_df.head(modified_length)
        self._error_df.loc[:, ErrorAnalyzerConstants.ERROR_COLUMN] = self.error_train_y

        self._tree_parser = TreeParser(self._model_accessor.model_handler, self._error_clf)
        self._tree = self._tree_parser.build_tree(self._error_df, self._features_in_model_performance_predictor)
        self._tree.parse_nodes(self._tree_parser,
                               self._features_in_model_performance_predictor,
                               self.error_train_x)

        self._features_dict = self._model_accessor.get_per_feature()

    def get_path_to_node(self, node_id):
        """ return path to node as a list of split steps from the nodes of the de-processed
        dku_error_analysis_decision_tree.tree.InteractiveTree object """
        run_node_idx = node_id
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

    def mpp_summary(self, dku_test_dataset=None, output_dict=False):
        """ print ErrorAnalyzer summary metrics """
        if dku_test_dataset is None:
            return super(DkuErrorAnalyzer, self).mpp_summary(self._test_x, self._test_y, output_dict)
        else:
            test_df = dku_test_dataset.get_dataframe()
            if self._target not in test_df:
                raise ValueError('The provided dataset does not contain target "{}".'.format(self._target))
            test_x, test_y = self._preprocess_dataframe(test_df)
            return super(DkuErrorAnalyzer, self).mpp_summary(test_x, test_y, output_dict)
