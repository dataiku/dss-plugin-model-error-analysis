# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import collections
from sklearn.model_selection import train_test_split
from dku_error_analysis_model_parser.model_handler_utils import get_original_test_df
from dku_error_analysis_tree_parsing.tree_parser import TreeParser
from dku_error_analysis_utils import DkuMEAConstants
import logging
from mealy import ErrorAnalyzer, ErrorAnalyzerConstants

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='Error Analysis Plugin | %(levelname)s - %(message)s')


class DkuErrorAnalyzer(ErrorAnalyzer):
    """
    DkuErrorAnalyzer analyzes the errors of a DSS prediction model on its test set.
    It uses model predictions and ground truth target to compute the model errors on the test set.
    It then trains a Decision Tree on the same test set by using the model error as target.
    The nodes of the decision tree are different segments of errors to be studied individually.
    """

    def __init__(self,
                 model_handler,
                 max_num_rows=DkuMEAConstants.MAX_NUM_ROWS,
                 param_grid=None,
                 random_state=65537):

        if model_handler is None:
            raise NotImplementedError('You need to define a model handler.')

        self._model_handler = model_handler
        self._target = model_handler.get_target_variable()
        self._model_predictor = model_handler.get_predictor()
        self._max_num_rows = max_num_rows

        feature_names = self._model_predictor.get_features()
        super(DkuErrorAnalyzer, self).__init__(model_handler.get_clf(), feature_names, param_grid, random_state)

        self._train_x = None
        self._test_x = None
        self._train_y = None
        self._test_y = None

        self._error_df = None

        self._tree = None
        self._tree_parser = None

    @property
    def tree(self):
        if self._tree is None:
            self.parse_tree()
        return self._tree

    @property
    def tree_parser(self):
        return self._tree_parser

    def fit(self):
        """
        Trains a Decision Tree to discriminate between samples that are correctly predicted or wrongly predicted
        (errors) by a primary model.
        """

        self._prepare_data_from_dku_saved_model()

        super(DkuErrorAnalyzer, self).fit(self._train_x, self._train_y)

    def _preprocess_dataframe(self, df, with_target=True):
        """ Preprocess input DataFrame with primary model preprocessor """
        if with_target and self._target not in df:
            raise ValueError('The dataset does not contain target "{}".'.format(self._target))
        if with_target:
            x, input_mf_index, _, y = self._model_predictor.preprocessing.preprocess(
                df,
                with_target=True)
            return x, y, input_mf_index
        return self._model_predictor.preprocessing.preprocess(df)[0]

    def _prepare_data_from_dku_saved_model(self):
        """ Preprocess and split original test set from Dku saved model
        into train and test set for the error analyzer """
        np.random.seed(self.random_state)

        original_df = get_original_test_df(self._model_handler)[:self._max_num_rows]

        preprocessed_x, y, input_mf_index = self._preprocess_dataframe(original_df)

        x_df = pd.DataFrame(preprocessed_x, index=input_mf_index)
        y_df = pd.Series(y, index=input_mf_index)

        train_x_df, test_x_df, train_y_df, test_y_df = train_test_split(
            x_df, y_df, test_size=ErrorAnalyzerConstants.TEST_SIZE
        )

        self._train_x = train_x_df.values
        self._train_y = train_y_df.values

        self._test_x = test_x_df.values
        self._test_y = test_y_df.values

        original_train_df = original_df.loc[train_x_df.index]

        self._error_df = original_train_df.drop(self._target, axis=1)

    def parse_tree(self):
        """ Parse Decision Tree and get features information used to display distributions """
        self._error_df.loc[:, DkuMEAConstants.ERROR_COLUMN] = self._error_train_y
        self._tree_parser = TreeParser(self._model_handler, self.error_tree.estimator_)
        self._tree = self._tree_parser.build_tree(self._error_df, self.preprocessed_feature_names, self._error_train_x)

    def _get_path_to_node(self, node_id):
        """ return path to node as a list of split steps from the nodes of the de-processed
        dku_error_analysis_decision_tree.tree.InteractiveTree object """
        cur_node = self.tree.get_node(node_id)
        path_to_node = collections.deque()
        while cur_node is not None and cur_node.id != 0:
            path_to_node.appendleft(cur_node.print_decision_rule())
            cur_node = self.tree.get_node(cur_node.parent_id)

        return path_to_node

    def evaluate(self, dku_test_dataset=None, output_format='dict'):
        """ print ErrorAnalyzer summary metrics """
        if dku_test_dataset is None:
            return super(DkuErrorAnalyzer, self).evaluate(self._test_x, self._test_y, output_format=output_format)
        else:
            test_df = dku_test_dataset.get_dataframe()
            test_x, test_y, _ = self._preprocess_dataframe(test_df)
            return super(DkuErrorAnalyzer, self).evaluate(test_x, test_y.values, output_format=output_format)

    def predict(self, dku_test_dataset):
        """ Predict model performance on Dku dataset """
        test_df = dku_test_dataset.get_dataframe()

        test_x = self._preprocess_dataframe(test_df, with_target=False)

        return super(DkuErrorAnalyzer, self).predict(test_x)
