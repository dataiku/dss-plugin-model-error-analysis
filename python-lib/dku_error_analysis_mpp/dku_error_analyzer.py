# -*- coding: utf-8 -*-
import numpy as np
from dku_error_analysis_utils import DkuMEAConstants
import logging
from mealy_local.error_analyzer import ErrorAnalyzer
from dku_error_analysis_tree_parsing.tree_parser import TreeParser

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='Error Analysis Plugin | %(levelname)s - %(message)s')


class DkuErrorAnalyzer(ErrorAnalyzer):
    """
    DkuErrorAnalyzer analyzes the errors of a DSS prediction model on its test set.
    It uses model predictions and ground truth target to compute the model errors on the test set.
    It then trains a Decision Tree on the same test set by using the model error as target.
    The nodes of the decision tree are different segments of errors to be studied individually.
    """

    def __init__(self, model_handler, max_num_rows=DkuMEAConstants.MAX_NUM_ROWS, param_grid=None,
        random_state=65537):

        self._model_handler = model_handler
        self._target = model_handler.get_target_variable()
        self._model_predictor = model_handler.get_predictor()
        self._max_num_rows = max_num_rows

        probability_threshold = self._model_predictor.params.model_perf.get('usedThreshold', None)
        feature_names = self._model_predictor.get_features()

        estimator = model_handler.get_clf()
        if not hasattr(estimator, "_estimator_type"):
            # This param is needed for the analyzer to properly treat the estimator as a
            # regressor/classifier. It can be absent if the model does not match the sklearn
            # convention for estimators (see sc-78591).
            estimator._estimator_type = "regressor" if model_handler.get_prediction_type() == "REGRESSION" else "classifier"
        super(DkuErrorAnalyzer, self).__init__(
            estimator, feature_names, param_grid, probability_threshold, random_state
        )

        self._train_x = None
        self._train_y = None

        self._error_df = None

        self._tree = None

    @property
    def tree(self):
        if self._tree is None:
            self._parse_tree()
        return self._tree

    @property
    def error_df(self):
        if DkuMEAConstants.ERROR_COLUMN not in self._error_df:
            self._error_df.loc[:, DkuMEAConstants.ERROR_COLUMN] = self._error_train_y
        return self._error_df

    def _parse_tree(self):
        # Beware that self.generated_features_mapping is computed during the init, meaning the methods call sequencing is important !
        tree_parser = TreeParser(self._model_handler, self.error_tree.estimator_,
                                 self.preprocessed_feature_names, self.generated_features_mapping)
        tree = tree_parser.create_tree(self.error_df)
        tree_parser.parse_nodes(tree, self._error_train_x)
        self._tree = tree

    def fit(self):
        """
        Trains a Decision Tree to discriminate between samples that are correctly predicted or wrongly predicted
        (errors) by a primary model.
        """

        self._prepare_data_from_dku_saved_model()

        super(DkuErrorAnalyzer, self).fit(self._train_x, self._train_y)

    def _preprocess_dataframe(self, df):
        """ Preprocess input DataFrame with primary model preprocessor """
        if self._target not in df:
            raise ValueError('The dataset does not contain target "{}".'.format(self._target))
        x, input_mf_index, _, y = self._model_predictor.preprocessing.preprocess(
            df,
            with_target=True)
        return x, y, input_mf_index

    def _prepare_data_from_dku_saved_model(self):
        """ Preprocess and split original test set from Dku saved model
        into train and test set for the error analyzer """
        np.random.seed(self.random_state)
        try:
            original_df = self._model_handler.get_test_df()[0][:self._max_num_rows]
        except Exception as e:
            logger.warning('Cannot retrieve original test set: %s.' +
                'The plugin will take the whole original dataset.', e)
            original_df = self._model_handler.get_full_df()[0][:self._max_num_rows]
        self._train_x, self._train_y, input_mf_index = self._preprocess_dataframe(original_df)
        self.generated_features_mapping = self._model_predictor.preprocessing.pipeline_with_target.generated_features_mapping
        original_train_df = original_df.loc[input_mf_index]
        self._error_df = original_train_df.drop(self._target, axis=1)

    def evaluate(self, dku_test_dataset=None, output_format='dict'):
        """ Return ErrorAnalyzer summary metrics """
        if dku_test_dataset is None:
            return super(DkuErrorAnalyzer, self).evaluate(self._train_x, self._train_y,
                                                          output_format=output_format)
        test_df = dku_test_dataset.get_dataframe()
        test_x, test_y, _ = self._preprocess_dataframe(test_df)
        return super(DkuErrorAnalyzer, self).evaluate(test_x, test_y.values,
                                                      output_format=output_format)
