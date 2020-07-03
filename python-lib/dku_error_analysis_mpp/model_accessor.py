#!/usr/bin/python
# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger(__name__)


class ModelAccessor:

    def __init__(self, model_handler):
        self.model_handler = model_handler
        # check missing model_handler
        if self.model_handler is None:
            raise ValueError('model_handler object is not specified')

    def is_regression(self):
        """
        Return whether the prediciton is a regression or not (i.e. a classification)
        """
        if 'REGRESSION' in self.model_handler.get_prediction_type():
            return True
        return False

    def get_target_variable(self):
        """
        Return the name of the target variable
        """
        return self.model_handler.get_target_variable()

    def get_original_test_df(self, limit):
        try:
            return self.model_handler.get_test_df()[0][:limit]
        except Exception as e:
            logger.warning(
                'Can not retrieve original test set: {}. The plugin will take the whole original dataset.'.format(e))
            return self.model_handler.get_full_df()[0][:limit]

    def get_predictor(self):
        return self.model_handler.get_predictor()

    def get_per_feature(self):
        return self.model_handler.get_per_feature()

    def get_selected_features(self):
        selected_features = []
        for feat, feat_info in self.get_per_feature().items():
            if feat_info.get('role') == 'INPUT':
                selected_features.append(feat)
        return selected_features

    def predict(self, df):
        return self.get_predictor().predict(df, with_probas=False)

    def get_clf(self):
        return self.model_handler.get_clf()
