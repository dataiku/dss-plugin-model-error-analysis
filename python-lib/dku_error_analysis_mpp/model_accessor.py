#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import logging

logger = logging.getLogger(__name__)

ALGORITHMS_WITH_VARIABLE_IMPORTANCE = [RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier,
                                       DecisionTreeClassifier]
MAX_NUM_ROW = 100000
SURROGATE_TARGET = "_dku_predicted_label_"


class ModelAccessor:

    def __init__(self, model_handler=None):
        self.model_handler = model_handler

    def get_prediction_type(self):
        """
        Wrap the prediction type accessor of the model
        """
        if 'CLASSIFICATION' in self.model_handler.get_prediction_type():
            return 'CLASSIFICATION'
        elif 'REGRESSION' in self.model_handler.get_prediction_type():
            return 'REGRESSION'
        else:
            return 'CLUSTERING'

    def check(self):
        """
        Check missing model_handler
        """
        if self.model_handler is None:
            raise ValueError('model_handler object is not specified')

    def get_target_variable(self):
        """
        Return the name of the target variable
        """
        return self.model_handler.get_target_variable()

    def get_original_test_df(self, limit=MAX_NUM_ROW):
        try:
            return self.model_handler.get_test_df()[0][:limit]
        except Exception as e:
            logger.warning(
                'Can not retrieve original test set: {}. The plugin will take the whole original dataset.'.format(e))
            return self.model_handler.get_full_df()[0][:limit]

    def get_predictor(self):
        return self.model_handler.get_predictor()

    def get_selected_features(self):
        selected_features = []
        for feat, feat_info in self.get_per_feature().items():
            if feat_info.get('role') == 'INPUT':
                selected_features.append(feat)
        return selected_features

    def predict(self, df):
        return self.get_predictor().predict(df)
