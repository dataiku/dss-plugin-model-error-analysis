#!/usr/bin/python
# -*- coding: utf-8 -*-

import logging, sys
from dku_error_analysis_utils import safe_str

from dataiku.doctor.posttraining.model_information_handler import PredictionModelInformationHandler

logger = logging.getLogger(__name__)

class ModelAccessor(object):
    def __init__(self, model, version_id=None):
        params = model.get_predictor(version_id).params

        assert params.core_params.get("taskType") == "PREDICTION", "Model error analysis view can only be used with prediction models"

        try:
            self.model_handler = PredictionModelInformationHandler(params.split_desc, params.core_params, params.model_folder, params.model_folder)
        except Exception as e:
            from future.utils import raise_
            if "ordinal not in range(128)" in safe_str(e):
                raise_(Exception, "The plugin is using a python3 code-env, cannot load a python2 model.", sys.exc_info()[2])
            elif safe_str(e) == "non-string names in Numpy dtype unpickling":
                raise_(Exception, "The plugin is using a python2 code-env, cannot load a python3 model.", sys.exc_info()[2])
            else:
                raise_(Exception, "Fail to load saved model: {}".format(e), sys.exc_info()[2])

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

    def get_target_map(self):
        return self.model_handler.get_target_map()

    def predict(self, df):
        return self.get_predictor().predict(df, with_probas=False)

    def get_clf(self):
        return self.model_handler.get_clf()
