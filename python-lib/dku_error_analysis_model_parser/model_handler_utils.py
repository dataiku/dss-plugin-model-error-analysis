#!/usr/bin/python
# -*- coding: utf-8 -*-

import logging, sys
from dku_error_analysis_utils import safe_str

from dataiku.doctor.posttraining.model_information_handler import PredictionModelInformationHandler

LOGGER = logging.getLogger(__name__)

def get_model_handler(model, version_id=None):
    params = model.get_predictor(version_id).params

    assert params.core_params.get("taskType") == "PREDICTION", "Model error analysis view can only be used with prediction models"

    try:
        return PredictionModelInformationHandler(params.split_desc, params.core_params, params.model_folder, params.model_folder)
    except Exception as e:
        from future.utils import raise_
        if "ordinal not in range(128)" in safe_str(e):
            raise_(Exception, "The plugin is using a python3 code-env, cannot load a python2 model.", sys.exc_info()[2])
        elif safe_str(e) == "non-string names in Numpy dtype unpickling":
            raise_(Exception, "The plugin is using a python2 code-env, cannot load a python3 model.", sys.exc_info()[2])
        else:
            raise_(Exception, "Fail to load saved model: {}".format(e), sys.exc_info()[2])

def get_original_test_df(model_handler):
    try:
        return model_handler.get_test_df()[0]
    except Exception as e:
        LOGGER.warning('Cannot retrieve original test set: {}. The plugin will take the whole original dataset.'.format(e))
        return model_handler.get_full_df()[0]
