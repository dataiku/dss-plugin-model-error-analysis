#!/usr/bin/python
# -*- coding: utf-8 -*-

import logging, sys
from dku_error_analysis_utils import safe_str

from dataiku.doctor.posttraining.model_information_handler import PredictionModelInformationHandler

LOGGER = logging.getLogger(__name__)

def get_model_handler(model, version_id=None):
    try:
        params = model.get_predictor(version_id).params
        return PredictionModelInformationHandler(params.split_desc, params.core_params, params.model_folder, params.model_folder)
    except Exception as e:
        from future.utils import raise_
        if "ordinal not in range(128)" in safe_str(e):
            raise_(Exception, "Model Error Analysis requires models built with python3. This one is on python2.", sys.exc_info()[2])
        else:
            raise_(Exception, "Fail to load saved model: {}".format(e), sys.exc_info()[2])

def get_original_test_df(model_handler):
    try:
        return model_handler.get_test_df()[0]
    except Exception as e:
        LOGGER.warning('Cannot retrieve original test set: {}. The plugin will take the whole original dataset.'.format(e))
        return model_handler.get_full_df()[0]
