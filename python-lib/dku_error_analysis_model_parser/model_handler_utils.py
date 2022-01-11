#!/usr/bin/python
# -*- coding: utf-8 -*-

from dku_error_analysis_utils import safe_str
from dataiku.doctor.posttraining.model_information_handler import PredictionModelInformationHandler


def get_model_handler(model, version_id=None):
    try:
        params = model.get_predictor(version_id).params
        return PredictionModelInformationHandler(
            params.split_desc, params.core_params, params.model_folder, params.model_folder
        )
    except Exception as e:
        if "ordinal not in range(128)" in safe_str(e):
            raise Exception("Model stress test only supports models built with Python 3. This one was built with Python 2.") from None
        else:
            raise e
