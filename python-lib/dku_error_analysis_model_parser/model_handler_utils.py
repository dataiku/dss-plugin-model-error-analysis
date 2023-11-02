#!/usr/bin/python
# -*- coding: utf-8 -*-

from dataiku.doctor.posttraining.model_information_handler import PredictionModelInformationHandler

# Only used in the notebook template
def get_model_handler(model, version_id=None):
    fmi = "S-{project_key}-{model_id}-{version_id}".format(project_key=model.project_key, model_id=model.get_id(), version_id=version_id)
    return PredictionModelInformationHandler.from_full_model_id(fmi)
