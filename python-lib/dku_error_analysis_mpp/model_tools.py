#!/usr/bin/python
# -*- coding: utf-8 -*-

import logging
import numpy as np
import math
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from dku_error_analysis_mpp.preprocessing import Preprocessor

logger = logging.getLogger(__name__)


class SurrogateModel:

    def __init__(self, prediction_type):
        self.feature_names = None
        self.target = None
        self.prediction_type = prediction_type
        #TODO should we define some params of RF to avoid long computation ?
        if prediction_type == 'CLASSIFICATION':
            self.clf = RandomForestClassifier(random_state=1407)
        else:
            self.clf = RandomForestRegressor(random_state=1407)
        self.check()

    def check(self):
        if self.prediction_type not in ['CLASSIFICATION', 'REGRESSION']:
            raise ValueError('Prediction type must either be CLASSIFICATION or REGRESSION.')

    def fit(self, df, target):
        preprocessor = Preprocessor(df, target)
        train, test = preprocessor.get_processed_train_test()
        train_X = train.drop(target, axis=1)
        train_Y = train[target]
        self.clf.fit(train_X, train_Y)
        self.feature_names = train_X.columns

    def get_feature_importance(self, cumulative_percentage_threshold=95):
        feature_importance = []
        feature_importances = self.clf.feature_importances_
        for feature_name, feat_importance in zip(self.feature_names, feature_importances):
            feature_importance.append({
                'feature': feature_name,
                'importance': 100 * feat_importance / sum(feature_importances)
            })

        dfx = pd.DataFrame(feature_importance).sort_values(by='importance', ascending=False).reset_index(drop=True)
        dfx['cumulative_importance'] = dfx['importance'].cumsum()
        dfx_top = dfx.loc[dfx['cumulative_importance'] <= cumulative_percentage_threshold]
        return dfx_top.rename_axis('rank').reset_index().set_index('feature')