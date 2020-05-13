# -*- coding: utf-8 -*-
import numpy as np
from dku_error_tree_parsing.tree_parser import TreeParser
from dku_error_analysis_mpp.error_analyzer import ErrorAnalyzer
from dku_error_analysis_mpp.model_accessor import ModelAccessor

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='Error Analysis Plugin | %(levelname)s - %(message)s')


def get_error_dt(model_handler):
    error_analyzer = ErrorAnalyzer()
    model_accessor = ModelAccessor(model_handler)

    error_analyzer.fit(model_accessor)

    error_clf = error_analyzer.get_model_performance_predictor()
    test_df = error_analyzer.get_model_performance_predictor_test_df()
    feature_names = error_analyzer.get_model_performance_predictor_features()

    return error_clf, test_df, feature_names


# rank features according to their correlation with the model performance
def rank_features_by_error_correlation(clf, feature_names, max_number_histograms=3):
    feature_idx_by_importance = np.argsort(-clf.feature_importances_)
    ranked_features = []
    for feature_idx in feature_idx_by_importance:
        feature = TreeParser.read_feature(feature_names[feature_idx])[0]
        if feature not in ranked_features:
            ranked_features.append(feature)
            if len(ranked_features) == max_number_histograms:
                return ranked_features
    return ranked_features
