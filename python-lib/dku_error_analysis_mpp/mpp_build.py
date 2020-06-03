# -*- coding: utf-8 -*-
import numpy as np
from dku_error_analysis_tree_parsing.tree_parser import TreeParser
from dku_error_analysis_mpp.error_analyzer import ErrorAnalyzer
from dku_error_analysis_mpp.model_accessor import ModelAccessor

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='Error Analysis Plugin | %(levelname)s - %(message)s')


def get_error_dt(model_handler):

    model_accessor = ModelAccessor(model_handler)
    error_analyzer = ErrorAnalyzer(model_accessor)

    error_analyzer.fit()

    error_clf = error_analyzer.get_model_performance_predictor()
    test_df = error_analyzer.get_model_performance_predictor_test_df()
    feature_names = error_analyzer.get_model_performance_predictor_features()
    preprocessed_x = error_analyzer.get_preprocessed_array()

    return error_clf, test_df, preprocessed_x, feature_names


# rank features according to their correlation with the model performance
def rank_features_by_error_correlation(clf, feature_names, tree_parser, max_number_histograms=3):
    sorted_features = sorted(-clf.feature_importances_)
    ranked_features = []
    for feature_idx, feature_importance in enumerate(sorted_features):
        if feature_importance != 0:
            preprocessed_name = feature_names[feature_idx]
            feature = tree_parser.get_preprocessed_feature_details(preprocessed_name)[1]
            if feature not in ranked_features:
                ranked_features.append(feature)
                if len(ranked_features) == max_number_histograms:
                    return ranked_features
    return ranked_features
