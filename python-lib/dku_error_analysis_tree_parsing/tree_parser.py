import numpy as np
from dku_error_analysis_decision_tree.node import Node
from dku_error_analysis_decision_tree.tree import InteractiveTree
from dataiku.doctor.preprocessing.dataframe_preprocessing import RescalingProcessor2, QuantileBinSeries, UnfoldVectorProcessor, BinarizeSeries, \
    FastSparseDummifyProcessor, ImpactCodingStep, FlagMissingValue2, TextCountVectorizerProcessor, TextHashingVectorizerWithSVDProcessor, \
    TextHashingVectorizerProcessor, TextTFIDFVectorizerProcessor
from dku_error_analysis_utils import ErrorAnalyzerConstants, rank_features_by_error_correlation
from functools import reduce


class TreeParser(object):
    class SplitParameters(object):
        def __init__(self, node_type, feature, value=None, value_func=lambda threshold: threshold, uses_preprocessed_feature=False, force_others_on_right=False):
            self.node_type = node_type
            self.feature = feature
            self.value = value
            self.value_func = value_func
            self.uses_preprocessed_feature = uses_preprocessed_feature
            self.force_others_on_right = force_others_on_right

    def __init__(self, model_handler, error_model):
        self.model_handler = model_handler
        self.error_model = error_model
        self.preprocessed_feature_mapping = {}
        self.rescalers = []
        self._create_preprocessed_feature_mapping()

    def add_flag_missing_value_mapping(self, step):
        self.preprocessed_feature_mapping[step._output_name()] = self.SplitParameters(Node.TYPES.CAT, step.feature, [np.nan])

    def add_dummy_mapping(self, step):
        for value in step.values:
            preprocessed_name = "dummy:{}:{}".format(step.input_column_name, value)
            self.preprocessed_feature_mapping[preprocessed_name] = self.SplitParameters(Node.TYPES.CAT, step.input_column_name, [value], force_others_on_right=True)
        self.preprocessed_feature_mapping["dummy:{}:N/A".format(step.input_column_name)] = self.SplitParameters(Node.TYPES.CAT, step.input_column_name, [np.nan], force_others_on_right=True)
        if not step.should_drop:
            self.preprocessed_feature_mapping["dummy:{}:__Others__".format(step.input_column_name)] = self.SplitParameters(Node.TYPES.CAT, step.input_column_name, step.values)

    def add_impact_mapping(self, step):
        for value in step.impact_coder._impact_map.columns.values:
            preprocessed_name = "impact:{}:{}".format(step.column_name, value)
            display_name = "{} [{}]".format(step.column_name, value)
            self.preprocessed_feature_mapping[preprocessed_name] = self.SplitParameters(Node.TYPES.NUM, display_name, uses_preprocessed_feature=True)

    def add_binarize_mapping(self, step):
        self.preprocessed_feature_mapping["num_binarized:" + step._output_name()] = self.SplitParameters(Node.TYPES.NUM, step.in_col, step.threshold)

    def add_quantize_mapping(self, step):
        bounds = step.r["bounds"]
        value_func = lambda threshold: float(bounds[int(threshold) + 1])
        preprocessed_name = "num_quantized:{0}:quantile:{1}".format(step.in_col, step.nb_bins)
        self.preprocessed_feature_mapping[preprocessed_name] = self.SplitParameters(Node.TYPES.NUM, step.in_col, value_func=value_func)

    def add_unfold_mapping(self, step):
        for i in xrange(step.vector_length):
            preprocessed_name = "unfold:{}:{}".format(step.input_column_name, i)
            display_name = "{} [element #{}]".format(step.input_column_name, i)
            self.preprocessed_feature_mapping[preprocessed_name] = self.SplitParameters(Node.TYPES.NUM, display_name, uses_preprocessed_feature=True)

    def add_hashing_vect_mapping(self, step, with_svd=False):
        prefix = "thsvd" if with_svd else "hashvect"
        for i in range(step.n_features):
            preprocessed_name = "{}:{}:{}".format(prefix, step.column_name, i)
            display_name = "{} [text #{}]".format(step.column_name, i)
            self.preprocessed_feature_mapping[preprocessed_name] = self.SplitParameters(Node.TYPES.NUM, display_name, uses_preprocessed_feature=True)

    def add_text_count_vect_mapping(self, step):
        for word in step.resource["vectorizer"].get_feature_names():
            preprocessed_name = "{}:{}:{}".format(step.prefix, step.column_name, word)
            display_name = "{}: occurrences of {}".format(step.column_name, word)
            self.preprocessed_feature_mapping[preprocessed_name] = self.SplitParameters(Node.TYPES.NUM, display_name, value_func=lambda threshold: int(threshold), uses_preprocessed_feature=True)

    def add_tfidf_vect_mapping(self, step):
        vec = step.resource["vectorizer"]
        for word, idf in zip(vec.get_feature_names(), vec.idf_):
            preprocessed_name = "tfidfvec:{}:{:.3f}:{}".format(step.column_name, idf, word)
            display_name = "{}: tfidf of {}".format(step.column_name, word)
            self.preprocessed_feature_mapping[preprocessed_name] = self.SplitParameters(Node.TYPES.NUM, display_name, uses_preprocessed_feature=True)

    def _create_preprocessed_feature_mapping(self):
        for step in self.model_handler.get_pipeline().steps:
            if isinstance(step, RescalingProcessor2):
                self.rescalers.append(step)
            {
                FlagMissingValue2: self.add_flag_missing_value_mapping,
                QuantileBinSeries: self.add_quantize_mapping,
                FastSparseDummifyProcessor: self.add_dummy_mapping,
                BinarizeSeries: self.add_binarize_mapping,
                UnfoldVectorProcessor: self.add_unfold_mapping,
                ImpactCodingStep: self.add_impact_mapping,
                TextCountVectorizerProcessor: self.add_text_count_vect_mapping,
                TextHashingVectorizerWithSVDProcessor: lambda step: self.add_hashing_vect_mapping(step, with_svd=True),
                TextHashingVectorizerProcessor: lambda step: self.add_hashing_vect_mapping(step),
                TextTFIDFVectorizerProcessor: self.add_tfidf_vect_mapping
            }.get(step.__class__, lambda step: None)(step)

    def get_split_parameters(self, preprocessed_name, threshold=None):
        return self.preprocessed_feature_mapping.get(preprocessed_name, self.SplitParameters(Node.TYPES.NUM, preprocessed_name, threshold))

    def build_tree(self, df, feature_list, target=ErrorAnalyzerConstants.ERROR_COLUMN):
        features = {}
        for name, settings in self.model_handler.get_preproc_handler().collector_data.get('per_feature').items():
            avg = settings.get('stats').get('average')
            if avg is not None: # TODO AGU: add cases where missing is not replaced by mean (ch49216)
                features[name] = {
                    'mean':  avg
                }

        ranked_feature_ids = rank_features_by_error_correlation(self.error_model.feature_importances_)
        def get_unique_ranked_features(accumulated_list, current_value, seen_values=set()):
            unprocessed_name = self.get_split_parameters(feature_list[current_value]).feature
            if unprocessed_name not in seen_values:
                accumulated_list.append(unprocessed_name)
                seen_values.add(unprocessed_name)
            return accumulated_list

        ranked_features = list(reduce(get_unique_ranked_features, ranked_feature_ids, []))
        tree = InteractiveTree(df, target, ranked_features, features)
        return tree
