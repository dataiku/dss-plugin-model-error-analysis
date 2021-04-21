import numpy as np
from dku_error_analysis_decision_tree.node import Node
from dku_error_analysis_decision_tree.tree import InteractiveTree
from dataiku.doctor.preprocessing.dataframe_preprocessing import RescalingProcessor2, QuantileBinSeries, UnfoldVectorProcessor, BinarizeSeries, \
    FastSparseDummifyProcessor, ImpactCodingStep, FlagMissingValue2, TextCountVectorizerProcessor, TextHashingVectorizerWithSVDProcessor, \
    TextHashingVectorizerProcessor, TextTFIDFVectorizerProcessor
from dku_error_analysis_utils import DkuMEAConstants
from dku_error_analysis_tree_parsing.depreprocessor import descale_numerical_thresholds
from collections import deque
from mealy import ErrorAnalyzerConstants

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

    def _add_flag_missing_value_mapping(self, step):
        self.preprocessed_feature_mapping[step._output_name()] = self.SplitParameters(Node.TYPES.CAT, step.feature, [np.nan])

    # CATEGORICAL HANDLING
    def _add_dummy_mapping(self, step):
        for value in step.values:
            preprocessed_name = "dummy:{}:{}".format(step.input_column_name, value)
            self.preprocessed_feature_mapping[preprocessed_name] = self.SplitParameters(Node.TYPES.CAT, step.input_column_name, [value], force_others_on_right=True)
        self.preprocessed_feature_mapping["dummy:{}:N/A".format(step.input_column_name)] = self.SplitParameters(Node.TYPES.CAT, step.input_column_name, [np.nan], force_others_on_right=True)
        if not step.should_drop:
            self.preprocessed_feature_mapping["dummy:{}:__Others__".format(step.input_column_name)] = self.SplitParameters(Node.TYPES.CAT, step.input_column_name, step.values)

    def _add_impact_mapping(self, step):
        impact_map = getattr(step.impact_coder, "_impact_map", getattr(step.impact_coder, "encoding_map", None)) # To handle DSS10 new implem
        for value in impact_map.columns.values:
            preprocessed_name = "impact:{}:{}".format(step.column_name, value)
            display_name = "{} [{}]".format(step.column_name, value)
            self.preprocessed_feature_mapping[preprocessed_name] = self.SplitParameters(Node.TYPES.NUM, display_name, uses_preprocessed_feature=True)

    # NUMERICAL HANDLING
    def _add_binarize_mapping(self, step):
        self.preprocessed_feature_mapping["num_binarized:" + step._output_name()] = self.SplitParameters(Node.TYPES.NUM, step.in_col, step.threshold)

    def _add_quantize_mapping(self, step):
        bounds = step.r["bounds"]
        value_func = lambda threshold: float(bounds[int(threshold) + 1])
        preprocessed_name = "num_quantized:{0}:quantile:{1}".format(step.in_col, step.nb_bins)
        self.preprocessed_feature_mapping[preprocessed_name] = self.SplitParameters(Node.TYPES.NUM, step.in_col, value_func=value_func)

    # VECTOR HANDLING
    def _add_unfold_mapping(self, step):
        for i in range(step.vector_length):
            preprocessed_name = "unfold:{}:{}".format(step.input_column_name, i)
            display_name = "{} [element #{}]".format(step.input_column_name, i)
            self.preprocessed_feature_mapping[preprocessed_name] = self.SplitParameters(Node.TYPES.NUM, display_name, uses_preprocessed_feature=True)

    # TEXT HANDLING
    def _add_hashing_vect_mapping(self, step, with_svd=False):
        prefix = "thsvd" if with_svd else "hashvect"
        for i in range(step.n_features):
            preprocessed_name = "{}:{}:{}".format(prefix, step.column_name, i)
            display_name = "{} [text #{}]".format(step.column_name, i)
            self.preprocessed_feature_mapping[preprocessed_name] = self.SplitParameters(Node.TYPES.NUM, display_name, uses_preprocessed_feature=True)

    def _add_text_count_vect_mapping(self, step):
        for word in step.resource["vectorizer"].get_feature_names():
            preprocessed_name = "{}:{}:{}".format(step.prefix, step.column_name, word)
            display_name = "{}: occurrences of {}".format(step.column_name, word)
            self.preprocessed_feature_mapping[preprocessed_name] = self.SplitParameters(Node.TYPES.NUM, display_name, value_func=lambda threshold: int(threshold), uses_preprocessed_feature=True)

    def _add_tfidf_vect_mapping(self, step):
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
                FlagMissingValue2: self._add_flag_missing_value_mapping,
                QuantileBinSeries: self._add_quantize_mapping,
                FastSparseDummifyProcessor: self._add_dummy_mapping,
                BinarizeSeries: self._add_binarize_mapping,
                UnfoldVectorProcessor: self._add_unfold_mapping,
                ImpactCodingStep: self._add_impact_mapping,
                TextCountVectorizerProcessor: self._add_text_count_vect_mapping,
                TextHashingVectorizerWithSVDProcessor: lambda step: self._add_hashing_vect_mapping(step, with_svd=True),
                TextHashingVectorizerProcessor: lambda step: self._add_hashing_vect_mapping(step),
                TextTFIDFVectorizerProcessor: self._add_tfidf_vect_mapping
            }.get(step.__class__, lambda step: None)(step)

    def _get_split_parameters(self, preprocessed_name, threshold=None):
        return self.preprocessed_feature_mapping.get(preprocessed_name, self.SplitParameters(Node.TYPES.NUM, preprocessed_name, threshold))

    def build_tree(self, df, feature_list, preprocessed_x, target=DkuMEAConstants.ERROR_COLUMN):
        num_features = {}
        for name, settings in self.model_handler.get_preproc_handler().collector_data.get('per_feature').items():
            avg = settings.get('stats').get('average')
            if avg is not None: # TODO AGU: add cases where missing is not replaced by mean (ch49216)
                num_features[name] = {
                    'mean':  avg
                }

        # Retrieve feature names without duplicates while keeping the ranking order
        ranked_feature_ids = np.argsort(-self.error_model.feature_importances_)
        ranked_feature_ids_unique, seen_values = [], set()
        for idx in ranked_feature_ids:
            unprocessed_name = self._get_split_parameters(feature_list[idx]).feature
            if unprocessed_name not in seen_values:
                ranked_feature_ids_unique.append(unprocessed_name)
                seen_values.add(unprocessed_name)

        tree = InteractiveTree(df, target, ranked_feature_ids_unique, num_features)
        self.parse_nodes(tree, feature_list, preprocessed_x)
        return tree

    def parse_nodes(self, tree, feature_list, preprocessed_x):
        error_model_tree = self.error_model.tree_
        thresholds = descale_numerical_thresholds(error_model_tree, feature_list, self.rescalers, False)
        children_left, children_right, features = error_model_tree.children_left, error_model_tree.children_right, error_model_tree.feature
        error_class_idx = np.where(self.error_model.classes_ == ErrorAnalyzerConstants.WRONG_PREDICTION)[0][0]

        ids = deque()
        ids.append(0)
        while ids:
            node_id = ids.popleft()

            # Add node info
            class_samples = {
                ErrorAnalyzerConstants.WRONG_PREDICTION: error_model_tree.value[node_id, 0, error_class_idx],
                ErrorAnalyzerConstants.CORRECT_PREDICTION: error_model_tree.value[node_id, 0, 1 - error_class_idx]
            }
            samples = error_model_tree.n_node_samples[node_id]
            tree.set_node_info(node_id, samples, class_samples)

            # Create its children if any
            if children_left[node_id] < 0:
                # Current node is a leaf
                tree.leaves.add(node_id)
                continue

            feature_idx, threshold = features[node_id], thresholds[node_id]
            preprocessed_feature = feature_list[feature_idx]
            split_parameters = self._get_split_parameters(preprocessed_feature, threshold)

            if split_parameters.uses_preprocessed_feature:
                tree.df[split_parameters.feature] = preprocessed_x[:, feature_idx]
            if split_parameters.value is None:
                split_parameters.value = split_parameters.value_func(threshold)
            if split_parameters.force_others_on_right:
                left_child_id, right_child_id = children_right[node_id], children_left[node_id]
            else:
                left_child_id, right_child_id = children_left[node_id], children_right[node_id]

            tree.add_split_no_siblings(split_parameters.node_type, node_id, split_parameters.feature, split_parameters.value, left_child_id, right_child_id)

            ids.append(left_child_id)
            ids.append(right_child_id)
