import numpy as np
from dku_error_analysis_decision_tree.node import Node
from dataiku.doctor.preprocessing.dataframe_preprocessing import RescalingProcessor2, QuantileBinSeries, UnfoldVectorProcessor, BinarizeSeries, \
    FastSparseDummifyProcessor, ImpactCodingStep, FlagMissingValue2, TextCountVectorizerProcessor, TextHashingVectorizerWithSVDProcessor, \
    TextHashingVectorizerProcessor, TextTFIDFVectorizerProcessor, CategoricalFeatureHashingProcessor
from dku_error_analysis_tree_parsing.depreprocessor import descale_numerical_thresholds, denormalize_feature_value
from dku_error_analysis_utils import format_float
from dku_error_analysis_decision_tree.tree import InteractiveTree
from dku_error_analysis_utils import DkuMEAConstants
from collections import deque
from mealy import ErrorAnalyzerConstants
import logging
from json import loads
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='Error Analysis Plugin | %(levelname)s - %(message)s')

class TreeParser(object):
    class SplitParameters(object):
        def __init__(self, node_type, chart_name, value=None, friendly_name=None,
                     value_func=lambda threshold: threshold,
                     add_preprocessed_feature=lambda array, col: array[:, col],
                     invert_left_and_right=lambda threshold: False):
            self.node_type = node_type
            self.chart_name = chart_name
            self.friendly_name = friendly_name
            self.value = value
            self.value_func = value_func
            self.add_preprocessed_feature = add_preprocessed_feature
            self.invert_left_and_right = invert_left_and_right

        @property
        def feature(self):
            return self.friendly_name or self.chart_name

    def __init__(self, model_handler, error_model, feature_list):
        self.model_handler = model_handler
        self.error_model = error_model
        self.feature_list = feature_list
        self.preprocessed_feature_mapping = {}
        self.rescalers = {}
        self.num_features = set()
        self._create_preprocessed_feature_mapping()

    def _add_flag_missing_value_mapping(self, step):
        if step.output_block_name == "num_flagonly":
            self.num_features.add(step.feature)
        self.preprocessed_feature_mapping[step._output_name()] = \
            self.SplitParameters(Node.TYPES.CAT, step.feature, [np.nan])

    # CATEGORICAL HANDLING
    def _add_cat_hashing_not_whole_mapping(self, step):
        logger.warning(
            "The model uses categorical hashing without whole category hashing enabled.\
            This is not recommanded."
        )
        for i in range(step.n_features):
            preprocessed_name = "hashing:{}:{}".format(step.column_name, i)
            friendly_name = "Hash #{} of {}".format(i, step.column_name)
            self.preprocessed_feature_mapping[preprocessed_name] = \
                self.SplitParameters(Node.TYPES.NUM, step.column_name, friendly_name=friendly_name)

    def _add_cat_hashing_whole_mapping(self, step):
        value_func = lambda i: lambda threshold: [threshold * 2 * i]
        friendly_name = "Hash of {}".format(step.column_name)
        add_preprocessed_feature = lambda i: lambda array, col: np.sum(
            np.multiply(range(step.n_features), array[:, col - i : col - i + step.n_features]),
            axis=1)

        for i in range(step.n_features):
            preprocessed_name = "hashing:{}:{}".format(step.column_name, i)
            self.preprocessed_feature_mapping[preprocessed_name] = \
                self.SplitParameters(Node.TYPES.CAT, step.column_name,
                                     friendly_name=friendly_name,
                                     value_func=value_func(i),
                                     add_preprocessed_feature=add_preprocessed_feature(i),
                                     invert_left_and_right=lambda threshold: threshold > 0)

    def _add_dummy_mapping(self, step):
        for value in step.values:
            preprocessed_name = "dummy:{}:{}".format(step.input_column_name, value)
            self.preprocessed_feature_mapping[preprocessed_name] = self.SplitParameters(Node.TYPES.CAT, step.input_column_name, [value], invert_left_and_right=lambda threshold: True)
        self.preprocessed_feature_mapping["dummy:{}:N/A".format(step.input_column_name)] = self.SplitParameters(Node.TYPES.CAT, step.input_column_name, [np.nan], invert_left_and_right=lambda threshold: True)
        if not step.should_drop:
            self.preprocessed_feature_mapping["dummy:{}:__Others__".format(step.input_column_name)] = self.SplitParameters(Node.TYPES.CAT, step.input_column_name, step.values)

    def _add_impact_mapping(self, step):
        impact_map = getattr(step.impact_coder, "_impact_map", getattr(step.impact_coder, "encoding_map", None)) # To handle DSS10 new implem
        for value in impact_map.columns.values:
            preprocessed_name = "impact:{}:{}".format(step.column_name, value)
            friendly_name = "{} [{}]".format(step.column_name, value)
            self.preprocessed_feature_mapping[preprocessed_name] = self.SplitParameters(Node.TYPES.NUM, step.column_name, friendly_name=friendly_name)

    # NUMERICAL HANDLING
    def _add_preprocessed_rescaled_num_feature(self, original_name):
        original_idx = self.feature_list.index(original_name)
        add_feature = lambda array, col: pd.Series(array[:, original_idx]).apply(lambda x: denormalize_feature_value(self.rescalers.get(original_name), x))
        name = "preprocessed:rescaled:{}".format(original_name)
        return add_feature, name

    def _add_identity_mapping(self, original_name):
        self.num_features.add(original_name)
        add_feature, name = self._add_preprocessed_rescaled_num_feature(original_name)
        self.preprocessed_feature_mapping[original_name] = self.SplitParameters(Node.TYPES.NUM,
            original_name, add_preprocessed_feature=add_feature, friendly_name=name)

    def _add_binarize_mapping(self, step):
        self.num_features.add(step.in_col)
        add_feature, name = self._add_preprocessed_rescaled_num_feature(step.in_col)
        self.preprocessed_feature_mapping["num_binarized:" + step._output_name()] = self.SplitParameters(Node.TYPES.NUM, step.in_col, step.threshold, add_preprocessed_feature=add_feature, friendly_name=name)

    def _add_quantize_mapping(self, step):
        bounds = step.r["bounds"]
        value_func = lambda threshold: float(bounds[int(threshold) + 1])
        preprocessed_name = "num_quantized:{0}:quantile:{1}".format(step.in_col, step.nb_bins)
        add_feature, name = self._add_preprocessed_rescaled_num_feature(step.in_col)
        self.num_features.add(step.in_col)
        self.preprocessed_feature_mapping[preprocessed_name] = self.SplitParameters(Node.TYPES.NUM, step.in_col, value_func=value_func, add_preprocessed_feature=add_feature, friendly_name=name)

    # VECTOR HANDLING
    def _add_unfold_mapping(self, step):
        for i in range(step.vector_length):
            preprocessed_name = "unfold:{}:{}".format(step.input_column_name, i)
            friendly_name = "{} [element #{}]".format(step.input_column_name, i)
            self.num_features.add(friendly_name)
            self.preprocessed_feature_mapping[preprocessed_name] = self.SplitParameters(Node.TYPES.NUM, friendly_name)

    # TEXT HANDLING
    def _add_hashing_vect_mapping(self, step, with_svd=False):
        logger.info("Feature {} is a text feature. ".format(step.column_name) +
            "Its distribution plot will not be available")
        prefix = "thsvd" if with_svd else "hashvect"
        for i in range(step.n_features):
            preprocessed_name = "{}:{}:{}".format(prefix, step.column_name, i)
            friendly_name = "{} [text #{}]".format(step.column_name, i)
            self.preprocessed_feature_mapping[preprocessed_name] = self.SplitParameters(Node.TYPES.NUM, None, friendly_name=friendly_name)

    def _add_text_count_vect_mapping(self, step):
        logger.info("Feature {} is a text feature. ".format(step.column_name) +
            "Its distribution plot will not be available")
        for word in step.resource["vectorizer"].get_feature_names():
            preprocessed_name = "{}:{}:{}".format(step.prefix, step.column_name, word)
            friendly_name = "{}: occurrences of {}".format(step.column_name, word)
            self.preprocessed_feature_mapping[preprocessed_name] = self.SplitParameters(Node.TYPES.NUM, None, friendly_name=friendly_name)

    def _add_tfidf_vect_mapping(self, step):
        logger.info("Feature {} is a text feature. ".format(step.column_name) +
            "Its distribution plot will not be available")
        vec = step.resource["vectorizer"]
        for word, idf in zip(vec.get_feature_names(), vec.idf_):
            preprocessed_name = "tfidfvec:{}:{}:{}".format(step.column_name, format_float(idf, 3), word)
            friendly_name = "{}: tf-idf of {} (idf={})".format(step.column_name, word, format_float(idf, 3))
            self.preprocessed_feature_mapping[preprocessed_name] = self.SplitParameters(Node.TYPES.NUM, None, friendly_name=friendly_name)

    def _create_preprocessed_feature_mapping(self):
        for step in self.model_handler.get_pipeline().steps:
            if isinstance(step, RescalingProcessor2):
                self.rescalers[step.in_col] = step
            {
                CategoricalFeatureHashingProcessor: \
                    lambda step: self._add_cat_hashing_whole_mapping(step) if getattr(step, "hash_whole_categories", False)\
                        else self._add_cat_hashing_not_whole_mapping(step),
                FlagMissingValue2: self._add_flag_missing_value_mapping,
                QuantileBinSeries: self._add_quantize_mapping,
                FastSparseDummifyProcessor: self._add_dummy_mapping,
                BinarizeSeries: self._add_binarize_mapping,
                UnfoldVectorProcessor: self._add_unfold_mapping,
                ImpactCodingStep: self._add_impact_mapping,
                TextCountVectorizerProcessor: self._add_text_count_vect_mapping,
                TextHashingVectorizerWithSVDProcessor: lambda step: self._add_hashing_vect_mapping(step, with_svd=True),
                TextHashingVectorizerProcessor: self._add_hashing_vect_mapping,
                TextTFIDFVectorizerProcessor: self._add_tfidf_vect_mapping
            }.get(step.__class__, lambda step: None)(step)

    def _get_split_parameters(self, preprocessed_name):
        # Numerical features can have no preprocessing performed on them ("kept as regular")
        if preprocessed_name not in self.preprocessed_feature_mapping:
            self._add_identity_mapping(preprocessed_name)
        return self.preprocessed_feature_mapping[preprocessed_name]

    # PARSING

    def create_tree(self, df):
        # Retrieve feature names without duplicates while keeping the ranking order
        ranked_feature_ids = np.argsort(- self.error_model.feature_importances_)
        unique_ranked_feature_names, seen_values = [], set()
        for idx in ranked_feature_ids:
            chart_name = self._get_split_parameters(self.feature_list[idx]).chart_name
            if chart_name not in seen_values and chart_name is not None:
                unique_ranked_feature_names.append(chart_name)
                seen_values.add(chart_name)

        # Add features that were rejected in the original model
        for name, params in self.model_handler.get_per_feature().items():
            if params["role"] == "REJECT" or params["role"] == "WEIGHT":
                if params["type"] == "VECTOR":
                    # Unfold vector column
                    try:
                        vector_col_no_nan = df[name].dropna()
                        unfolded = pd.DataFrame(vector_col_no_nan.map(loads).tolist(), index=vector_col_no_nan.index).replace("", np.nan)
                        columns = ["{} [element #{}]".format(name, i)
                                for i in range(unfolded.shape[1])]
                        df[columns] = unfolded
                        unique_ranked_feature_names += columns
                        self.num_features.update(column for i, column in enumerate(columns)
                                                if pd.api.types.is_numeric_dtype(unfolded[i]))
                    except Exception as e:
                        logger.warning(("Error while parsing vector feature {}: {}. ".format(name, e) +
                            "Its distribution plot will not be available"))
                elif params["type"] == "NUMERIC":
                    self.num_features.add(name)
                    unique_ranked_feature_names.append(name)
                elif params["type"] == "CATEGORY":
                    unique_ranked_feature_names.append(name)
                elif params["type"] == "TEXT":
                    logger.info("Feature {} is a text feature. ".format(name) +
                    "Its distribution plot will not be available")
        return InteractiveTree(df, DkuMEAConstants.ERROR_COLUMN, unique_ranked_feature_names, self.num_features)

    def parse_nodes(self, tree, preprocessed_x):
        error_model_tree = self.error_model.tree_
        thresholds = descale_numerical_thresholds(error_model_tree, self.feature_list, self.rescalers)
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
            tree.set_node_info(node_id, class_samples)

            # Create its children if any
            if children_left[node_id] < 0:
                continue

            feature_idx, threshold = features[node_id], thresholds[node_id]
            preprocessed_feature = self.feature_list[feature_idx]
            split_parameters = self._get_split_parameters(preprocessed_feature)
            if split_parameters.feature not in tree.df:
                tree.df[split_parameters.feature] = split_parameters.add_preprocessed_feature(preprocessed_x, feature_idx)
            value = split_parameters.value
            if value is None:
                value = split_parameters.value_func(threshold)
            if split_parameters.invert_left_and_right(threshold):
                left_child_id, right_child_id = children_right[node_id], children_left[node_id]
            else:
                left_child_id, right_child_id = children_left[node_id], children_right[node_id]

            tree.add_split_no_siblings(split_parameters.node_type, node_id, split_parameters.feature, value, left_child_id, right_child_id)

            ids.append(left_child_id)
            ids.append(right_child_id)
