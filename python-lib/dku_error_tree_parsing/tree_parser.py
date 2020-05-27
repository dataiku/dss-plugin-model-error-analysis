import numpy as np
from collections import deque
from dku_error_analysis_utils.sm_metadata import get_model_handler
from dku_error_analysis_decision_tree.node import Node, NumericalNode, CategoricalNode
from dku_error_analysis_decision_tree.tree import InteractiveTree
from dataiku.doctor.preprocessing.dataframe_preprocessing import RescalingProcessor2, QuantileBinSeries, UnfoldVectorProcessor, BinarizeSeries, \
FastSparseDummifyProcessor, ImpactCodingStep, FlagMissingValue2
from dku_error_tree_parsing.depreprocessor import descale_numerical_thresholds

class TreeParser(object):
    class Preprocessing:
        DUMMY="dummy"
        DUMMY_OTHERS="dummy_others"
        FLAG="flag"
        IMPACT="impact"
        QUANTIZE="quantize"
        UNFOLD="unfold"
        BINARIZE="binarize"

    def __init__(self, model_handler, error_tree):
        self.model_handler = model_handler
        self.error_tree = error_tree
        self.preprocessed_feature_mapping = {}
        self.rescalers = []

    def create_preprocessed_feature_mapping(self):
        for step in self.model_handler.get_pipeline().steps:
            if isinstance(step, RescalingProcessor2):
                self.rescalers.append(step)
            elif isinstance(step, FlagMissingValue2):
                self.preprocessed_feature_mapping[step._output_name()] = (Node.TYPES.CAT, step.feature, [np.nan], TreeParser.Preprocessing.FLAG)
            elif isinstance(step, QuantileBinSeries):
                bounds = step.r["bounds"]
                split_value_func = lambda threshold: float(bounds[int(threshold) + 1])
                preprocessed_name = "num_quantized:{0}:quantile:{1}".format(step.in_col, step.nb_bins)
                self.preprocessed_feature_mapping[preprocessed_name] = (Node.TYPES.NUM, step.in_col, split_value_func, TreeParser.Preprocessing.QUANTIZE)
            elif isinstance(step, FastSparseDummifyProcessor):
                for value in step.values:
                    preprocessed_name = "dummy:{}:{}".format(step.input_column_name, value)
                    self.preprocessed_feature_mapping[preprocessed_name] = (Node.TYPES.CAT, step.input_column_name, [value], TreeParser.Preprocessing.DUMMY)
                self.preprocessed_feature_mapping["dummy:{}:N/A".format(step.input_column_name)] = (Node.TYPES.CAT, step.input_column_name, [np.nan], TreeParser.Preprocessing.DUMMY)
                if not step.should_drop:
                    self.preprocessed_feature_mapping["dummy:{}:__Others__".format(step.input_column_name)] = (Node.TYPES.CAT, step.input_column_name, step.values, TreeParser.Preprocessing.DUMMY_OTHERS)
            elif isinstance(step, BinarizeSeries):
                self.preprocessed_feature_mapping["num_binarized:" + step._output_name()] = (Node.TYPES.NUM, step.in_col, step.threshold, TreeParser.Preprocessing.BINARIZE)
            elif isinstance(step, UnfoldVectorProcessor):
                for i in xrange(step.vector_length):
                    preprocessed_name = "unfold:{}:{}".format(step.input_column_name, i)
                    display_name = "{} [element #{}]".format(step.input_column_name, i)
                    self.preprocessed_feature_mapping[preprocessed_name] = (Node.TYPES.NUM, display_name, lambda threshold: threshold, TreeParser.Preprocessing.UNFOLD)
            elif isinstance(step, ImpactCodingStep):
                for value in step.impact_coder._impact_map.columns.values:
                    preprocessed_name = "impact:{}:{}".format(step.column_name, value)
                    display_name = "{} [{}]".format(step.column_name, value)
                    self.preprocessed_feature_mapping[preprocessed_name] = (Node.TYPES.NUM, display_name, lambda threshold: threshold, TreeParser.Preprocessing.IMPACT) 

    @staticmethod
    def split_value_is_threshold_dependant(method):
        return method == TreeParser.Preprocessing.IMPACT or method == TreeParser.Preprocessing.UNFOLD or method == TreeParser.Preprocessing.QUANTIZE

    @staticmethod
    def split_uses_preprocessed_feature(method):
        return method == TreeParser.Preprocessing.IMPACT or method == TreeParser.Preprocessing.UNFOLD

    @staticmethod
    def force_others_on_the_right(method):
        return method == TreeParser.Preprocessing.DUMMY

    def build_all_nodes(self, tree, feature_list, transformed_df):
        thresholds = descale_numerical_thresholds(self.error_tree, feature_list, self.rescalers, False)
        children_left, children_right, features = self.error_tree.children_left, self.error_tree.children_right, self.error_tree.feature
        root_node = Node(0, -1)
        ids = deque()
        tree.add_node(root_node)

        ids.append(0)
        while ids:
            parent_id = ids.popleft()
            feature_idx, threshold = features[parent_id], thresholds[parent_id]
            preprocessed_feature = feature_list[feature_idx]
            node_type, feature, split_value, method = self.preprocessed_feature_mapping.get(preprocessed_feature, (Node.TYPES.NUM, preprocessed_feature, threshold, None))
            if TreeParser.split_uses_preprocessed_feature(method):
                tree.df[feature] = transformed_df[:, feature_idx]
            if TreeParser.split_value_is_threshold_dependant(method):
                split_value = split_value(threshold)
            if TreeParser.force_others_on_the_right(method):
                left_child_id, right_child_id = children_right[parent_id], children_left[parent_id]
            else:
                left_child_id, right_child_id = children_left[parent_id], children_right[parent_id]
            tree.add_split_no_siblings(node_type, parent_id, feature, split_value, left_child_id, right_child_id)
            if children_left[left_child_id] > 0:
                ids.append(left_child_id)
            if children_left[right_child_id] > 0:
                ids.append(right_child_id)

    def build_tree(self, df, ranked_features, target="error"):
        features = {}
        for name, settings in self.model_handler.get_preproc_handler().collector_data.get('per_feature').iteritems():
            avg = settings.get('stats').get('average')
            if avg is not None: #TODO AGU: add cases where missing is not replaced by mean
                features[name] = {
                    'mean':  avg
                }
        return InteractiveTree(df, target, ranked_features, features)
