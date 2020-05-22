import numpy as np
from dataiku import Model
from collections import deque, defaultdict
from dku_error_analysis_utils.sm_metadata import get_model_handler
from dku_error_analysis_decision_tree.node import Node, NumericalNode, CategoricalNode
from dku_error_analysis_decision_tree.tree import InteractiveTree
from dataiku.doctor.preprocessing.dataframe_preprocessing import RescalingProcessor2 as rescaler, QuantileBinSeries as biner
from dku_error_tree_parsing.depreprocessor import descale_numerical_thresholds

class TreeParser(object):
    def __init__(self, model_handler, error_tree, columns):
        self.model_handler = model_handler
        self.error_tree = error_tree
        self.dummy_feature_map = defaultdict(list)
        self.find_others_dummy(columns)
        rescalers, self.bounds = [], {}
        for step in model_handler.get_pipeline().steps:
            if isinstance(step, rescaler):
                rescalers.append(step)
            if isinstance(step, biner):
                for feature, bounds in step.resource.items():
                    self.bounds[feature] = bounds["bounds"]
        self.thresholds = descale_numerical_thresholds(error_tree, columns, rescalers, False)

    def find_others_dummy(self, columns):
        for feature in columns:
            split_preprocessed_feature = feature.split(":")
            if split_preprocessed_feature[0] == "dummy":
                real_name, value = split_preprocessed_feature[1], split_preprocessed_feature[2]
                if value != "__Others__" and value != "N/A":
                    self.dummy_feature_map[real_name].append(value)

    def translate_preprocessed_features(self, feature, parent_id):
        split_preprocessed_feature = feature.split(":")
        #TODO: safer way to check preprocessing?
        if len(split_preprocessed_feature) == 1 or split_preprocessed_feature[0] == "unfold" or split_preprocessed_feature[0] == "impact":
            return Node.TYPES.NUM, feature, self.thresholds[parent_id]
        if split_preprocessed_feature[0] == "dummy":
            real_name, value = split_preprocessed_feature[1], split_preprocessed_feature[2]
            if value == "__Others__":
                return Node.TYPES.CAT, real_name, self.dummy_feature_map[real_name]
            if value == "N/A":
                return Node.TYPES.CAT, real_name, [np.nan]
            return Node.TYPES.CAT, real_name, [value]
        if "binarized" in split_preprocessed_feature[0]:
            return Node.TYPES.NUM, split_preprocessed_feature[1], float(split_preprocessed_feature[3])
        if "quantized" in split_preprocessed_feature[0]:
            real_name = split_preprocessed_feature[1]
            bound_index = int(self.thresholds[parent_id]) + 1
            return Node.TYPES.NUM, real_name, float(self.bounds[real_name][bound_index])
        if "flag" in split_preprocessed_feature[0]:
            return Node.TYPES.CAT, split_preprocessed_feature[1], [np.nan]
        raise ValueError("Feature uses unknown preprocessing")

    def build_all_nodes(self, tree, feature_list):
        children_left, children_right, features = self.error_tree.children_left, self.error_tree.children_right, self.error_tree.feature
        root_node = Node(0, -1)
        ids = deque()
        tree.add_node(root_node)

        ids.append(0)
        while ids:
            parent_id = ids.popleft()
            node_type, feature, split_value = self.translate_preprocessed_features(feature_list[features[parent_id]], parent_id)
            left_child_id, right_child_id = children_left[parent_id], children_right[parent_id]
            if node_type == Node.TYPES.NUM:
                left_child = NumericalNode(left_child_id, parent_id, feature, end=split_value)
                right_child = NumericalNode(right_child_id, parent_id, feature, beginning=split_value)
            else:
                left_child = CategoricalNode(left_child_id, parent_id, feature, values=split_value, others=True)
                right_child = CategoricalNode(right_child_id, parent_id, feature, values=split_value)
            tree.add_node(left_child)
            tree.add_node(right_child)
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
