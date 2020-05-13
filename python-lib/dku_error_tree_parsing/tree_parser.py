import numpy as np
from dataiku import Model
from collections import deque
from dku_error_analysis_utils.sm_metadata import get_model_handler
from dku_error_analysis_decision_tree.node import Node, NumericalNode, CategoricalNode
from dku_error_analysis_decision_tree.tree import InteractiveTree
from dku_error_analysis_mpp.error_analyzer import IS_ERROR_COLUMN

class TreeParser(object):
    def __init__(self, model_handler, thresholds, error_tree):
        self.model_handler = model_handler
        self.thresholds = thresholds
        self.error_tree = error_tree

    @staticmethod
    def read_feature(preprocessed_feature):
        split_preprocessed_feature = preprocessed_feature.split(":")
        if len(split_preprocessed_feature) == 1:
            return preprocessed_feature, None
        if split_preprocessed_feature[0] == "dummy":
            return split_preprocessed_feature[1], split_preprocessed_feature[2]
        raise Exception("Feature uses unknown preprocessing") #TODO: add other preprocessing handling

    def build_all_nodes(self, clf_tree, tree, thresholds, feature_list):
        children_left, children_right, features = clf_tree.children_left, clf_tree.children_right, clf_tree.feature
        root_node = Node(0, -1)
        ids = deque()
        tree.add_node(root_node)

        ids.append(0)
        while ids:
            parent_id = ids.popleft()
            feature, split_value = TreeParser.read_feature(feature_list[features[parent_id]])
            left_child_id, right_child_id = children_left[parent_id], children_right[parent_id]
            if self.model_handler.get_per_feature().get(feature).get("type") == "NUMERIC":
                left_child = NumericalNode(left_child_id, parent_id, feature, end=thresholds[parent_id])
                right_child = NumericalNode(right_child_id, parent_id, feature, beginning=thresholds[parent_id])
            else:
                right_child = CategoricalNode(left_child_id, parent_id, feature, [split_value], others=True)
                left_child = CategoricalNode(right_child_id, parent_id, feature, [split_value])
            tree.add_node(left_child)
            tree.add_node(right_child)
            if children_left[left_child_id] > 0:
                ids.append(left_child_id)
            if children_left[right_child_id] > 0:
                ids.append(right_child_id)


    def build_tree(self, df, ranked_features, target=IS_ERROR_COLUMN):
        features = {}
        for name, settings in self.model_handler.get_preproc_handler().collector_data.get('per_feature').iteritems():
            avg = settings.get('stats').get('average')
            if avg is not None:
                features[name] = {
                    'mean':  avg
                }
        return InteractiveTree(df, target, ranked_features, features)
