from dku_error_analysis_decision_tree.node import Node, NumericalNode, CategoricalNode
from dku_error_analysis_utils import safe_str
import pandas as pd
from collections import deque
from dku_error_analysis_tree_parsing.depreprocessor import descale_numerical_thresholds

from dku_error_analysis_mpp.error_config import ErrorAnalyzerConstants


class InteractiveTree(object):
    """
    A decision tree

    ATTRIBUTES
    df: pd.DataFrame, the dataset

    target: str, the name of the target feature

    nodes: dict, a map from ids to the corresponding nodes in the tree

    features: dict, a map from feature names to the number of usage in the various splits of the tree (useful for recipes) \
            and the mean of the feature if can be treated as numerical

    leaves: set, set of leaves id

    target_values: list, a list of the values the target can take

    sample_method: string, the method used for sampling

    sample_size: positive integer, the number of rows for the sampling
    """
    def __init__(self, df, target, ranked_features, features):
        try:
            df = df.dropna(subset=[target])
            df.loc[:, target] = df.loc[:, target].apply(safe_str) # for classification
        except KeyError:
            raise Exception("The target %s is not one of the columns of the dataset" % target)
        self.target = target
        self.target_values = list(df[target].unique())
        self.features = features
        self.nodes = {}
        self.ranked_features = ranked_features
        self.df = df
        self.bins = {}

    def parse_nodes(self, tree_parser, feature_list, preprocessed_x):
        error_model_tree = tree_parser.error_model.tree_
        thresholds = descale_numerical_thresholds(error_model_tree, feature_list, tree_parser.rescalers, False)
        children_left, children_right, features = error_model_tree.children_left, error_model_tree.children_right, error_model_tree.feature
        root_node = Node(0, -1)
        ids = deque()
        self.add_node(root_node)

        ids.append(0)
        while ids:
            parent_id = ids.popleft()
            feature_idx, threshold = features[parent_id], thresholds[parent_id]
            preprocessed_feature = feature_list[feature_idx]
            split_parameters = tree_parser.get_split_parameters(preprocessed_feature, threshold)

            if split_parameters.uses_preprocessed_feature:
                self.df[split_parameters.feature] = preprocessed_x[:, feature_idx]
            if split_parameters.value is None:
                split_parameters.value = split_parameters.value_func(threshold)
            if split_parameters.force_others_on_right:
                left_child_id, right_child_id = children_right[parent_id], children_left[parent_id]
            else:
                left_child_id, right_child_id = children_left[parent_id], children_right[parent_id]

            self.add_split_no_siblings(split_parameters.node_type, parent_id, split_parameters.feature, split_parameters.value, left_child_id, right_child_id)

            if children_left[left_child_id] > 0:
                ids.append(left_child_id)
            if children_left[right_child_id] > 0:
                ids.append(right_child_id)

    def set_node_info(self, node):
        nr_errors = self.df[self.df[self.target] == ErrorAnalyzerConstants.WRONG_PREDICTION].shape[0]
        filtered_df = self.get_filtered_df(node, self.df)
        probabilities = filtered_df[self.target].value_counts()
        if ErrorAnalyzerConstants.WRONG_PREDICTION in probabilities:
            error = probabilities[ErrorAnalyzerConstants.WRONG_PREDICTION] / float(nr_errors)
        else:
            error = 0
        samples = filtered_df.shape[0]
        sorted_proba = sorted((probabilities/samples).to_dict().items(), key=lambda x: (-x[1], x[0]))
        if samples > 0:
            prediction = sorted_proba[0][0]
        else:
            prediction = None
        if node.id == 0:
            node.set_node_info(samples, samples, sorted_proba, prediction, error)
        else:
            node.set_node_info(samples, self.get_node(0).samples[0], sorted_proba, prediction, error)

    def jsonify(self):
        return {"target": self.target,
                "target_values": self.target_values,
                "features": self.features,
                "nodes": self.jsonify_nodes()}

    def jsonify_nodes(self):
        jsonified_tree = {}
        for key, node in self.nodes.items():
            jsonified_tree[key] = node.jsonify()
        return jsonified_tree

    def add_node(self, node):
        self.nodes[node.id] = node
        parent_node = self.get_node(node.parent_id)
        if parent_node is not None:
            parent_node.children_ids.append(node.id)
        self.set_node_info(node)

    def get_node(self, i):
        return self.nodes.get(i)

    def add_split_no_siblings(self, node_type, parent_id, feature, value, left_node_id, right_child_id):
        parent_node = self.get_node(parent_id)
        if node_type == Node.TYPES.NUM:
            self.add_numerical_split_no_siblings(parent_node, feature, value, left_node_id, right_child_id)
        else:
            self.add_categorical_split_no_siblings(parent_node, feature, value, left_node_id, right_child_id)

    def add_numerical_split_no_siblings(self, parent_node, feature, value, left_node_id, right_child_id):
        new_node_left = NumericalNode(left_node_id, parent_node.id, feature, end=value)
        new_node_right = NumericalNode(right_child_id, parent_node.id, feature, beginning=value)
        self.add_node(new_node_left)
        self.add_node(new_node_right)

    def add_categorical_split_no_siblings(self, parent_node, feature, values, left_node_id, right_child_id):
        left = CategoricalNode(left_node_id, parent_node.id, feature, values)
        right = CategoricalNode(right_child_id, parent_node.id, feature, list(values), others=True)
        self.add_node(left)
        self.add_node(right)

    def get_filtered_df(self, node, df):
        node_id = node.id
        while node_id > 0:
            node = self.get_node(node_id)
            if node.get_type() == Node.TYPES.NUM:
                #TODO: change with ch49216
                df = node.apply_filter(df, self.features.get(node.feature, {"mean": None})["mean"])
            else:
                df = node.apply_filter(df)
            node_id = node.parent_id
        return df

    def get_stats(self, i, col):
        node = self.get_node(i)
        filtered_df = self.get_filtered_df(node, self.df)
        column = filtered_df[col]
        target_column = filtered_df[self.target]
        if col in self.features:
            mean = self.features[col]["mean"]
            bin_labels = self.bins.get(col)
            if bin_labels is None:
                bins, bin_labels = pd.cut(column.fillna(mean), bins=min(10, column.nunique()), include_lowest=True, right=False, retbins=True)
                self.bins[col] = bin_labels
            else:
                bins = pd.cut(column.fillna(mean), bin_labels, right=False)
            return self.get_stats_numerical_node(column, target_column, mean, bins)
        return self.get_stats_categorical_node(column, target_column)

    def get_stats_numerical_node(self, column, target_column, mean, bins):
        stats = []
        if not column.empty:
            full_count = column.shape[0]
            target_grouped = target_column.groupby(bins) #could be simplified but well no time :)
            target_distrib = target_grouped.apply(lambda x: x.value_counts())
            target_distrib = target_distrib / full_count
            col_distrib = target_grouped.count()
            for interval, count in col_distrib.items():
                stats.append({"value": safe_str(interval),
                                "target_distrib": target_distrib[interval].to_dict() if count > 0 else {},
                                "mid": interval.mid,
                                "count": count/float(full_count)})
        return stats

    def get_stats_categorical_node(self, column, target_column):
        stats = []
        if not column.empty:
            full_count = column.shape[0]
            target_grouped = target_column.groupby(column.fillna("No values").apply(safe_str))
            target_distrib = target_grouped.value_counts(dropna=False)
            target_distrib = target_distrib / full_count
            col_distrib = target_grouped.count().sort_values(ascending=False)
            for value in col_distrib.index:
                stats.append({"value": value,
                                "target_distrib": target_distrib[value].to_dict(),
                                "count": col_distrib[value]/float(full_count)})
                if len(stats) == 10:
                    return stats
        return stats
