from dku_error_analysis_decision_tree.node import Node, NumericalNode, CategoricalNode
from dku_error_analysis_utils import safe_str, ErrorAnalyzerConstants
import pandas as pd
from collections import deque
from dku_error_analysis_tree_parsing.depreprocessor import descale_numerical_thresholds

class InteractiveTree(object):
    """
    A decision tree

    ATTRIBUTES
    df: pd.DataFrame, the dataset

    target: str, the name of the target feature

    nodes: dict, a map from ids to the corresponding nodes in the tree

    num_features: dict, a map from feature names to the mean of the feature if can be treated as numerical

    leaves: set, set of leaves id

    target_values: list, a list of the values the target can take

    """
    def __init__(self, df, target, ranked_features, num_features):
        try:
            df = df.dropna(subset=[target])
            df.loc[:, target] = df.loc[:, target].apply(safe_str) # for classification
        except KeyError:
            raise Exception("The target %s is not one of the columns of the dataset" % target)
        self.target = target
        self.target_values = list(df[target].unique())
        self.num_features = num_features # TODO: remove this arg (see handling of missing features)
        self.nodes = {}
        self.ranked_features = ranked_features
        self.df = df
        self.bins = {}
        self.leaves = set()

    def to_dot_string(self):
        dot_str = 'digraph Tree {\nnode [shape=box, style="filled, rounded", color="black", fontname=helvetica] ;\n'
        dot_str += 'edge [fontname=helvetica] ;\ngraph [ranksep=equally, splines=polyline] ;\n'
        ids = deque()
        ids.append(0)

        while ids:
            node = self.get_node(ids.popleft())
            dot_str += node.to_dot_string() + "\n"
            if node.parent_id >= 0:
                dot_str += '{} -> {} ;\n'.format(node.parent_id, node.id)
            ids += node.children_ids
        dot_str += '{rank=same ; '+ '; '.join(map(safe_str, self.leaves)) + '} ;\n'
        dot_str += "}"
        return dot_str

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
            else:
                self.leaves.add(left_child_id)
            if children_left[right_child_id] > 0:
                ids.append(right_child_id)
            else:
                self.leaves.add(right_child_id)

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

    def jsonify_nodes(self):
        jsonified_tree = {}
        for key, node in self.nodes.items():
            jsonified_tree[str(key)] = node.jsonify()
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
                # TODO: change with ch49216
                df = node.apply_filter(df, self.num_features.get(node.feature, {"mean": None})["mean"])
            else:
                df = node.apply_filter(df)
            node_id = node.parent_id
        return df

    def get_stats(self, i, col, nr_bins=10):
        node = self.get_node(i)
        filtered_df = self.get_filtered_df(node, self.df)
        column = filtered_df[col]
        target_column = filtered_df[self.target]
        if col in self.num_features:
            bins = self.bins.get(col)
            if bins is None:
                mean = self.num_features[col]["mean"]
                bins, bin_edges = pd.cut(self.df[col].fillna(mean), bins=min(nr_bins, self.df[col].nunique()), retbins=True, include_lowest=True, right=False)
                if i > 0:
                    bins = pd.cut(column.fillna(mean), bins=bin_edges, right=False)
            return self.get_stats_numerical_node(column, target_column, bins)
        return self.get_stats_categorical_node(column, target_column, nr_bins if i > 0 else -1)

    def get_stats_numerical_node(self, column, target_column, bins):
        stats = {
            "bin_edge": [],
            "target_distrib": {ErrorAnalyzerConstants.WRONG_PREDICTION: [], ErrorAnalyzerConstants.CORRECT_PREDICTION: []},
            "mid": [],
            "count": []
        }
        if not column.empty:
            target_grouped = target_column.groupby(bins)
            target_distrib = target_grouped.apply(lambda x: x.value_counts())
            full_count = column.shape[0]
            target_distrib = target_distrib / full_count
            col_distrib = target_grouped.count()
            for interval, count in col_distrib.items():
                target_distrib_dict = target_distrib[interval].to_dict() if count > 0 else {}
                stats["target_distrib"][ErrorAnalyzerConstants.WRONG_PREDICTION].append(target_distrib_dict.get(ErrorAnalyzerConstants.WRONG_PREDICTION, 0))
                stats["target_distrib"][ErrorAnalyzerConstants.CORRECT_PREDICTION].append(target_distrib_dict.get(ErrorAnalyzerConstants.CORRECT_PREDICTION, 0))
                stats["count"].append(count)
                stats["mid"].append(interval.mid)
                if len(stats["bin_edge"]) == 0:
                    stats["bin_edge"].append(interval.left)
                stats["bin_edge"].append(interval.right)
        return stats

    def get_stats_categorical_node(self, column, target_column, nr_bins):
        stats = {
            "bin_value": [],
            "target_distrib": {ErrorAnalyzerConstants.WRONG_PREDICTION: [], ErrorAnalyzerConstants.CORRECT_PREDICTION: []},
            "count": []
        }
        if not column.empty:
            full_count = column.shape[0]
            target_grouped = target_column.groupby(column.fillna("No values").apply(safe_str))
            target_distrib = target_grouped.value_counts(dropna=False)
            target_distrib = target_distrib / full_count
            col_distrib = target_grouped.count().sort_values(ascending=False)
            for value in col_distrib.index:
                target_distrib_dict = target_distrib[value].to_dict()
                stats["target_distrib"][ErrorAnalyzerConstants.WRONG_PREDICTION].append(target_distrib_dict.get(ErrorAnalyzerConstants.WRONG_PREDICTION, 0))
                stats["target_distrib"][ErrorAnalyzerConstants.CORRECT_PREDICTION].append(target_distrib_dict.get(ErrorAnalyzerConstants.CORRECT_PREDICTION, 0))
                stats["count"].append(col_distrib[value])
                stats["bin_value"].append(value)
                if len(stats["bin_value"]) == nr_bins:
                    return stats
        return stats
