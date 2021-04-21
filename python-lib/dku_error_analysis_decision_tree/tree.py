from dku_error_analysis_decision_tree.node import Node, NumericalNode, CategoricalNode
from dku_error_analysis_utils import safe_str
from mealy import ErrorAnalyzerConstants

import pandas as pd
from collections import deque

class InteractiveTree(object):
    """
    A decision tree

    ATTRIBUTES
    df: pd.DataFrame, the dataset

    target: str, the name of the target feature

    nodes: dict, a map from ids to the corresponding nodes in the tree

    num_features: dict, a map from feature names to the mean of the feature if can be treated as numerical

    ranked_features: list of dict with three keys (name - name of the feature; numerical - whether the feature is numerical; rank - the feature importance)

    leaves: set, set of leaves id

    """
    def __init__(self, df, target, ranked_features, num_features):
        self.df = df.dropna(subset=[target]) # TODO
        self.target = target
        self.num_features = num_features
        self.nodes = {0: Node(0, -1)}
        self.ranked_features = []
        for idx, ranked_feature in enumerate(ranked_features):
            self.ranked_features.append({
                "rank": idx,
                "name": ranked_feature,
                "numerical": ranked_feature in num_features
            })
        self.bins = {}
        self.leaves = set()

    def to_dot_string(self, size=(50, 50)):
        dot_str = 'digraph Tree {{\n size="{0},{1}!";\nnode [shape=box, style="filled, rounded", color="black", fontname=helvetica] ;\n'.format(size[0], size[1])
        dot_str += 'edge [fontname=helvetica] ;\ngraph [ranksep=equally, splines=polyline] ;\n'
        ids = deque()
        ids.append(0)

        while ids:
            node = self.get_node(ids.popleft())
            dot_str += node.to_dot_string() + "\n"
            if node.parent_id >= 0:
                edge_width = max(1, ErrorAnalyzerConstants.GRAPH_MAX_EDGE_WIDTH * node.global_error)
                dot_str += '{} -> {} [penwidth={}];\n'.format(node.parent_id, node.id, edge_width)
            ids += node.children_ids
        dot_str += '{rank=same ; '+ '; '.join(map(safe_str, self.leaves)) + '} ;\n'
        dot_str += "}"
        return dot_str

    def set_node_info(self, node_id, samples, class_samples):
        sorted_class_samples = sorted(class_samples.items(), key=lambda x: -x[1])
        if samples > 0:
            prediction = sorted_class_samples[0][0]
        else:
            prediction = None

        node = self.get_node(node_id)
        if node_id == 0:
            node.set_node_info(samples, samples, sorted_class_samples, prediction, 1)
        else:
            root = self.get_node(0)
            global_error = class_samples[ErrorAnalyzerConstants.WRONG_PREDICTION] / root.local_error[1]
            node.set_node_info(samples, root.samples[0], sorted_class_samples, prediction, global_error)

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

    def get_node(self, i):
        return self.nodes.get(i)

    def add_split_no_siblings(self, node_type, parent_id, feature, value, left_node_id, right_child_id):
        if node_type == Node.TYPES.NUM:
            left = NumericalNode(left_node_id, parent_id, feature, end=value)
            right = NumericalNode(right_child_id, parent_id, feature, beginning=value)
        else:
            left = CategoricalNode(left_node_id, parent_id, feature, value)
            right = CategoricalNode(right_child_id, parent_id, feature, list(value), others=True) #TODO: why the list
        self.add_node(left)
        self.add_node(right)

    def get_filtered_df(self, node, df):
        node_id = node.id
        while node_id > 0:
            node = self.get_node(node_id)
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
                bins, bin_edges = pd.cut(self.df[col], bins=min(nr_bins, self.df[col].nunique()), retbins=True, include_lowest=True, right=False)
                if i > 0:
                    bins = pd.cut(column, bins=bin_edges, right=False)
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
            target_grouped = target_column.groupby(column.fillna("No values").apply(safe_str)) # TODO: see CH card on missing values
            target_distrib = target_grouped.value_counts(dropna=False)
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
