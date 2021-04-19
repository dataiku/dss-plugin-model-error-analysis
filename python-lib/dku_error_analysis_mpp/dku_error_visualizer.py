# -*- coding: utf-8 -*-
import numpy as np
from graphviz import Source
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from dku_error_analysis_mpp.dku_error_analyzer import DkuErrorAnalyzer
from mealy import _BaseErrorVisualizer, ErrorAnalyzerConstants


import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='Error Analysis Plugin | %(levelname)s - %(message)s')

plt.rc('font', family="sans-serif")
SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 8, 10, 12
plt.rc('axes', titlesize=BIGGER_SIZE, labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE) 
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc("hatch", color="white", linewidth=4)


class DkuErrorVisualizer(_BaseErrorVisualizer):
    """
    ErrorVisualizer provides visual utilities to analyze the error classifier in ErrorAnalyzer and DkuErrorAnalyzer.
    """

    def __init__(self, error_analyzer):

        if not isinstance(error_analyzer, DkuErrorAnalyzer):
            raise NotImplementedError('You need to input a DkuErrorAnalyzer object.')

        super(DkuErrorVisualizer, self).__init__(error_analyzer)

        self._tree = error_analyzer.tree
        self._tree_parser = error_analyzer.tree_parser

    def plot_error_tree(self, size=None):
        """ Plot the graph of the decision tree """

        return Source(self._tree.to_dot_string())

    def plot_feature_distributions_on_leaves(self, leaf_selector=None, top_k_features=ErrorAnalyzerConstants.TOP_K_FEATURES,
                                            show_global=True, show_class=False, rank_leaves_by="total_error_fraction", nr_bins=10, figsize=(15, 10)):
        """ Return plot of error node feature distribution and compare to global baseline """

        leaf_nodes = self._get_ranked_leaf_ids(leaf_selector, rank_leaves_by)
        ranked_features = self._tree.ranked_features[:top_k_features]
        if show_global:
            root_samples = self._tree.get_node(0).samples[0]
            root_hist_data_all_features = {}

        for leaf_id in leaf_nodes:
            for feature in ranked_features:
                feature_name = feature["name"]
                leaf = self._tree.get_node(leaf_id)
                node_summary = 'Leaf {} ({}: {:.3f}'.format(leaf.id, *leaf.probabilities[0])
                if len(leaf.probabilities) > 1:
                    node_summary += ', {}: {:.3f})'.format(*leaf.probabilities[1])
                else:
                    node_summary += ')'
                print(node_summary)

                leaf_stats = self._tree.get_stats(leaf.id, feature_name, nr_bins)
                feature_is_numerical = feature["numerical"]
                bins = leaf_stats["bin_edge"] if feature_is_numerical else leaf_stats["bin_value"]

                if show_global:
                    if feature_name not in root_hist_data_all_features:
                        root_hist_data_all_features[feature_name] = self._tree.get_stats(0, feature_name, min(len(bins), nr_bins))
                    if show_class:
                        root_hist_data = {}
                        for class_value, bar_heights in root_hist_data_all_features[feature_name]["target_distrib"].items():
                            root_hist_data[class_value] = np.array(bar_heights)/float(root_samples)
                    else:
                        root_hist_data, root_prediction = {}, self._tree.get_node(0).prediction
                        root_hist_data[root_prediction] = np.array(root_hist_data_all_features[feature_name]["count"])/float(root_samples)

                leaf_hist_data = {}
                if show_class:
                    for class_value, bar_heights in leaf_stats["target_distrib"].items():
                        leaf_hist_data[class_value] = np.array(bar_heights)/float(leaf.samples[0])
                else:
                    leaf_hist_data = {leaf.prediction: np.array(leaf_stats["count"])/float(leaf.samples[0])}

                x_ticks = _BaseErrorVisualizer._add_new_plot(figsize, bins, feature_name, leaf.id)
                _BaseErrorVisualizer._plot_feature_distribution(x_ticks, feature_is_numerical, leaf_hist_data, root_hist_data if show_global else None)

        plt.show()