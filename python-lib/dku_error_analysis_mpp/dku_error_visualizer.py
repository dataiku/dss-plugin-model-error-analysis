# -*- coding: utf-8 -*-
import numpy as np
from graphviz import Source
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from dku_error_analysis_mpp.dku_error_analyzer import DkuErrorAnalyzer
from mealy import _BaseErrorVisualizer, ErrorAnalyzerConstants
from dku_error_analysis_utils import safe_str, format_float

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
            raise TypeError('You need to input a DkuErrorAnalyzer object.')

        super(DkuErrorVisualizer, self).__init__(error_analyzer)

        self._tree = error_analyzer.tree

    def plot_error_tree(self, size=(50, 50)):
        """ Plot the graph of the decision tree
        Args:
            size (tuple): Size of the output plot as (width, length), in inches.

        """

        return Source(self._tree.to_dot_string(size))

    def plot_feature_distributions_on_leaves(self, leaf_selector=None, top_k_features=ErrorAnalyzerConstants.TOP_K_FEATURES,
                                            show_global=True, show_class=False, rank_leaves_by="total_error_fraction", nr_bins=10, figsize=(15, 10)):
        """ Return plot of error node feature distribution and compare to global baseline """

        leaf_nodes = self._get_ranked_leaf_ids(leaf_selector, rank_leaves_by)
        ranked_features = self._tree.ranked_features[:top_k_features]

        for leaf_id in leaf_nodes:
            leaf = self._tree.get_node(leaf_id)
            suptitle = 'Leaf {} ({}: {}'.format(leaf.id, leaf.probabilities[0][0], format_float(leaf.probabilities[0][1], 3))
            suptitle += ', {}: {})'.format(leaf.probabilities[1][0], format_float(leaf.probabilities[1][1], 3))
            for feature in ranked_features:
                feature_name = feature["name"]

                leaf_stats = self._tree.get_stats(leaf.id, feature_name, nr_bins)
                feature_is_numerical = feature["numerical"]
                bins = leaf_stats["bin_edge"] if feature_is_numerical else leaf_stats["bin_value"]

                if show_global:
                    root_samples = self._tree.get_node(0).samples[0]
                    root_stats = self._tree.get_stats(0, feature_name, nr_bins, set(bins)) # TODO: optimize
                    if show_class:
                        root_hist_data = {}
                        for class_value, bar_heights in root_stats["target_distrib"].items():
                            root_hist_data[class_value] = np.array(bar_heights)/root_samples
                    else:
                        root_hist_data, root_prediction = {}, self._tree.get_node(0).prediction
                        root_hist_data[root_prediction] = np.array(root_stats["count"])/root_samples
                else:
                    root_hist_data = None

                if bins:
                    leaf_hist_data = {}
                    if show_class:
                        for class_value, bar_heights in leaf_stats["target_distrib"].items():
                            leaf_hist_data[class_value] = np.array(bar_heights)/leaf.samples[0]
                    else:
                        leaf_hist_data = {leaf.prediction: np.array(leaf_stats["count"])/leaf.samples[0]}
                else:
                    leaf_hist_data = None
                    logger.info("No values for the feature {} at the leaf {}".format(feature_name, leaf.id))
                    if show_global:
                        bins = root_stats["bin_edge"] if feature_is_numerical else root_stats["bin_value"]

                x_ticks = range(len(bins))
                _BaseErrorVisualizer._add_new_plot(figsize, bins, x_ticks, feature_name, suptitle)
                _BaseErrorVisualizer._plot_feature_distribution(x_ticks, feature_is_numerical, leaf_hist_data, root_hist_data)

        plt.show()
