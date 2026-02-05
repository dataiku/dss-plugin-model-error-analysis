# -*- coding: utf-8 -*-
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from graphviz import Source

from mealy_local.error_analyzer_constants import ErrorAnalyzerConstants
from mealy_local.error_analyzer import ErrorAnalyzer

plt.rc('font', family="sans-serif")
SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 8, 10, 12
plt.rc('axes', titlesize=BIGGER_SIZE, labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc("hatch", color="white", linewidth=4)


class _BaseErrorVisualizer(object):
    def __init__(self, error_analyzer):
        if not isinstance(error_analyzer, ErrorAnalyzer):
            raise TypeError('You need to input an ErrorAnalyzer object.')

        self._error_analyzer = error_analyzer

        self._get_ranked_leaf_ids = lambda leaf_selector, rank_by: \
            error_analyzer._get_ranked_leaf_ids(leaf_selector, rank_by)

    @staticmethod
    def _plot_histograms(hist_data, label, **params):
        bottom = None
        for class_value in [ErrorAnalyzerConstants.CORRECT_PREDICTION, ErrorAnalyzerConstants.WRONG_PREDICTION]:
            bar_heights = hist_data.get(class_value)
            if bar_heights is not None:
                plt.bar(height=bar_heights,
                        label="{} ({})".format(class_value, label),
                        edgecolor="white",
                        linewidth=1,
                        color=ErrorAnalyzerConstants.ERROR_TREE_COLORS[class_value],
                        bottom=bottom,
                        **params)
                bottom = bar_heights

    @staticmethod
    def _add_new_plot(figsize, bins, x_ticks, feature_name, suptitle):
        plt.figure(figsize=figsize)
        plt.xticks(x_ticks, rotation=90)
        plt.gca().set_xticklabels(labels=bins)
        plt.ylabel('Proportion of samples')
        plt.title('Distribution of {}'.format(feature_name))
        plt.suptitle(suptitle)

    @staticmethod
    def _plot_feature_distribution(x_ticks, feature_is_numerical, leaf_data, root_data=None):
        width, x = 1.0, x_ticks
        align = "edge"
        if root_data is not None:
            width /= 2
            if feature_is_numerical:
                x = x_ticks[1:]
            _BaseErrorVisualizer._plot_histograms(root_data, label="global data", x=x, hatch="///",
                                                  width=-width, align=align)
        if leaf_data is not None:
            if feature_is_numerical:
                x = x_ticks[:-1]
            elif root_data is None:
                align = "center"
            _BaseErrorVisualizer._plot_histograms(leaf_data, label="leaf data", x=x,
                                                  align=align, width=width)
        plt.legend()
        plt.pause(0.05)
