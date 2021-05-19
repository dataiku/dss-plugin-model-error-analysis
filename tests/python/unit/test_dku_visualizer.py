from mealy import ErrorAnalyzer
from dku_error_analysis_mpp.dku_error_visualizer import DkuErrorVisualizer
from dku_error_analysis_mpp.dku_error_analyzer import DkuErrorAnalyzer
import pytest
import logging

@pytest.fixture
def mocked_nodes(mocker):
    return {
        0: mocker.Mock(id=0, samples=[100], prediction="root_prediction"),
        1: mocker.Mock(id=1, samples=[50], prediction="node_prediction",
                       probabilities=[["W", 1], ["C", .4561]])
    }

@pytest.fixture
def mocked_get_stats():
    def f(has_hist_bins=True):
        def _get_stats(node_id, feature_name, nr_bins, bins=None):
            if node_id == 0:
                if feature_name == "cat":
                    return {
                        "bin_value": ["whatever_cat"],
                        "target_distrib": {"W": [6, 3], "C": [2, 1]},
                        "count": [2, 1]
                    }
                return {
                    "bin_edge": ["whatever_num"],
                    "target_distrib": {"W": [3, 6], "C": [1, 2]},
                    "count": [1, 2]
                }
            if node_id == 1:
                if feature_name == "cat":
                    return {
                        "bin_value": ["works_cat"] if has_hist_bins else [],
                        "target_distrib": {"W": [-6, -3], "C": [-2, -1]},
                        "count": [20, 10]
                    }
                return {
                    "bin_edge": ["works_num"] if has_hist_bins else [],
                    "target_distrib": {"W": [-3, -6], "C": [-1, -2]},
                    "count": [10, 20]
                }
        return _get_stats
    return f

def test_plot_feature_distributions_no_show_global(mocker, mocked_nodes, mocked_get_stats, caplog):
    analyzer = mocker.Mock(spec=DkuErrorAnalyzer)
    mocker.patch.object(analyzer, "_get_ranked_leaf_ids", return_value=[1])
    analyzer.tree.ranked_features = [
        {"name": "num", "numerical": True},
        {"name": "cat", "numerical": False}
    ]
    patched_get_node = mocker.patch.object(analyzer.tree, "get_node", side_effect=lambda x: mocked_nodes[x])
    patched_get_stats = mocker.patch.object(analyzer.tree, "get_stats", side_effect=mocked_get_stats())
    patched_add_plot = mocker.patch("dku_error_analysis_mpp.dku_error_visualizer._BaseErrorVisualizer._add_new_plot", return_value=None)
    patched_plot_distrib = mocker.patch("dku_error_analysis_mpp.dku_error_visualizer._BaseErrorVisualizer._plot_feature_distribution", return_value=None)

    # Not show class
    visualizer = DkuErrorVisualizer(analyzer)
    visualizer.plot_feature_distributions_on_leaves(show_global=False, show_class=False)

    assert patched_add_plot.call_count == 2
    assert patched_plot_distrib.call_count == 2
    assert patched_add_plot.call_args_list[0][0] == ((15, 10), ["works_num"], range(1), "num", "Leaf 1 (W: 1, C: 0.456)")
    x_ticks, feature_is_numerical, leaf_hist_data, root_hist_data = patched_plot_distrib.call_args_list[0][0]
    assert (x_ticks, feature_is_numerical, root_hist_data) == (range(1), True, None)
    assert len(leaf_hist_data) == 1 and (leaf_hist_data["node_prediction"] == [.2, .4]).all()

    assert patched_add_plot.call_args_list[1][0] == ((15, 10), ["works_cat"], range(1), "cat", "Leaf 1 (W: 1, C: 0.456)")
    x_ticks, feature_is_numerical, leaf_hist_data, root_hist_data = patched_plot_distrib.call_args_list[1][0]
    assert (x_ticks, feature_is_numerical, root_hist_data) == (range(1), False, None)
    assert len(leaf_hist_data) == 1 and (leaf_hist_data["node_prediction"] == [.4, .2]).all()

    patched_get_node.assert_called_once_with(1)
    assert patched_get_stats.call_count == 2
    assert patched_get_stats.call_args_list[0][0] == (1, "num", 10)
    assert patched_get_stats.call_args_list[1][0] == (1, "cat", 10)

    # Show class
    visualizer = DkuErrorVisualizer(analyzer)   
    visualizer.plot_feature_distributions_on_leaves(show_global=False, show_class=True, nr_bins=5)

    assert patched_add_plot.call_count == 4
    assert patched_plot_distrib.call_count == 4
    assert patched_add_plot.call_args_list[2][0] == ((15, 10), ["works_num"], range(1), "num", "Leaf 1 (W: 1, C: 0.456)")
    x_ticks, feature_is_numerical, leaf_hist_data, root_hist_data = patched_plot_distrib.call_args_list[2][0]
    assert (x_ticks, feature_is_numerical, root_hist_data) == (range(1), True, None)
    assert len(leaf_hist_data) == 2
    assert (leaf_hist_data["W"] == [-3/50, -3/25]).all()
    assert (leaf_hist_data["C"] == [-.02, -.04]).all()

    assert patched_add_plot.call_args_list[3][0] == ((15, 10), ["works_cat"], range(1), "cat", "Leaf 1 (W: 1, C: 0.456)")
    x_ticks, feature_is_numerical, leaf_hist_data, root_hist_data = patched_plot_distrib.call_args_list[3][0]
    assert (x_ticks, feature_is_numerical, root_hist_data) == (range(1), False, None)
    assert len(leaf_hist_data) == 2 
    assert (leaf_hist_data["W"] == [-3/25, -3/50]).all()
    assert (leaf_hist_data["C"] == [-.04, -.02]).all()

    assert patched_get_node.call_args_list[1][0] == (1,) and patched_get_node.call_count == 2
    assert patched_get_stats.call_count == 4
    assert patched_get_stats.call_args_list[2][0] == (1, "num", 5)
    assert patched_get_stats.call_args_list[3][0] == (1, "cat", 5)

    # No bins
    visualizer = DkuErrorVisualizer(analyzer)   
    caplog.set_level(logging.INFO)
    patched_get_stats = mocker.patch.object(analyzer.tree, "get_stats", side_effect=mocked_get_stats(False))
    visualizer.plot_feature_distributions_on_leaves(show_global=False, show_class=True, top_k_features=1, figsize=(30, 40))

    assert patched_add_plot.call_count == 5
    assert patched_plot_distrib.call_count == 5
    assert patched_add_plot.call_args_list[4][0] == ((30, 40), [], range(0), "num", "Leaf 1 (W: 1, C: 0.456)")
    x_ticks, feature_is_numerical, leaf_hist_data, root_hist_data = patched_plot_distrib.call_args_list[4][0]
    assert (x_ticks, feature_is_numerical, leaf_hist_data, root_hist_data) == (range(0), True, None, None)

    log = caplog.records[-1]
    assert log.levelname == "INFO"
    assert log.msg == "No values for the feature num at the leaf 1"

    assert patched_get_node.call_args_list[2][0] == (1,) and patched_get_node.call_count == 3
    patched_get_stats.assert_called_once_with(1, "num", 10)

def test_plot_feature_distributions_show_global(mocker, mocked_nodes, mocked_get_stats, caplog):
    analyzer = mocker.Mock(spec=DkuErrorAnalyzer)
    mocker.patch.object(analyzer, "_get_ranked_leaf_ids", return_value=[1])
    analyzer.tree.ranked_features = [
        {"name": "num", "numerical": True},
        {"name": "cat", "numerical": False}
    ]
    patched_get_node = mocker.patch.object(analyzer.tree, "get_node", side_effect=lambda x: mocked_nodes[x])
    patched_get_stats = mocker.patch.object(analyzer.tree, "get_stats", side_effect=mocked_get_stats())
    patched_add_plot = mocker.patch("dku_error_analysis_mpp.dku_error_visualizer._BaseErrorVisualizer._add_new_plot", return_value=None)
    patched_plot_distrib = mocker.patch("dku_error_analysis_mpp.dku_error_visualizer._BaseErrorVisualizer._plot_feature_distribution", return_value=None)

    # Not show class
    visualizer = DkuErrorVisualizer(analyzer)
    visualizer.plot_feature_distributions_on_leaves(show_global=True, show_class=False)

    assert patched_add_plot.call_count == 2
    assert patched_plot_distrib.call_count == 2
    assert patched_add_plot.call_args_list[0][0] == ((15, 10), ["works_num"], range(1), "num", "Leaf 1 (W: 1, C: 0.456)")
    x_ticks, feature_is_numerical, leaf_hist_data, root_hist_data = patched_plot_distrib.call_args_list[0][0]
    assert (x_ticks, feature_is_numerical) == (range(1), True)
    assert len(leaf_hist_data) == 1 and (leaf_hist_data["node_prediction"] == [.2, .4]).all()
    assert len(root_hist_data) == 1 and (root_hist_data["root_prediction"] == [.01, .02]).all()

    assert patched_add_plot.call_args_list[1][0] == ((15, 10), ["works_cat"], range(1), "cat", "Leaf 1 (W: 1, C: 0.456)")
    x_ticks, feature_is_numerical, leaf_hist_data, root_hist_data = patched_plot_distrib.call_args_list[1][0]
    assert (x_ticks, feature_is_numerical) == (range(1), False)
    assert len(leaf_hist_data) == 1 and (leaf_hist_data["node_prediction"] == [.4, .2]).all()
    assert len(root_hist_data) == 1 and (root_hist_data["root_prediction"] == [.02, .01]).all()

    assert patched_get_node.call_count == 5
    assert patched_get_node.call_args_list[0][0] == (1,)
    assert patched_get_node.call_args_list[1][0] == (0,)
    assert patched_get_node.call_args_list[2][0] == (0,)
    assert patched_get_node.call_args_list[3][0] == (0,)
    assert patched_get_node.call_args_list[4][0] == (0,)

    assert patched_get_stats.call_count == 4
    assert patched_get_stats.call_args_list[0][0] == (1, "num", 10)
    assert patched_get_stats.call_args_list[1][0] == (0, "num", 10, ["works_num"])
    assert patched_get_stats.call_args_list[2][0] == (1, "cat", 10)
    assert patched_get_stats.call_args_list[3][0] == (0, "cat", 10, ["works_cat"])

    # Show class
    visualizer = DkuErrorVisualizer(analyzer)   
    visualizer.plot_feature_distributions_on_leaves(show_global=True, show_class=True, nr_bins=5)

    assert patched_add_plot.call_count == 4
    assert patched_plot_distrib.call_count == 4
    assert patched_add_plot.call_args_list[2][0] == ((15, 10), ["works_num"], range(1), "num", "Leaf 1 (W: 1, C: 0.456)")
    x_ticks, feature_is_numerical, leaf_hist_data, root_hist_data = patched_plot_distrib.call_args_list[2][0]
    assert (x_ticks, feature_is_numerical) == (range(1), True)
    assert len(leaf_hist_data) == 2
    assert (leaf_hist_data["W"] == [-3/50, -3/25]).all()
    assert (leaf_hist_data["C"] == [-.02, -.04]).all()
    assert len(root_hist_data) == 2
    assert (root_hist_data["W"] == [.03, .06]).all()
    assert (root_hist_data["C"] == [.01, .02]).all()

    assert patched_add_plot.call_args_list[3][0] == ((15, 10), ["works_cat"], range(1), "cat", "Leaf 1 (W: 1, C: 0.456)")
    x_ticks, feature_is_numerical, leaf_hist_data, root_hist_data = patched_plot_distrib.call_args_list[3][0]
    assert (x_ticks, feature_is_numerical) == (range(1), False)
    assert len(leaf_hist_data) == 2 
    assert (leaf_hist_data["W"] == [-3/25, -3/50]).all()
    assert (leaf_hist_data["C"] == [-.04, -.02]).all()
    assert len(root_hist_data) == 2
    assert (root_hist_data["W"] == [.06, .03]).all()
    assert (root_hist_data["C"] == [.02, .01]).all()

    assert patched_get_node.call_count == 8
    assert patched_get_node.call_args_list[5][0] == (1,)
    assert patched_get_node.call_args_list[6][0] == (0,)
    assert patched_get_node.call_args_list[7][0] == (0,)

    assert patched_get_stats.call_count == 8
    assert patched_get_stats.call_args_list[4][0] == (1, "num", 5)
    assert patched_get_stats.call_args_list[5][0] == (0, "num", 5, ["works_num"])
    assert patched_get_stats.call_args_list[6][0] == (1, "cat", 5)
    assert patched_get_stats.call_args_list[7][0] == (0, "cat", 5, ["works_cat"])

    # No bins
    visualizer = DkuErrorVisualizer(analyzer)   
    caplog.set_level(logging.INFO)
    patched_get_stats = mocker.patch.object(analyzer.tree, "get_stats", side_effect=mocked_get_stats(False))
    visualizer.plot_feature_distributions_on_leaves(show_global=True, show_class=False, top_k_features=1, figsize=(30, 40))

    assert patched_add_plot.call_count == 5
    assert patched_plot_distrib.call_count == 5
    assert patched_add_plot.call_args_list[4][0] == ((30, 40), ["whatever_num"], range(1), "num", "Leaf 1 (W: 1, C: 0.456)")
    x_ticks, feature_is_numerical, leaf_hist_data, root_hist_data = patched_plot_distrib.call_args_list[4][0]
    assert (x_ticks, feature_is_numerical, leaf_hist_data) == (range(1), True, None)
    assert len(root_hist_data) == 1 and (root_hist_data["root_prediction"] == [.01, .02]).all()

    log = caplog.records[-1]
    assert log.levelname == "INFO"
    assert log.msg == "No values for the feature num at the leaf 1"

    assert patched_get_node.call_count == 11
    assert patched_get_node.call_args_list[8][0] == (1,)
    assert patched_get_node.call_args_list[9][0] == (0,)
    assert patched_get_node.call_args_list[10][0] == (0,)

    assert patched_get_stats.call_count == 2
    assert patched_get_stats.call_args_list[0][0] == (1, "num", 10)
    assert patched_get_stats.call_args_list[1][0] == (0, "num", 10, [])

def test_failed_init(mocker):
    with pytest.raises(TypeError, match="You need to input a DkuErrorAnalyzer object."):
        DkuErrorVisualizer(mocker.Mock(spec=ErrorAnalyzer))
