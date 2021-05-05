from mealy import ErrorAnalyzerConstants
import pandas as pd
import numpy as np
from dku_error_analysis_decision_tree.tree import InteractiveTree
from dku_error_analysis_decision_tree.node import Node
import pytest

@pytest.fixture
def create_tree():
    df = pd.DataFrame([
        [1,      5.5, "x", "n",    "A"],
        [2,      7.7, "y", np.nan, "A"],
        [np.nan, 7,   "z", np.nan, "B"],
        [3,      1.2, "z", "n",    "B"],
        [4,      7.1, "z", np.nan, "C"],
        [5,      .4,  "x", "p",    "A"],
        [6,      8,   "z", np.nan, "A"],
        [7,      5.5, "y", "p",    "B"],
        [8,      1.5, "z", "n",    "B"],
        [9,      3,   "y", "n",    "C"],
        [10,     7.5, "x", np.nan, "B"],
        [11,     6,   "x", np.nan, "B"]
    ], columns=("num_1", "num_2", "cat_1", "cat_2", "target"))
    return lambda: InteractiveTree(df, "target", ["cat_1", "num_2", "cat_2", "num_1"], {"num_1", "num_2"})

@pytest.fixture
def target():
    return pd.Series([
        ErrorAnalyzerConstants.WRONG_PREDICTION,
        ErrorAnalyzerConstants.CORRECT_PREDICTION,
        ErrorAnalyzerConstants.CORRECT_PREDICTION,
        ErrorAnalyzerConstants.CORRECT_PREDICTION,
        ErrorAnalyzerConstants.WRONG_PREDICTION
    ])

@pytest.fixture
def num_column():
    def col(full=True):
        return pd.Series(
        pd.Categorical([
            pd.Interval(0, 2, "left"),
            pd.Interval(8, 10, "left"),
            pd.Interval(4, 6, "left") if full else np.nan,
            pd.Interval(0, 2, "left"),
            pd.Interval(6, 8, "left")
        ], categories=[
            pd.Interval(0, 2, "left"),
            pd.Interval(2, 4, "left"),
            pd.Interval(4, 6, "left"),
            pd.Interval(6, 8, "left"),
            pd.Interval(8, 10, "left")
    ]))
    return col

@pytest.fixture
def cat_column():
    def col(full=True):
        return pd.Series(["C", "B", "B" if full else np.nan, "B", "Q"])
    return col

def test_on_add_splits(create_tree):
    # Add numerical split
    tree = create_tree()
    tree.add_split_no_siblings(Node.TYPES.NUM, 0, "num_2", .9, 1, 2)
    assert tree.leaves == {1, 2}
    assert tree.get_node(0).children_ids == [1, 2]

    left_node = tree.get_node(1)
    assert left_node.feature == "num_2"
    assert left_node.end == .9
    assert left_node.parent_id == 0

    right_node = tree.get_node(2)
    assert right_node.feature == "num_2"
    assert right_node.beginning == .9
    assert right_node.parent_id == 0

    # Add failing numerical split
    tree = create_tree()
    with pytest.raises(ValueError):
        tree.add_split_no_siblings(Node.TYPES.NUM, 0, "num_2", None, 1, 2)

    # Add categorical split
    tree = create_tree()
    tree.add_split_no_siblings(Node.TYPES.CAT, 0, "cat_1", ["n"], 1, 2)
    assert tree.leaves == {1, 2}
    assert tree.get_node(0).children_ids == [1, 2]

    left_node = tree.get_node(1)
    assert left_node.feature == "cat_1"
    assert left_node.values == ["n"]
    assert left_node.parent_id == 0
    assert not left_node.others

    right_node = tree.get_node(2)
    assert right_node.feature == "cat_1"
    assert right_node.values == ["n"]
    assert right_node.parent_id == 0
    assert right_node.others

    # Add failing numerical split
    tree = create_tree()
    with pytest.raises(ValueError):
        tree.add_split_no_siblings(Node.TYPES.CAT, 0, "cat_1", None, 1, 2)

def test_get_stats_numerical_node(target, num_column):
    # Check nominal case
    binned_column = num_column()

    stats = InteractiveTree.get_stats_numerical_node(binned_column, target)
    assert stats["bin_edge"] == [0, 2, 4, 6, 8, 10]
    assert stats["mid"] == [1, 3, 5, 7, 9]
    assert stats["count"] == [2, 0, 1, 1, 1]
    assert stats["target_distrib"][ErrorAnalyzerConstants.WRONG_PREDICTION] == [1, 0, 0, 1, 0]
    assert stats["target_distrib"][ErrorAnalyzerConstants.CORRECT_PREDICTION] == [1, 0, 1, 0, 1]

    # Check with empty column
    stats = InteractiveTree.get_stats_numerical_node(pd.Series(dtype=int), target)
    assert stats["bin_edge"] == []
    assert stats["mid"] == []
    assert stats["count"] == []
    assert stats["target_distrib"][ErrorAnalyzerConstants.WRONG_PREDICTION] == []
    assert stats["target_distrib"][ErrorAnalyzerConstants.CORRECT_PREDICTION] == []

    # Check with NaN
    binned_column_nan = num_column(False)

    stats = InteractiveTree.get_stats_numerical_node(binned_column_nan, target)
    assert stats["bin_edge"] == [0, 2, 4, 6, 8, 10]
    assert stats["mid"] == [1, 3, 5, 7, 9]
    assert stats["count"] == [2, 0, 0, 1, 1]
    assert stats["target_distrib"][ErrorAnalyzerConstants.WRONG_PREDICTION] == [1, 0, 0, 1, 0]
    assert stats["target_distrib"][ErrorAnalyzerConstants.CORRECT_PREDICTION] == [1, 0, 0, 0, 1]

def test_get_stats_categorical_node(target, cat_column):
    # Check nominal case
    binned_column = cat_column()
    stats = InteractiveTree.get_stats_categorical_node(binned_column, target, 10, None)
    assert stats["bin_value"] == ["B", "Q", "C"]
    assert stats["count"] == [3, 1, 1]
    assert stats["target_distrib"][ErrorAnalyzerConstants.WRONG_PREDICTION] == [0, 1, 1]
    assert stats["target_distrib"][ErrorAnalyzerConstants.CORRECT_PREDICTION] == [3, 0, 0]

    # Check nominal case - less bins
    binned_column = cat_column()
    stats = InteractiveTree.get_stats_categorical_node(binned_column, target, 2, None)
    assert stats["bin_value"] == ["B", "Q"]
    assert stats["count"] == [3, 1]
    assert stats["target_distrib"][ErrorAnalyzerConstants.WRONG_PREDICTION] == [0, 1]
    assert stats["target_distrib"][ErrorAnalyzerConstants.CORRECT_PREDICTION] == [3, 0]

    # Check nominal case - enforced bins
    binned_column = cat_column()
    stats = InteractiveTree.get_stats_categorical_node(binned_column, target, 1, set({"C", "Q"}))
    assert stats["bin_value"] == ["Q", "C"]
    assert stats["count"] == [1, 1]
    assert stats["target_distrib"][ErrorAnalyzerConstants.WRONG_PREDICTION] == [1, 1]
    assert stats["target_distrib"][ErrorAnalyzerConstants.CORRECT_PREDICTION] == [0, 0]

    # Check with nan
    binned_column = cat_column(False)
    stats = InteractiveTree.get_stats_categorical_node(binned_column, target, 10, None)
    assert stats["bin_value"] == ["B", "Q", "No values", "C"]
    assert stats["count"] == [2, 1, 1, 1]
    assert stats["target_distrib"][ErrorAnalyzerConstants.WRONG_PREDICTION] == [0, 1, 0, 1]
    assert stats["target_distrib"][ErrorAnalyzerConstants.CORRECT_PREDICTION] == [2, 0, 1, 0]

    # Check with nan - less bins
    binned_column = cat_column(False)
    stats = InteractiveTree.get_stats_categorical_node(binned_column, target, 3, None)
    assert stats["bin_value"] == ["B", "Q", "No values"]
    assert stats["count"] == [2, 1, 1]
    assert stats["target_distrib"][ErrorAnalyzerConstants.WRONG_PREDICTION] == [0, 1, 0]
    assert stats["target_distrib"][ErrorAnalyzerConstants.CORRECT_PREDICTION] == [2, 0, 1]

    # Check with nan - less bins
    binned_column = cat_column(False)
    stats = InteractiveTree.get_stats_categorical_node(binned_column, target, 1, set({"Q", "B"}))
    assert stats["bin_value"] == ["B", "Q"]
    assert stats["count"] == [2, 1]
    assert stats["target_distrib"][ErrorAnalyzerConstants.WRONG_PREDICTION] == [0, 1]
    assert stats["target_distrib"][ErrorAnalyzerConstants.CORRECT_PREDICTION] == [2, 0]

    # Check with empty column
    stats = InteractiveTree.get_stats_categorical_node(pd.Series(dtype=object), target, 10, None)
    assert stats["bin_value"] == []
    assert stats["count"] == []
    assert stats["target_distrib"][ErrorAnalyzerConstants.WRONG_PREDICTION] == []
    assert stats["target_distrib"][ErrorAnalyzerConstants.CORRECT_PREDICTION] == []

def test_get_stats(create_tree, mocker):
    # Retrieving stats for categorical features
    tree = create_tree()
    spy = mocker.spy(InteractiveTree, 'get_stats_categorical_node')
    tree.get_stats(0, "cat_1", 7)
    cargs = spy.call_args[0]
    pd.testing.assert_series_equal(cargs[0], pd.Series(["x","y","z","z","z","x","z","y","z","y","x","x"], name="cat_1"))
    pd.testing.assert_series_equal(cargs[1], pd.Series(["A","A","B","B","C","A","A","B","B","C","B","B"], name="target"))
    assert cargs[2] == 7
    assert cargs[3] is None

    # Retrieving stats for numerical features - no bins
    tree = create_tree()
    spy = mocker.spy(InteractiveTree, 'get_stats_numerical_node')
    spy_cut = mocker.spy(pd, 'cut')
    tree.get_stats(0, "num_1", 10)
    cargs = spy.call_args[0]
    bins = pd.Series(pd.Categorical([
        pd.Interval(1.0,  2.0,  "left"),
        pd.Interval(2.0,  3.0,  "left"),
        np.nan,
        pd.Interval(3.0,  4.0,  "left"),
        pd.Interval(4.0,  5.0,  "left"),
        pd.Interval(5.0,  6.0,  "left"),
        pd.Interval(6.0,  7.0,  "left"),
        pd.Interval(7.0,  8.0,  "left"),
        pd.Interval(8.0,  9.0,  "left"),
        pd.Interval(9.0,  10.0, "left"),
        pd.Interval(10.0, 11.01, "left"),
        pd.Interval(10.0, 11.01, "left")
    ], ordered=True), name="num_1")
    pd.testing.assert_series_equal(cargs[0], bins)
    pd.testing.assert_series_equal(cargs[1], pd.Series(["A","A","B","B","C","A","A","B","B","C","B","B"], name="target"))
    assert (tree.bin_edges["num_1"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.01]).all()
    assert spy_cut.call_count == 2

    # Retrieving stats for numerical features - with bins
    tree = create_tree()
    bin_edges = np.array([1.0, 5.0, 11.01])
    tree.bin_edges["num_1"] = bin_edges
    spy = mocker.spy(InteractiveTree, 'get_stats_numerical_node')
    tree.get_stats(0, "num_1", 8, pd.Series(pd.Categorical([
        pd.Interval(1.0, 3.0, "left"),
        pd.Interval(3.0, 11.01, "left")],
    ordered=True), name="num_1"))
    cargs = spy.call_args[0]
    bins = pd.Series(pd.Categorical([
        pd.Interval(1.0,  5.0,  "left"),
        pd.Interval(1.0,  5.0,  "left"),
        np.nan,
        pd.Interval(1.0,  5.0,  "left"),
        pd.Interval(1.0,  5.0,  "left"),
        pd.Interval(5.0,  11.01,  "left"),
        pd.Interval(5.0,  11.01,  "left"),
        pd.Interval(5.0,  11.01,  "left"),
        pd.Interval(5.0,  11.01,  "left"),
        pd.Interval(5.0,  11.01, "left"),
        pd.Interval(5.0, 11.01, "left"),
        pd.Interval(5.0, 11.01, "left")
    ], ordered=True), name="num_1")
    pd.testing.assert_series_equal(cargs[0], bins)
    pd.testing.assert_series_equal(cargs[1], pd.Series(["A","A","B","B","C","A","A","B","B","C","B","B"], name="target"))
    assert (tree.bin_edges["num_1"] == [1.0, 5.0, 11.01]).all()
    assert spy_cut.call_count == 3

def test_to_dot_string(create_tree, mocker):
    tree = create_tree()
    root = mocker.Mock(parent_id=-1, children_ids=[1, 2])
    left_child = mocker.Mock(id=1, parent_id=0, children_ids=[], global_error=2/ErrorAnalyzerConstants.GRAPH_MAX_EDGE_WIDTH)
    right_child = mocker.Mock(id=2, parent_id=0, children_ids=[], global_error=1/(1+ErrorAnalyzerConstants.GRAPH_MAX_EDGE_WIDTH))

    tree.nodes = { 0: root, 1: left_child, 2: right_child }
    tree.leaves = {1, 2}
    mocker.patch.object(root, "to_dot_string", return_value="root string")
    mocker.patch.object(left_child, "to_dot_string", return_value="left string")
    mocker.patch.object(right_child, "to_dot_string", return_value="right string")

    digraph_desc, size, node_desc, edge_desc, graph_desc, root_string, left_string, left_edge,\
        right_string, right_edge, leaves_desc, final_line = tree.to_dot_string((20, 30)).split("\n")

    assert digraph_desc == 'digraph Tree {'
    assert size == ' size="20,30!";'
    assert node_desc == 'node [shape=box, style="filled, rounded", color="black", fontname=helvetica] ;'
    assert edge_desc == 'edge [fontname=helvetica] ;'
    assert graph_desc == 'graph [ranksep=equally, splines=polyline] ;'
    assert root_string == 'root string'
    assert left_string == 'left string'
    assert left_edge == '0 -> 1 [penwidth=2.0];'
    assert right_string == 'right string'
    assert right_edge == '0 -> 2 [penwidth=1];'
    assert leaves_desc == '{rank=same ; 1; 2} ;'
    assert final_line == '}'

def test_set_node_info(create_tree, mocker):
    tree = create_tree()
    root = mocker.Mock(samples=[100, 1], local_error=[100, 100])
    node = mocker.Mock()
    set_root_info = mocker.patch.object(root, "set_node_info", return_value=None)
    set_node_info = mocker.patch.object(node, "set_node_info", return_value=None)

    tree.nodes = { 0: root, 1: node }
    class_samples = {
        ErrorAnalyzerConstants.WRONG_PREDICTION: 40,
        ErrorAnalyzerConstants.CORRECT_PREDICTION: 60
    }
    tree.set_node_info(0, class_samples)
    set_root_info.assert_called_once_with(12, class_samples, 1)

    tree.set_node_info(1, class_samples)
    set_node_info.assert_called_once_with(100, class_samples, .4)
