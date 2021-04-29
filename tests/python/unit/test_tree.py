import sys
sys.path.append("/Users/aguillemot/devenv/dss-plugin-model-error-analysis/python-lib")
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
def cat_column(create_tree):
    def col(full=True):
        return pd.Series(["C", "B", "B" if full else np.nan, "B", "Q"])
    return col

def test_on_add_splits(create_tree):
    # Add numerical split
    tree = create_tree()
    tree.add_split_no_siblings(Node.TYPES.NUM, 0, "num_2", .9, 1, 2)
    assert not (tree.leaves - {1, 2})
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
    assert not (tree.leaves - {1, 2})
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

# TODO
def test_get_stats(create_tree):
    tree = create_tree()

# TODO
def test_to_dot_string(create_tree):
    tree = create_tree()
    pass

# TODO
def test_set_node_info(create_tree):
    pass
