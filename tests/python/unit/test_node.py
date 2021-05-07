from dku_error_analysis_decision_tree.node import Node, CategoricalNode, NumericalNode
from mealy import ErrorAnalyzerConstants
import pandas as pd
import numpy as np
import pytest

def test_print_decision_rule():
    with pytest.raises(NotImplementedError):
        Node(0, -1).print_decision_rule()

    cat_node_one_value = CategoricalNode(1, 0, "test", ["A"])
    assert cat_node_one_value.print_decision_rule() == "test is A"

    cat_node_one_value_others = CategoricalNode(1, 0, "test", ["A"], others=True)
    assert cat_node_one_value_others.print_decision_rule() == "test is not A"

    cat_node_several_values = CategoricalNode(1, 0, "test", ["A", "B"])
    assert cat_node_several_values.print_decision_rule() == "test in [A, B]"

    cat_node_several_values_others = CategoricalNode(1, 0, "test", ["A", "B"], others=True)
    assert cat_node_several_values_others.print_decision_rule() == "test not in [A, B]"

    num_node_beginning = NumericalNode(1, 0, "test", beginning=1)
    assert num_node_beginning.print_decision_rule() == "1.00 < test"

    num_node_end = NumericalNode(1, 0, "test", end=890.896)
    assert num_node_end.print_decision_rule() == "test <= 890.90"

def test_apply_filter():
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

    with pytest.raises(NotImplementedError):
        Node(0, -1).apply_filter(df)

    cat_node_one_value = CategoricalNode(1, 0, "cat_1", ["x"])
    pd.testing.assert_frame_equal(cat_node_one_value.apply_filter(df), df.loc[[0, 5, 10, 11], :])

    cat_node_one_value_others = CategoricalNode(1, 0, "cat_1", ["x"], others=True)
    pd.testing.assert_frame_equal(cat_node_one_value_others.apply_filter(df), df.loc[[1,2,3,4,6,7,8,9], :])

    cat_node_several_values = CategoricalNode(1, 0, "cat_1", ["z", "y"])
    pd.testing.assert_frame_equal(cat_node_several_values.apply_filter(df), df.loc[[1,2,3,4,6,7,8,9], :])

    cat_node_several_values_others = CategoricalNode(1, 0, "cat_1", ["z", "y"], others=True)
    pd.testing.assert_frame_equal(cat_node_several_values_others.apply_filter(df), df.loc[[0, 5, 10, 11], :])

    num_node_beginning = NumericalNode(1, 0, "num_1", beginning=9.08)
    pd.testing.assert_frame_equal(num_node_beginning.apply_filter(df), df.loc[[10, 11], :])

    num_node_end = NumericalNode(1, 0, "num_1", end=5)
    pd.testing.assert_frame_equal(num_node_end.apply_filter(df), df.loc[[0,1,3,4,5], :])

    num_node_beginning_and_end = NumericalNode(1, 0, "num_1", beginning=5, end=9.11)
    pd.testing.assert_frame_equal(num_node_beginning_and_end.apply_filter(df), df.loc[[6,7,8,9], :])

def test_set_node_info():
    error_node = Node(0, -1)
    class_samples = {
        ErrorAnalyzerConstants.WRONG_PREDICTION: 100,
        ErrorAnalyzerConstants.CORRECT_PREDICTION: 50
    }
    error_node.set_node_info(200, class_samples, .765)
    assert error_node.samples == [150, 75]
    assert error_node.probabilities == [[ErrorAnalyzerConstants.WRONG_PREDICTION, 2/3, 100], [ErrorAnalyzerConstants.CORRECT_PREDICTION, 1/3,  50]]
    assert error_node.local_error == [2/3, 100]
    assert error_node.global_error == .765
    assert error_node.prediction == ErrorAnalyzerConstants.WRONG_PREDICTION

    correct_node = Node(0, -1)
    class_samples = {
        ErrorAnalyzerConstants.WRONG_PREDICTION: 50,
        ErrorAnalyzerConstants.CORRECT_PREDICTION: 100
    }
    correct_node.set_node_info(200, class_samples, .765)
    assert correct_node.samples == [150, 75]
    assert correct_node.probabilities == [[ErrorAnalyzerConstants.CORRECT_PREDICTION, 2/3, 100], [ErrorAnalyzerConstants.WRONG_PREDICTION, 1/3,  50]]
    assert correct_node.local_error == [1/3, 50]
    assert correct_node.global_error == .765
    assert correct_node.prediction == ErrorAnalyzerConstants.CORRECT_PREDICTION

    no_pred_node = Node(0, -1)
    class_samples = {
        ErrorAnalyzerConstants.WRONG_PREDICTION: 0,
        ErrorAnalyzerConstants.CORRECT_PREDICTION: 0
    }
    no_pred_node.set_node_info(200, class_samples, .765)
    assert no_pred_node.samples == [0, 0]
    assert no_pred_node.probabilities == [[ErrorAnalyzerConstants.WRONG_PREDICTION, 0, 0], [ErrorAnalyzerConstants.CORRECT_PREDICTION, 0, 0]]
    assert no_pred_node.local_error == [0, 0]
    assert no_pred_node.global_error == .765
    assert no_pred_node.prediction is None

def test_to_dot_string(mocker):
    root = Node(0, -1)
    root.samples = [100, 1]
    root.local_error = [.5, "_"]
    root.global_error = .6789598
    label, samples, local_err, global_err, color_n_tooltip = root.to_dot_string().split("\n")
    assert label == '0 [label="node #0'
    assert samples == 'samples = 1%'
    assert local_err == 'local error = 50%'
    assert global_err == 'fraction of total error = 67.9%'
    assert color_n_tooltip == '", fillcolor="{}", tooltip="root"] ;'.format(
        ErrorAnalyzerConstants.ERROR_TREE_COLORS[ErrorAnalyzerConstants.WRONG_PREDICTION]+'7f')

    node = Node(1, 0)
    node.samples = [100, .4932]
    node.local_error = [0, "_"]
    node.global_error = .6789501
    mocker.patch.object(node, "print_decision_rule", return_value="decision rule")
    label, decision_rule, samples, local_err, global_err, color_n_tooltip = \
        node.to_dot_string().split("\n")
    assert label == '1 [label="node #1'
    assert decision_rule == 'decision rule'
    assert samples == 'samples = 0.49%'
    assert local_err == 'local error = 0%'
    assert global_err == 'fraction of total error = 67.9%'
    assert color_n_tooltip == '", fillcolor="{}", tooltip="decision rule"] ;'.format(
        ErrorAnalyzerConstants.ERROR_TREE_COLORS[ErrorAnalyzerConstants.WRONG_PREDICTION]+'00')

    node.local_error = [1, "_"]
    fake_rule = "pretty really very extremely fairly long decision rule"
    mocker.patch.object(node, "print_decision_rule", return_value=fake_rule)
    label, decision_rule, samples, local_err, global_err, color_n_tooltip = \
        node.to_dot_string().split("\n")
    assert label == '1 [label="node #1'
    assert decision_rule == 'pretty really very extremely fai...'
    assert samples == 'samples = 0.49%'
    assert local_err == 'local error = 100%'
    assert global_err == 'fraction of total error = 67.9%'
    assert color_n_tooltip == '", fillcolor="{}", tooltip="{}"] ;'.format(
        ErrorAnalyzerConstants.ERROR_TREE_COLORS[ErrorAnalyzerConstants.WRONG_PREDICTION]+'ff', fake_rule)

def test_jsonify():
    jsonified_node = {
        "node_id": 0,
        "parent_id": -1,
        "feature": "foo",
        "children_ids": [],
        "probabilities": [],
        "global_error": None,
        "local_error": None,
        "samples": None,
        "prediction": None
    }

    node = Node(0, -1, "foo")
    assert node.jsonify() == jsonified_node

    cat_node = CategoricalNode(0, -1, "foo", ["A"])
    jsonified_cat_node = dict(values=["A"], others=False, **jsonified_node)
    assert cat_node.jsonify() == jsonified_cat_node

    cat_node = CategoricalNode(0, -1, "foo", [np.nan], True)
    jsonified_cat_node = dict(values=["No values"], others=True, **jsonified_node)
    assert cat_node.jsonify() == jsonified_cat_node

    cat_node = CategoricalNode(0, -1, "foo", ["A", np.nan], True)
    jsonified_cat_node = dict(values=["A", np.nan], others=True, **jsonified_node)
    assert cat_node.jsonify() == jsonified_cat_node

    num_node = NumericalNode(0, -1, "foo", 1)
    jsonified_num_node = dict(beginning=1, **jsonified_node)
    assert num_node.jsonify() == jsonified_num_node

    num_node = NumericalNode(0, -1, "foo", 1, 2)
    jsonified_num_node = dict(beginning=1, end=2, **jsonified_node)
    assert num_node.jsonify() == jsonified_num_node

    num_node = NumericalNode(0, -1, "foo", end=1)
    jsonified_num_node = dict(end=1, **jsonified_node)
    assert num_node.jsonify() == jsonified_num_node
