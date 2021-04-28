import sys
sys.path.append("/Users/aguillemot/devenv/dss-plugin-model-error-analysis/python-lib")
from dku_error_analysis_decision_tree.node import Node, CategoricalNode, NumericalNode
from mealy import ErrorAnalyzerConstants
import pandas as pd
import numpy as np

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

def test_print_decision_rule():
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
    cat_node_one_value = CategoricalNode(1, 0, "cat_1", ["x"])
    assert not (set(cat_node_one_value.apply_filter(df).index) - set([0,5,10,11]))

    cat_node_one_value_others = CategoricalNode(1, 0, "cat_1", ["x"], others=True)
    assert not (set(cat_node_one_value_others.apply_filter(df).index) - set([1,2,3,4,6,7,8,9]))

    cat_node_several_values = CategoricalNode(1, 0, "cat_1", ["z", "y"])
    assert not (set(cat_node_several_values.apply_filter(df).index) - set([1,2,3,4,6,7,8,9]))

    cat_node_several_values_others = CategoricalNode(1, 0, "cat_1", ["z", "y"], others=True)
    assert not (set(cat_node_several_values_others.apply_filter(df).index) - set([0,5,10,11]))

    num_node_beginning = NumericalNode(1, 0, "num_1", beginning=9)
    assert not (set(num_node_beginning.apply_filter(df).index) - set([9,10,11]))

    num_node_end = NumericalNode(1, 0, "num_1", end=5)
    assert not (set(num_node_end.apply_filter(df).index) - set([0,1,3,4,5]))

    # should not happen as we have binary trees for now
    num_node_beginning_and_end = NumericalNode(1, 0, "num_1", beginning=5.02, end=9.11)
    assert len(set(num_node_beginning_and_end.apply_filter(df).index) - set([6,7,8,9])) == 0

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

def test_to_dot_string():
    # TODO
    pass
