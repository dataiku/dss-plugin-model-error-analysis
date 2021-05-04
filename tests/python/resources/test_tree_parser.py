import sys
sys.path.append("/Users/aguillemot/devenv/dss-plugin-model-error-analysis/python-lib")
sys.path.append("/Users/aguillemot/devenv/dip/src/main/python")
sys.path.append("/Users/aguillemot/devenv/dataiku-api-client-python")
from dku_error_analysis_tree_parsing.tree_parser import TreeParser
from dku_error_analysis_decision_tree.node import Node
from mealy import ErrorAnalyzerConstants
import pandas as pd
import numpy as np
import pytest

#from dataiku.doctor.preprocessing.dataframe_preprocessing import RescalingProcessor2, QuantileBinSeries, UnfoldVectorProcessor, BinarizeSeries, \
#    FastSparseDummifyProcessor, ImpactCodingStep, FlagMissingValue2, TextCountVectorizerProcessor, TextHashingVectorizerWithSVDProcessor, \
#    TextHashingVectorizerProcessor, TextTFIDFVectorizerProcessor, CategoricalFeatureHashingProcessor

@pytest.fixture
def create_parser(mocker):
    def _create(steps=None, per_feature=None, error_model=None, num_features=None, feature_names=None):
        model_handler = mocker.Mock()
        model_handler.get_pipeline.return_value.steps = [] if steps is None else steps
        model_handler.get_per_feature.return_value = {} if per_feature is None else per_feature
        parser = TreeParser(model_handler, error_model,
                          ["test"] if feature_names is None else feature_names)
        if num_features is not None:
            parser.num_features = num_features
        return parser
    return _create

@pytest.fixture
def df():
    return lambda: pd.DataFrame([
        [1,      5.5, "x", "n",    "A", "does_not_matter", '["e",1]',   "['e',1]"],
        [2,      7.7, "y", np.nan, "A", "does_not_matter", '["a",0]',   "['e',1]"],
        [np.nan, 7,   "z", np.nan, "B", "does_not_matter", np.nan,      "['a',"],
        [3,      1.2, "z", "n",    "B", "does_not_matter", '["e",0]',   "['e',1]"],
        [4,      7.1, "z", np.nan, "C", "does_not_matter", '["i",2]',   "['e',1]"],
        [5,      .4,  "x", "p",    "A", "does_not_matter", '["e"]',     "['e',1]"],
        [6,      8,   "z", np.nan, "A", "does_not_matter", '["e"]',     "['e',1]"],
        [7,      5.5, "y", "p",    "B", "does_not_matter", '["i",1]',   "['e',1]"],
        [8,      1.5, "z", "n",    "B", "does_not_matter", '["i",1]',   "['e',1]"],
        [9,      3,   "y", "n",    "C", "does_not_matter", '["i",2]',   "['e',1]"],
        [10,     7.5, "x", np.nan, "B", "does_not_matter", '["i",100]', "['e',1]"],
        [11,     6,   "x", np.nan, "B", "does_not_matter", '["i"]',     "['e',1]"]
    ], columns=("num_1", "num_2", "cat_1", "cat_2", "target", "text", "vector", "bad_vector"))

# PARSING METHODS
def test_rank_features(mocker, df, create_parser, caplog):
    error_model = mocker.Mock()
    error_model.feature_importances_ = np.array([1, 3, 0, 5, 2, 4])
    feature_names = [
        "prep_2_feat_a",
        "prep_1_feat_b",
        "prep_feat_c",
        "prep_feat_c",
        "prep_1_feat_b",
        "prep_1_feat_a"
    ]
    mapping = {
        "prep_1_feat_a": TreeParser.SplitParameters(None, "feat_a"),
        "prep_2_feat_a": TreeParser.SplitParameters(None, "feat_a"),
        "prep_1_feat_b": TreeParser.SplitParameters(None, "feat_b"),
        "prep_2_feat_b": TreeParser.SplitParameters(None, "feat_b"),
        "prep_feat_c": TreeParser.SplitParameters(None, "feat_c")
    }

    # Only ranked features from error model
    parser = create_parser(error_model=error_model,
                            feature_names=list(feature_names))
    parser.preprocessed_feature_mapping = dict(mapping)
    dataframe = df()
    ranked_features = parser.rank_features(dataframe)
    pd.testing.assert_frame_equal(df(), dataframe)
    assert ranked_features == ["feat_c", "feat_a", "feat_b"]
    assert not parser.num_features

    per_feature = {
        "vector": { "role": "REJECT", "type": "VECTOR" },
        "cat_1": { "role": "WEIGHT", "type": "CATEGORY" },
        "num_1": { "role": "REJECT", "type": "NUMERIC" },
        "num_2": { "role": "INPUT", "type": "NUMERIC" },
        "target": { "role": "TARGET", "type": "NUMERIC" },
        "text": { "role": "REJECT", "type": "TEXT"}
    }

    # Add rejected features from original model
    parser = create_parser(per_feature=per_feature,
                            error_model=error_model,
                            feature_names=list(feature_names))
    parser.preprocessed_feature_mapping = dict(mapping)
    dataframe = df()
    ranked_features = parser.rank_features(dataframe)
    pd.testing.assert_series_equal(dataframe["vector [element #0]"],
        pd.Series(['e','a',np.nan,'e','i','e','e','i','i','i','i','i'], name="vector [element #0]"))
    pd.testing.assert_series_equal(dataframe["vector [element #1]"],
        pd.Series([1, 0, np.nan, 0, 2, np.nan, np.nan, 1, 1, 2, 100, np.nan], name="vector [element #1]"))   
    assert ranked_features[:3] == ["feat_c", "feat_a", "feat_b"]
    assert set(ranked_features) == {"feat_c", "feat_a", "feat_b", "vector [element #0]", "vector [element #1]", "cat_1", "num_1"}
    assert parser.num_features == {"num_1", "vector [element #1]"}

    # Check badly formatted vector columns are properly handled
    parser = create_parser(per_feature={"bad_vector": {"role": "REJECT", "type": "VECTOR"}},
                            error_model=error_model,
                            feature_names=list(feature_names))
    parser.preprocessed_feature_mapping = dict(mapping)
    dataframe = df()
    pd.testing.assert_frame_equal(df(), dataframe)
    ranked_features = parser.rank_features(dataframe)
    assert ranked_features == ["feat_c", "feat_a", "feat_b"]
    assert not parser.num_features
    log = caplog.records[-1]
    assert log.levelname == "WARNING"
    assert log.msg.startswith("Error while parsing vector feature bad_vector:")
    assert log.msg.endswith("It will not be used for charts")

def mocked_get_split_param(feature):
    if feature == "cat_1":
        return TreeParser.SplitParameters(Node.TYPES.CAT, "cat_1", ["A"], "super_cat_1")
    if feature == "num_2":
        return TreeParser.SplitParameters(Node.TYPES.NUM, "num_2", None, None,
            invert_left_and_right=lambda x: True)
    if feature == "cat_2":
        return TreeParser.SplitParameters(Node.TYPES.CAT, "cat_2", None, "super_cat_2",
            value_func=lambda t: t*10, add_preprocessed_feature=lambda array, col: array[:, col+1],
            invert_left_and_right=lambda x: x < -2)
    if feature == "num_1":
        return TreeParser.SplitParameters(Node.TYPES.NUM, "foo", None, "num_1")

def test_build_tree(mocker, df, create_parser):
    mocker.patch("dku_error_analysis_tree_parsing.tree_parser.descale_numerical_thresholds",
                 return_value=[8, -2, .5, 3, -.5, -2, -2, -2, 1, -2, -2])

    # Mock the tree
    feature_names = ["num_1", "num_2", "cat_1", "cat_2", "text"]
    dataframe = df()[feature_names]
    tree = mocker.Mock(df=dataframe.copy())
    tree.set_node_info.return_value = None
    tree.add_split_no_siblings.return_value = None

    # Mock the error model
    error_model = mocker.Mock(classes_=np.array([ErrorAnalyzerConstants.WRONG_PREDICTION,
        ErrorAnalyzerConstants.CORRECT_PREDICTION]))
    error_model.tree_.children_left = np.array([1, -2, 3, 7, 5, -2, -2, -2, 9, -2, -2])
    error_model.tree_.children_right = np.array([2, -2, 4, 8, 6, -2, -2, -2, 10, -2, -2])
    error_model.tree_.feature = np.array([2, -2, 1, 0, 3, -2, -2, -2, 3, -2, -2])
    error_model.tree_.value = np.array([
        [[42, 69]],
        [[2, 9]],
        [[40, 70]],
        [[30, 60]],
        [[10, 10]],
        [[2, 9]],
        [[8, 1]],
        [[2, 30]],
        [[28, 30]],
        [[1, 10]],
        [[27, 20]]
    ])

    array = np.array([
        [.5, 0, "toast", "test", "hellow"],
        [.5, 0, "toast", "test", "hellow"],
        [.5, 0, "toast", "test", "hellow"],
        [.5, 0, "toast", "test", "hellow"],
        [.5, 0, "toast", "test", "hellow"],
        [.5, 0, "toast", "test", "hellow"],
        [.5, 0, "toast", "test", "hellow"],
        [.5, 0, "toast", "test", "hellow"],
        [.5, 0, "toast", "test", "hellow"],
        [.5, 0, "toast", "test", "hellow"],
        [.5, 0, "toast", "test", "hellow"],
        [.5, 0, "toast", "test", "hellow"]
    ])
    d = {
        "cat_1":  TreeParser.SplitParameters(Node.TYPES.CAT, "cat_1", ["A"], "super_cat_1"),
        "num_2": TreeParser.SplitParameters(Node.TYPES.NUM, "num_2", None, None,
            invert_left_and_right=lambda x: True),
        "cat_2": TreeParser.SplitParameters(Node.TYPES.CAT, "cat_2", None, "super_cat_2",
            value_func=lambda t: t*10, add_preprocessed_feature=lambda array, col: array[:, col+1],
            invert_left_and_right=lambda x: x < -2),
        "num_1": TreeParser.SplitParameters(Node.TYPES.NUM, "foo", None, "num_1")
    }

    spy_cat_1 = mocker.spy(d["cat_1"], "add_preprocessed_feature")
    spy_cat_2 = mocker.spy(d["cat_2"], "add_preprocessed_feature")
    spy_num_1 = mocker.spy(d["num_1"], "add_preprocessed_feature")
    spy_num_2 = mocker.spy(d["num_2"], "add_preprocessed_feature")

    parser = create_parser(error_model=error_model,
                           num_features={"num_1", "num_2"},
                           feature_names=feature_names)
    mocker.patch.object(parser, '_get_split_parameters', side_effect=lambda x: d[x])
    parser.parse_nodes(tree, array)

    assert tree.set_node_info.call_count == 11
    call_args = tree.set_node_info.call_args_list
    assert call_args[0][0] == (0, {ErrorAnalyzerConstants.WRONG_PREDICTION:42,
        ErrorAnalyzerConstants.CORRECT_PREDICTION: 69})
    assert call_args[1][0] == (1, {ErrorAnalyzerConstants.WRONG_PREDICTION:2,
    ErrorAnalyzerConstants.CORRECT_PREDICTION: 9})
    assert call_args[2][0] == (2, {ErrorAnalyzerConstants.WRONG_PREDICTION:40,
    ErrorAnalyzerConstants.CORRECT_PREDICTION: 70})
    assert call_args[3][0] == (4, {ErrorAnalyzerConstants.WRONG_PREDICTION:10,
        ErrorAnalyzerConstants.CORRECT_PREDICTION: 10})
    assert call_args[4][0] == (3, {ErrorAnalyzerConstants.WRONG_PREDICTION:30,
        ErrorAnalyzerConstants.CORRECT_PREDICTION: 60})
    assert call_args[5][0] == (5, {ErrorAnalyzerConstants.WRONG_PREDICTION:2,
        ErrorAnalyzerConstants.CORRECT_PREDICTION: 9})
    assert call_args[6][0] == (6, {ErrorAnalyzerConstants.WRONG_PREDICTION:8,
        ErrorAnalyzerConstants.CORRECT_PREDICTION: 1})
    assert call_args[7][0] == (7, {ErrorAnalyzerConstants.WRONG_PREDICTION:2,
        ErrorAnalyzerConstants.CORRECT_PREDICTION: 30})
    assert call_args[8][0] == (8, {ErrorAnalyzerConstants.WRONG_PREDICTION:28,
        ErrorAnalyzerConstants.CORRECT_PREDICTION: 30})
    assert call_args[9][0] == (9, {ErrorAnalyzerConstants.WRONG_PREDICTION:1,
        ErrorAnalyzerConstants.CORRECT_PREDICTION: 10})
    assert call_args[10][0] == (10, {ErrorAnalyzerConstants.WRONG_PREDICTION:27,
        ErrorAnalyzerConstants.CORRECT_PREDICTION: 20})

    assert tree.add_split_no_siblings.call_count == 5
    call_args = tree.add_split_no_siblings.call_args_list
    assert call_args[0][0] == (Node.TYPES.CAT, 0, "super_cat_1", ['A'], 1, 2)
    assert call_args[1][0] == (Node.TYPES.NUM, 2, "num_2", .5, 4, 3)
    assert call_args[2][0] == (Node.TYPES.CAT, 4, "super_cat_2", -5, 5, 6)
    assert call_args[3][0] == (Node.TYPES.NUM, 3, "num_1", 3, 7, 8)
    assert call_args[4][0] == (Node.TYPES.CAT, 8, "super_cat_2", 10, 9, 10)

    assert spy_cat_1.call_count == 1
    assert spy_cat_2.call_count == 1
    assert spy_num_1.call_count == 0
    assert spy_num_2.call_count == 0

    pd.testing.assert_frame_equal(tree.df, pd.concat([dataframe, pd.Series(["toast"]*12, name="super_cat_1"), pd.Series(["hellow"]*12, name="super_cat_2")], axis=1))

# CATEGORICAL HANDLINGS
def check_dummy(split, name, value=None, others=False):
    assert split.node_type == Node.TYPES.CAT
    assert split.feature == name
    if value is not None:
        assert split.value == value
    else:
        assert len(split.value) == 1 and np.isnan(split.value[0])
    if others:
        assert not split.invert_left_and_right(0) and not split.invert_left_and_right(-.5)\
            and not split.invert_left_and_right(.5)
    else:
        assert split.invert_left_and_right and split.invert_left_and_right(0)

def test_dummy(create_parser, mocker):
    parser = create_parser()
    step = mocker.Mock(values=["A", "B"], input_column_name="test", should_drop=True)
    parser._add_dummy_mapping(step)
    assert len(parser.preprocessed_feature_mapping) == 3

    a = parser.preprocessed_feature_mapping["dummy:test:A"]
    check_dummy(a, "test", ["A"])

    b = parser.preprocessed_feature_mapping["dummy:test:B"]
    check_dummy(b, "test", ["B"])

    nan = parser.preprocessed_feature_mapping["dummy:test:N/A"]
    check_dummy(nan, "test")

    parser = create_parser()
    step = mocker.Mock(values=["A", "B"], input_column_name="test", should_drop=False)
    parser._add_dummy_mapping(step)
    assert len(parser.preprocessed_feature_mapping) == 4

    a = parser.preprocessed_feature_mapping["dummy:test:A"]
    check_dummy(a, "test", ["A"])

    b = parser.preprocessed_feature_mapping["dummy:test:B"]
    check_dummy(b, "test", ["B"])

    nan = parser.preprocessed_feature_mapping["dummy:test:N/A"]
    check_dummy(nan, "test")

    others = parser.preprocessed_feature_mapping["dummy:test:__Others__"]
    check_dummy(others, "test", ["A", "B"], True)

def test_impact(create_parser, mocker):
    # < DSS 10
    parser = create_parser()
    step = mocker.Mock(column_name="test")
    step.impact_coder._impact_map.columns.values = ["A", "B"]
    parser._add_impact_mapping(step)
    assert len(parser.preprocessed_feature_mapping) == 2

    preproc_array = np.array([
        [-1, 0],
        [0,  2],
        [0,  3],
        [0,  0],
        [1,  1],
        [0,  0],
        [4,  0]
    ])

    a = parser.preprocessed_feature_mapping["impact:test:A"]
    assert a.node_type == Node.TYPES.NUM
    assert a.chart_name == "test"
    assert a.feature == "test [A]"
    assert a.value is None
    assert not a.invert_left_and_right(0) and not a.invert_left_and_right(-.5)\
        and not a.invert_left_and_right(.5)
    assert (a.add_preprocessed_feature(preproc_array, 0) == [-1,0,0,0,1,0,4]).all()

    b = parser.preprocessed_feature_mapping["impact:test:B"]
    assert b.node_type == Node.TYPES.NUM
    assert b.chart_name == "test"
    assert b.feature == "test [B]"
    assert not b.invert_left_and_right(0) and not b.invert_left_and_right(-.5)\
        and not b.invert_left_and_right(.5)
    assert (b.add_preprocessed_feature(preproc_array, 0) == [-1,0,0,0,1,0,4]).all()

    # >= DSS 10
    parser = create_parser()
    step = mocker.Mock(column_name="test")
    step.impact_coder.encoding_map.columns.values = ["A"]
    del step.impact_coder._impact_map
    parser._add_impact_mapping(step)
    assert len(parser.preprocessed_feature_mapping) == 1

    a = parser.preprocessed_feature_mapping["impact:test:A"]
    assert a.node_type == Node.TYPES.NUM
    assert a.chart_name == "test"
    assert a.feature == "test [A]"
    assert a.value is None
    assert not a.invert_left_and_right(0) and not a.invert_left_and_right(-.5)\
        and not a.invert_left_and_right(.5)
    assert (a.add_preprocessed_feature(preproc_array, 0) == [-1,0,0,0,1,0,4]).all()

    # Step is missing proper attribute
    parser = create_parser()
    step = mocker.Mock(column_name="test")
    del step.impact_coder.encoding_map
    del step.impact_coder._impact_map
    with pytest.raises(AttributeError):
        parser._add_impact_mapping(step)

def test_whole_cat_hashing(create_parser, mocker):
    parser = create_parser()
    step = mocker.Mock(column_name="test", n_features=3)
    parser._add_cat_hashing_whole_mapping(step)
    assert len(parser.preprocessed_feature_mapping) == 3

    preproc_array = np.array([
        [-1, 0,  0],
        [0,  1,  0],
        [0,  1,  0],
        [0,  0, -1],
        [0,  0,  1],
        [0,  0,  1],
        [1,  0,  0]
    ])

    added_column = np.array([0, 1, 1, -2, 2, 2, 0])

    first = parser.preprocessed_feature_mapping["hashing:test:0"]
    assert first.node_type == Node.TYPES.CAT
    assert first.chart_name == "test"
    assert first.feature == "Hash of test"
    assert first.value is None
    assert first.value_func(.5) == [0] and first.value_func(-.5) == [0]
    assert first.invert_left_and_right and first.invert_left_and_right(.5) and not first.invert_left_and_right(-.5)
    assert (first.add_preprocessed_feature(preproc_array, 0) == added_column).all()

    second = parser.preprocessed_feature_mapping["hashing:test:1"]
    assert second.node_type == Node.TYPES.CAT
    assert second.chart_name == "test"
    assert second.feature == "Hash of test"
    assert second.value is None
    assert second.value_func(.5) == [1] and second.value_func(-.5) == [-1]
    assert second.invert_left_and_right and second.invert_left_and_right(.5) and not second.invert_left_and_right(-.5)
    assert (second.add_preprocessed_feature(preproc_array, 1) == added_column).all()

    third = parser.preprocessed_feature_mapping["hashing:test:2"]
    assert third.node_type == Node.TYPES.CAT
    assert third.chart_name == "test"
    assert third.feature == "Hash of test"
    assert third.value is None
    assert third.value_func(.5) == [2] and third.value_func(-.5) == [-2]
    assert third.invert_left_and_right and third.invert_left_and_right(.5) and not third.invert_left_and_right(-.5)
    assert (third.add_preprocessed_feature(preproc_array, 2) == added_column).all()

def test_not_whole_cat_hashing(create_parser, mocker):
    parser = create_parser()
    step = mocker.Mock(column_name="test", n_features=2)
    parser._add_cat_hashing_not_whole_mapping(step)
    assert len(parser.preprocessed_feature_mapping) == 2

    preproc_array = np.array([
        [-1, 0],
        [0,  2],
        [0,  3],
        [0,  0],
        [1,  1],
        [0,  0],
        [4,  0]
    ])

    first = parser.preprocessed_feature_mapping["hashing:test:0"]
    assert first.node_type == Node.TYPES.NUM
    assert first.chart_name == "test"
    assert first.feature == "Hash #0 of test"
    assert first.value is None
    assert first.value_func(.5) == .5 and first.value_func(-.5) == -.5
    assert not first.invert_left_and_right(0) and not first.invert_left_and_right(-.5)\
        and not first.invert_left_and_right(.5)
    assert (first.add_preprocessed_feature(preproc_array, 0) == [-1,0,0,0,1,0,4]).all()

    second = parser.preprocessed_feature_mapping["hashing:test:1"]
    assert second.node_type == Node.TYPES.NUM
    assert second.chart_name == "test"
    assert second.feature == "Hash #1 of test"
    assert second.value is None
    assert second.value_func(.5) == .5 and second.value_func(-.5) == -.5
    assert not second.invert_left_and_right(0) and not second.invert_left_and_right(-.5)\
        and not second.invert_left_and_right(.5)
    assert (second.add_preprocessed_feature(preproc_array, 1) == [0,2,3,0,1,0,0]).all()

# VECTOR HANDLING
def test_unfold(create_parser, mocker):
    parser = create_parser()
    step = mocker.Mock(input_column_name="test", vector_length=2)
    parser._add_unfold_mapping(step)
    assert len(parser.preprocessed_feature_mapping) == 2
    assert {"test [element #0]", "test [element #1]"} == parser.num_features

    preproc_array = np.array([
        [-1, 0],
        [0,  2],
        [0,  3],
        [0,  0],
        [1,  1],
        [0,  0],
        [4,  0]
    ])
    elem_0 = parser.preprocessed_feature_mapping["unfold:test:0"]
    assert elem_0.node_type == Node.TYPES.NUM
    assert elem_0.chart_name == "test [element #0]"
    assert elem_0.friendly_name == None
    assert elem_0.value is None
    assert elem_0.value_func(.5) == .5 and elem_0.value_func(-.5) == -.5
    assert not elem_0.invert_left_and_right(0) and not elem_0.invert_left_and_right(-.5)\
        and not elem_0.invert_left_and_right(.5)
    assert (elem_0.add_preprocessed_feature(preproc_array, 0) == [-1,0,0,0,1,0,4]).all()

    elem_1 = parser.preprocessed_feature_mapping["unfold:test:1"]
    assert elem_1.node_type == Node.TYPES.NUM
    assert elem_1.chart_name == "test [element #1]"
    assert elem_1.friendly_name == None
    assert elem_1.value is None
    assert elem_1.value_func(.5) == .5 and elem_1.value_func(-.5) == -.5
    assert not elem_1.invert_left_and_right(0) and not elem_1.invert_left_and_right(-.5)\
        and not elem_1.invert_left_and_right(.5)
    assert (elem_1.add_preprocessed_feature(preproc_array, 1) == [0,2,3,0,1,0,0]).all()

# NUM HANDLINGS
def test_add_preprocessed_rescaled_num(create_parser, mocker):
    parser = create_parser()
    parser.rescalers = {"test": "does_not_matter"}
    mocker.patch("dku_error_analysis_tree_parsing.tree_parser.denormalize_feature_value",
        side_effect=lambda x, y: x + " " + str(y))
    func, name = parser._add_preprocessed_rescaled_num_feature("test")
    preproc_array = np.array([
        [-1, 0],
        [0,  2],
        [0,  3],
        [0,  0],
        [1,  1],
        [0,  0],
        [4,  0]
    ])
    assert (func(preproc_array, 32) == \
        ["does_not_matter -1",
        "does_not_matter 0",
        "does_not_matter 0",
        "does_not_matter 0",
        "does_not_matter 1",
        "does_not_matter 0",
        "does_not_matter 4"]).all()
    assert name == "preprocessed:rescaled:test"

def mocked_add_preprocessed_rescaled_num_feature(original_name):
    return "fake_function", "friendly_name"

def test_identity(create_parser, mocker):
    parser = create_parser()
    mocker.patch.object(parser, '_add_preprocessed_rescaled_num_feature', side_effect=mocked_add_preprocessed_rescaled_num_feature)
    parser._add_identity_mapping("test")
    assert len(parser.preprocessed_feature_mapping) == 1
    assert {"test"} == parser.num_features

    split = parser.preprocessed_feature_mapping["test"]
    assert split.node_type == Node.TYPES.NUM
    assert split.chart_name == "test"
    assert split.feature == "friendly_name"
    assert split.value is None
    assert split.value_func(.5) == .5 and split.value_func(-.5) == -.5
    assert not split.invert_left_and_right(0) and not split.invert_left_and_right(-.5)\
        and not split.invert_left_and_right(.5)
    assert split.add_preprocessed_feature == "fake_function"

def test_binarize(create_parser, mocker):
    parser = create_parser()
    patched = mocker.patch.object(parser, '_add_preprocessed_rescaled_num_feature', side_effect=mocked_add_preprocessed_rescaled_num_feature)
    step = mocker.Mock(in_col="test", threshold=42)
    step._output_name.return_value = "output"
    parser._add_binarize_mapping(step)
    assert len(parser.preprocessed_feature_mapping) == 1
    assert patched.call_count == 1
    assert {"test"} == parser.num_features

    split = parser.preprocessed_feature_mapping["num_binarized:output"]
    assert split.node_type == Node.TYPES.NUM
    assert split.chart_name == "test"
    assert split.feature == "friendly_name"
    assert split.value == 42
    assert split.value_func(.5) == .5 and split.value_func(-.5) == -.5
    assert not split.invert_left_and_right(0) and not split.invert_left_and_right(-.5)\
        and not split.invert_left_and_right(.5)
    assert split.add_preprocessed_feature == "fake_function"

def test_quantize(create_parser, mocker):
    parser = create_parser()
    patched = mocker.patch.object(parser, '_add_preprocessed_rescaled_num_feature', side_effect=mocked_add_preprocessed_rescaled_num_feature)
    step = mocker.Mock(in_col="test", nb_bins=42, r={"bounds": ["0.5", "1.6", "7.8"]})
    parser._add_quantize_mapping(step)
    assert len(parser.preprocessed_feature_mapping) == 1
    assert patched.call_count == 1
    assert {"test"} == parser.num_features

    split = parser.preprocessed_feature_mapping["num_quantized:test:quantile:42"]
    assert split.node_type == Node.TYPES.NUM
    assert split.chart_name == "test"
    assert split.feature == "friendly_name"
    assert split.value is None
    assert split.value_func(0) == 1.6 and split.value_func(1) == 7.8
    assert not split.invert_left_and_right(0) and not split.invert_left_and_right(-.5)\
        and not split.invert_left_and_right(.5)
    assert split.add_preprocessed_feature == "fake_function"

def test_flag_missing(create_parser, mocker):
    # Flag on numerical feature
    parser = create_parser()
    step = mocker.Mock(feature="test", output_block_name="num_flagonly")
    step._output_name.return_value = "output"
    parser._add_flag_missing_value_mapping(step)
    assert len(parser.preprocessed_feature_mapping) == 1
    assert {"test"} == parser.num_features

    split = parser.preprocessed_feature_mapping["output"]
    assert split.node_type == Node.TYPES.CAT
    assert split.chart_name == "test"
    assert split.friendly_name is None
    assert len(split.value) == 1 and np.isnan(split.value[0])
    assert split.value_func(.5) == .5 and split.value_func(-.5) == -.5
    assert not split.invert_left_and_right(0) and not split.invert_left_and_right(-.5)\
        and not split.invert_left_and_right(.5)

    # Flag on categorical feature
    parser = create_parser()
    step = mocker.Mock(feature="test", output_block_name="anything_else")
    step._output_name.return_value = "output"
    parser._add_flag_missing_value_mapping(step)
    assert len(parser.preprocessed_feature_mapping) == 1

    split = parser.preprocessed_feature_mapping["output"]
    assert split.node_type == Node.TYPES.CAT
    assert split.chart_name == "test"
    assert split.friendly_name is None
    assert len(split.value) == 1 and np.isnan(split.value[0])
    assert split.value_func(.5) == .5 and split.value_func(-.5) == -.5
    assert not split.invert_left_and_right(0) and not split.invert_left_and_right(-.5)\
        and not split.invert_left_and_right(.5)

# TEXT HANDLINGS
def check_text_features(preproc_array, split, name):
    assert split.node_type == Node.TYPES.NUM
    assert split.chart_name is None
    assert split.feature == name
    assert split.value is None
    assert split.value_func(.5) == .5 and split.value_func(-.5) == -.5
    assert not split.invert_left_and_right(0) and not split.invert_left_and_right(-.5)\
        and not split.invert_left_and_right(.5)
    assert (split.add_preprocessed_feature(preproc_array, 0) == [-1,0,0,0,1,0,4]).all()
    assert (split.add_preprocessed_feature(preproc_array, 1) == [0,2,3,0,1,0,0]).all()

def test_vect_hashing(create_parser, mocker):
    # Hash without SVD
    parser = create_parser()
    step = mocker.Mock(column_name="test", n_features=2)
    parser._add_hashing_vect_mapping(step)
    assert len(parser.preprocessed_feature_mapping) == 2

    preproc_array = np.array([
        [-1, 0],
        [0,  2],
        [0,  3],
        [0,  0],
        [1,  1],
        [0,  0],
        [4,  0]
    ])

    first = parser.preprocessed_feature_mapping["hashvect:test:0"]
    check_text_features(preproc_array, first, "test [text #0]")

    second = parser.preprocessed_feature_mapping["hashvect:test:1"]
    check_text_features(preproc_array, second, "test [text #1]")

    # Hash with SVD
    parser = create_parser()
    step = mocker.Mock(column_name="test", n_features=1)
    parser._add_hashing_vect_mapping(step, True)
    assert len(parser.preprocessed_feature_mapping) == 1

    first = parser.preprocessed_feature_mapping["thsvd:test:0"]
    check_text_features(preproc_array, first, "test [text #0]")

def test_count_vect(create_parser, mocker):
    parser = create_parser()
    step = mocker.Mock(column_name="test", prefix="prefix")
    vectorizer = mocker.Mock()
    vectorizer.get_feature_names.return_value = ["word", "random"]
    step.resource = {"vectorizer": vectorizer}
    parser._add_text_count_vect_mapping(step)
    assert len(parser.preprocessed_feature_mapping) == 2

    preproc_array = np.array([
        [-1, 0],
        [0,  2],
        [0,  3],
        [0,  0],
        [1,  1],
        [0,  0],
        [4,  0]
    ])

    first = parser.preprocessed_feature_mapping["prefix:test:word"]
    check_text_features(preproc_array, first, "test: occurrences of word")

    second = parser.preprocessed_feature_mapping["prefix:test:random"]
    check_text_features(preproc_array, second, "test: occurrences of random")

def test_tfidf_vect(create_parser, mocker):
    parser = create_parser()
    step = mocker.Mock(column_name="test")
    vectorizer = mocker.Mock(idf_=[42.4242])
    vectorizer.get_feature_names.return_value = ["word", "random"]
    step.resource = {"vectorizer": vectorizer}
    parser._add_tfidf_vect_mapping(step)
    assert len(parser.preprocessed_feature_mapping) == 1

    preproc_array = np.array([
        [-1, 0],
        [0,  2],
        [0,  3],
        [0,  0],
        [1,  1],
        [0,  0],
        [4,  0]
    ])

    first = parser.preprocessed_feature_mapping["tfidfvec:test:42.424:word"]
    check_text_features(preproc_array, first, "test: tf-idf of word (idf=42.424)")

    parser = create_parser()
    step = mocker.Mock(column_name="test")
    vectorizer = mocker.Mock(idf_=[42.4242, 1])
    vectorizer.get_feature_names.return_value = ["word", "random"]
    step.resource = {"vectorizer": vectorizer}
    parser._add_tfidf_vect_mapping(step)
    assert len(parser.preprocessed_feature_mapping) == 2

    first = parser.preprocessed_feature_mapping["tfidfvec:test:42.424:word"]
    check_text_features(preproc_array, first, "test: tf-idf of word (idf=42.424)")

    second = parser.preprocessed_feature_mapping["tfidfvec:test:1.000:random"]
    check_text_features(preproc_array, second, "test: tf-idf of random (idf=1.000)")
