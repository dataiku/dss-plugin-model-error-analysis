import numpy as np
from collections import deque
from dku_error_analysis_decision_tree.node import Node
from dku_error_analysis_decision_tree.tree import InteractiveTree
from dataiku.doctor.preprocessing.dataframe_preprocessing import RescalingProcessor2, QuantileBinSeries, UnfoldVectorProcessor, BinarizeSeries, \
    FastSparseDummifyProcessor, ImpactCodingStep, FlagMissingValue2, TextCountVectorizerProcessor, TextHashingVectorizerWithSVDProcessor, \
    TextHashingVectorizerProcessor, TextTFIDFVectorizerProcessor
from dku_error_analysis_tree_parsing.depreprocessor import descale_numerical_thresholds
from dku_error_analysis_mpp.error_analyzer import ERROR_COLUMN

MAX_MOST_IMPORTANT_FEATURES = 3

class TreeParser(object):
    class Preprocessing:
        DUMMY="dummy"
        DUMMY_OTHERS="dummy_others"
        FLAG="flag"
        IMPACT="impact"
        QUANTIZE="quantize"
        UNFOLD="unfold"
        BINARIZE="binarize"
        TEXT_PREPROC="text_preproc"

    def __init__(self, model_handler, error_model):
        self.model_handler = model_handler
        self.error_model = error_model
        self.preprocessed_feature_mapping = {}
        self.rescalers = []
        self._create_preprocessed_feature_mapping()

    def add_flag_missing_value_mapping(self, step):
        self.preprocessed_feature_mapping[step._output_name()] = (Node.TYPES.CAT, step.feature, [np.nan], TreeParser.Preprocessing.FLAG)

    def add_dummy_mapping(self, step):
        for value in step.values:
            preprocessed_name = "dummy:{}:{}".format(step.input_column_name, value)
            self.preprocessed_feature_mapping[preprocessed_name] = (Node.TYPES.CAT, step.input_column_name, [value], TreeParser.Preprocessing.DUMMY)
        self.preprocessed_feature_mapping["dummy:{}:N/A".format(step.input_column_name)] = (Node.TYPES.CAT, step.input_column_name, [np.nan], TreeParser.Preprocessing.DUMMY)
        if not step.should_drop:
            self.preprocessed_feature_mapping["dummy:{}:__Others__".format(step.input_column_name)] = (Node.TYPES.CAT, step.input_column_name, step.values, TreeParser.Preprocessing.DUMMY_OTHERS)

    def add_impact_mapping(self, step):
        for value in step.impact_coder._impact_map.columns.values:
            preprocessed_name = "impact:{}:{}".format(step.column_name, value)
            display_name = "{} [{}]".format(step.column_name, value)
            self.preprocessed_feature_mapping[preprocessed_name] = (Node.TYPES.NUM, display_name, lambda threshold: threshold, TreeParser.Preprocessing.IMPACT)

    def add_binarize_mapping(self, step):
        self.preprocessed_feature_mapping["num_binarized:" + step._output_name()] = (Node.TYPES.NUM, step.in_col, step.threshold, TreeParser.Preprocessing.BINARIZE)

    def add_quantize_mapping(self, step):
        bounds = step.r["bounds"]
        split_value_func = lambda threshold: float(bounds[int(threshold) + 1])
        preprocessed_name = "num_quantized:{0}:quantile:{1}".format(step.in_col, step.nb_bins)
        self.preprocessed_feature_mapping[preprocessed_name] = (Node.TYPES.NUM, step.in_col, split_value_func, TreeParser.Preprocessing.QUANTIZE)

    def add_unfold_mapping(self, step):
        for i in xrange(step.vector_length):
            preprocessed_name = "unfold:{}:{}".format(step.input_column_name, i)
            display_name = "{} [element #{}]".format(step.input_column_name, i)
            self.preprocessed_feature_mapping[preprocessed_name] = (Node.TYPES.NUM, display_name, lambda threshold: threshold, TreeParser.Preprocessing.UNFOLD)

    def add_hashing_vect_mapping(self, step, with_svd=False):
        prefix = "thsvd" if with_svd else "hashvect"
        for i in range(step.n_features):
            preprocessed_name = "{}:{}:{}".format(prefix, step.column_name, i)
            display_name = "{} [text #{}]".format(step.column_name, i)
            self.preprocessed_feature_mapping[preprocessed_name] = (Node.TYPES.NUM, display_name, lambda threshold: threshold, TreeParser.Preprocessing.TEXT_PREPROC)

    def add_text_count_vect_mapping(self, step):
        for word in step.resource["vectorizer"].get_feature_names():
            preprocessed_name = "{}:{}:{}".format(step.prefix, step.column_name, word)
            display_name = "{}: occurrences of {}".format(step.column_name, word)
            self.preprocessed_feature_mapping[preprocessed_name] = (Node.TYPES.NUM, display_name, lambda threshold: int(threshold), TreeParser.Preprocessing.TEXT_PREPROC)

    def add_tfidf_vect_mapping(self, step):
        vec = step.resource["vectorizer"]
        for word, idf in zip(vec.get_feature_names(), vec.idf_):
            preprocessed_name = "tfidfvec:{}:{:.3f}:{}".format(step.column_name, idf, word)
            display_name = "{}: tfidf of {}".format(step.column_name, word)
            self.preprocessed_feature_mapping[preprocessed_name] = (Node.TYPES.NUM, display_name, lambda threshold: threshold, TreeParser.Preprocessing.TEXT_PREPROC)

    def _create_preprocessed_feature_mapping(self):
        for step in self.model_handler.get_pipeline().steps:
            if isinstance(step, RescalingProcessor2):
                self.rescalers.append(step)
            {
                FlagMissingValue2: self.add_flag_missing_value_mapping,
                QuantileBinSeries: self.add_quantize_mapping,
                FastSparseDummifyProcessor: self.add_dummy_mapping,
                BinarizeSeries: self.add_binarize_mapping,
                UnfoldVectorProcessor: self.add_unfold_mapping,
                ImpactCodingStep: self.add_impact_mapping,
                TextCountVectorizerProcessor: self.add_text_count_vect_mapping,
                TextHashingVectorizerWithSVDProcessor: lambda step: self.add_hashing_vect_mapping(step, with_svd=True),
                TextHashingVectorizerProcessor: lambda step: self.add_hashing_vect_mapping(step),
                TextTFIDFVectorizerProcessor: self.add_tfidf_vect_mapping
            }.get(step.__class__, lambda step: None)(step)

    @staticmethod
    def split_value_is_threshold_dependant(method):
        return method == TreeParser.Preprocessing.IMPACT or method == TreeParser.Preprocessing.UNFOLD or method == TreeParser.Preprocessing.QUANTIZE or method == TreeParser.Preprocessing.TEXT_PREPROC

    @staticmethod
    def split_uses_preprocessed_feature(method):
        return method == TreeParser.Preprocessing.IMPACT or method == TreeParser.Preprocessing.UNFOLD or method == TreeParser.Preprocessing.TEXT_PREPROC

    @staticmethod
    def force_others_on_the_right(method):
        return method == TreeParser.Preprocessing.DUMMY

    def get_preprocessed_feature_details(self, preprocessed_name, threshold=None):
        return self.preprocessed_feature_mapping.get(preprocessed_name, (Node.TYPES.NUM, preprocessed_name, threshold, None))

    def build_all_nodes(self, tree, feature_list, preprocessed_x):
        error_model_tree = self.error_model.tree_
        thresholds = descale_numerical_thresholds(error_model_tree, feature_list, self.rescalers, False)
        children_left, children_right, features = error_model_tree.children_left, error_model_tree.children_right, error_model_tree.feature
        root_node = Node(0, -1)
        ids = deque()
        tree.add_node(root_node)

        ids.append(0)
        while ids:
            parent_id = ids.popleft()
            feature_idx, threshold = features[parent_id], thresholds[parent_id]
            preprocessed_feature = feature_list[feature_idx]
            node_type, feature, split_value, method = self.get_preprocessed_feature_details(preprocessed_feature, threshold)

            if TreeParser.split_uses_preprocessed_feature(method):
                tree.df[feature] = preprocessed_x[:, feature_idx]
            if TreeParser.split_value_is_threshold_dependant(method):
                split_value = split_value(threshold)
            if TreeParser.force_others_on_the_right(method):
                left_child_id, right_child_id = children_right[parent_id], children_left[parent_id]
            else:
                left_child_id, right_child_id = children_left[parent_id], children_right[parent_id]
            tree.add_split_no_siblings(node_type, parent_id, feature, split_value, left_child_id, right_child_id)

            if children_left[left_child_id] > 0:
                ids.append(left_child_id)
            if children_left[right_child_id] > 0:
                ids.append(right_child_id)


    def build_tree(self, df, feature_list, preprocessed_x, target=ERROR_COLUMN):
        features = {}
        for name, settings in self.model_handler.get_preproc_handler().collector_data.get('per_feature').iteritems():
            avg = settings.get('stats').get('average')
            if avg is not None: #TODO AGU: add cases where missing is not replaced by mean
                features[name] = {
                    'mean':  avg
                }
        ranked_features = self._rank_features_by_error_correlation(feature_list)
        tree = InteractiveTree(df, target, ranked_features, features)
        self.build_all_nodes(tree, feature_list, preprocessed_x)
        return tree

    # Rank features according to their correlation with the model performance
    def _rank_features_by_error_correlation(self, feature_list, max_number_features=MAX_MOST_IMPORTANT_FEATURES):
        sorted_feature_indices = np.argsort(- self.error_model.feature_importances_)
        ranked_features = []
        for feature_idx in sorted_feature_indices:
            feature_importance = - self.error_model.feature_importances_[feature_idx]
            if feature_importance != 0:
                preprocessed_name = feature_list[feature_idx]
                feature = self.get_preprocessed_feature_details(preprocessed_name)[1]
                if feature not in ranked_features:
                    ranked_features.append(feature)
                    if len(ranked_features) == max_number_features:
                        return ranked_features
        return ranked_features
