from collections import defaultdict
import logging
from dku_error_analyzer_custom.error_analyzer_constants import ErrorAnalyzerConstants
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
import numpy as np
from scipy.sparse import issparse
from sklearn.exceptions import NotFittedError
import pandas as pd


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='error_analyzer | %(levelname)s - %(message)s')

def check_enough_data(df, min_len):
    """
    Compare length of dataframe to minimum lenght of the test data.
    Used in the relevance of the measure.

    :param df: Input dataframe
    :param min_len:
    :return:
    """
    if df.shape[0] < min_len:
        raise ValueError(
            'The original dataset is too small ({} rows) to have stable result, it needs to have at least {} rows'.format(
                df.shape[0], min_len))



def compute_accuracy_score(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def compute_primary_model_accuracy(y):
    n_test_samples = y.shape[0]
    return float(np.count_nonzero(y == ErrorAnalyzerConstants.CORRECT_PREDICTION)) / n_test_samples

def compute_confidence_decision(primary_model_true_accuracy, primary_model_predicted_accuracy):
    difference_true_pred_accuracy = np.abs(primary_model_true_accuracy - primary_model_predicted_accuracy)
    decision = difference_true_pred_accuracy <= ErrorAnalyzerConstants.TREE_ACCURACY_TOLERANCE

    fidelity = 1. - difference_true_pred_accuracy

    # TODO Binomial test
    return fidelity, decision



def error_decision_tree_report(y_true, y_pred, output_format='str'):
    """Return a report showing the main Error Decision Tree metrics.

    Args:
        y_true (numpy.ndarray): Ground truth values of wrong/correct predictions of the error tree primary model.
            Expected values in [ErrorAnalyzerConstants.WRONG_PREDICTION, ErrorAnalyzerConstants.CORRECT_PREDICTION].
        y_pred (numpy.ndarray): Estimated targets as returned by the error tree. Expected values in
            [ErrorAnalyzerConstants.WRONG_PREDICTION, ErrorAnalyzerConstants.CORRECT_PREDICTION].
        output_format (string): Return format used for the report. Valid values are 'dict' or 'str'.

    Return:
        dict or str: dictionary or string report storing different metrics regarding the Error Decision Tree.
    """

    tree_accuracy_score = compute_accuracy_score(y_true, y_pred)
    tree_balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    primary_model_predicted_accuracy = compute_primary_model_accuracy(y_pred)
    primary_model_true_accuracy = compute_primary_model_accuracy(y_true)
    fidelity, confidence_decision = compute_confidence_decision(primary_model_true_accuracy,
                                                                primary_model_predicted_accuracy)
    if output_format == 'dict':
        report_dict = dict()
        report_dict[ErrorAnalyzerConstants.TREE_ACCURACY] = tree_accuracy_score
        report_dict[ErrorAnalyzerConstants.TREE_BALANCED_ACCURACY] = tree_balanced_accuracy
        report_dict[ErrorAnalyzerConstants.TREE_FIDELITY] = fidelity
        report_dict[ErrorAnalyzerConstants.PRIMARY_MODEL_TRUE_ACCURACY] = primary_model_true_accuracy
        report_dict[ErrorAnalyzerConstants.PRIMARY_MODEL_PREDICTED_ACCURACY] = primary_model_predicted_accuracy
        report_dict[ErrorAnalyzerConstants.CONFIDENCE_DECISION] = confidence_decision
        return report_dict

    if output_format == 'str':

        report = 'The Error Decision Tree was trained with accuracy %.2f%% and balanced accuracy %.2f%%.' % (tree_accuracy_score * 100, tree_balanced_accuracy * 100)
        report += '\n'
        report += 'The Decision Tree estimated the primary model''s accuracy to %.2f%%.' % \
                  (primary_model_predicted_accuracy * 100)
        report += '\n'
        report += 'The true accuracy of the primary model is %.2f.%%' % (primary_model_true_accuracy * 100)
        report += '\n'
        report += 'The Fidelity of the error tree is %.2f%%.' % \
                  (fidelity * 100)
        report += '\n'
        if not confidence_decision:
            report += 'Warning: the built tree might not be representative of the primary model performances.'
            report += '\n'
            report += 'The error tree predicted model accuracy is considered too different from the true model accuracy.'
            report += '\n'
        else:
            report += 'The error tree is considered representative of the primary model performances.'
            report += '\n'

        return report

    else:
        raise ValueError("Output format should either be 'dict' or 'str'")

def compute_fidelity_score(y_true, y_pred):
    difference_true_pred_accuracy = np.abs(compute_primary_model_accuracy(y_true) -
                                           compute_primary_model_accuracy(y_pred))
    fidelity = 1. - difference_true_pred_accuracy

    return fidelity


def fidelity_balanced_accuracy_score(y_true, y_pred):
    return compute_fidelity_score(y_true, y_pred) + balanced_accuracy_score(y_true, y_pred)

class FeatureNameTransformer(object):
    """ Transformer of feature names and indices.

        A FeatureNameTransformer parses an input Pipeline preprocessor and generate
        a mapping between the input unprocessed feature names/indices and the output
        preprocessed feature names/indices.

        Args:
            ct_preprocessor (sklearn.compose.ColumnTransformer): preprocessor.
            orig_feats (list): list of original unpreprocessed feature names, default=None.

        Attributes:
            original_feature_names (list): list of original unpreprocessed feature names.
            preprocessed_feature_names (list): list of preprocessed feature names.

    """
    def __init__(self, original_features, preprocessed_features):
        self.original_feature_names = original_features
        self.preprocessed_feature_names = preprocessed_features

    def get_original_feature_names(self):
        return self.original_feature_names

    def get_preprocessed_feature_names(self):
        return self.preprocessed_feature_names

    def is_categorical(self, index=None, name=None):
        raise NotImplementedError

    def inverse_transform_feature_id(self, index):
        raise NotImplementedError

    def inverse_transform(self, x):
        raise NotImplementedError

    def transform(self, x):
        raise NotImplementedError

    def get_top_ranked_feature_ids(self, feature_importances, max_nr_features):
        raise NotImplementedError

    def inverse_thresholds(self, tree):
        raise NotImplementedError

def check_lists_having_same_elements(list_A, list_B):
    return set(list_A) == set(list_B)

def generate_preprocessing_steps(transformer, invert_order=False):
    if isinstance(transformer, Pipeline):
        steps = [step for name, step in transformer.steps]
        if invert_order:
            steps = reversed(steps)
    else:
        steps = [transformer]
    for step in steps:
        if step == 'drop':
            # Skip the drop step of ColumnTransformer
            continue
        if step != 'passthrough' and not isinstance(step, ErrorAnalyzerConstants.SUPPORTED_STEPS):
            # Check all the preprocessing steps are supported by mealy
            unsupported_class = step.__class__
            raise TypeError('Mealy package does not support {}. '.format(unsupported_class) +
                        'It might be because it changes output dimension without ' +
                        'providing a get_feature_names function to keep track of the ' +
                        'generated features, or that it does not provide an ' +
                        'inverse_tranform method.')
        yield step

def invert_transform_via_identity(step):
    if isinstance(step, ErrorAnalyzerConstants.STEPS_THAT_CAN_BE_INVERSED_WITH_IDENTICAL_FUNCTION):
        return True
    if step == 'passthrough' or step is None:
        return True
    return False

class PipelinePreprocessor(FeatureNameTransformer):
    """Transformer of feature values from the original values to preprocessed ones.

        A PipelinePreprocessor parses an input Pipeline preprocessor and generate
        a mapping between the input unprocessed feature values and the output
        preprocessed feature values.

        Args:
            ct_preprocessor (sklearn.compose.ColumnTransformer): preprocessing steps.
            original_features (list): list of original unpreprocessed feature names, default=None.

    """

    def __init__(self, ct_preprocessor, original_features=None):
        self.ct_preprocessor = ct_preprocessor
        self.original2preprocessed = defaultdict(list)
        self.preprocessed2original = {}
        self.categorical_features = []

        logger.info('Retrieving the list of features used in the pipeline')
        original_features_from_ct = self._get_feature_list_from_column_transformer()
        if original_features is None:
            original_features = original_features_from_ct
        elif not check_lists_having_same_elements(original_features, original_features_from_ct):
            # If user explicitly gives a list of input features, we compare it with the list derived from the ColumnTransformer
            raise ValueError('The list of features given by user does not correspond to the list of features handled by the Pipeline.')

        super(PipelinePreprocessor, self).__init__(original_features=original_features, preprocessed_features=[])

        logger.info('Generating the feature id mapping dict')
        self._create_feature_mapping()

    def _get_feature_list_from_column_transformer(self):
        all_features = []
        for _, transformer, feature_names in self.ct_preprocessor.transformers_:
            for step in generate_preprocessing_steps(transformer):
                if isinstance(step, ErrorAnalyzerConstants.VALID_CATEGORICAL_STEPS):
                    # Check for categorical features
                    self.categorical_features += feature_names
                    break

            all_features += feature_names
        return all_features

    def _create_feature_mapping(self):
        """
        Update the dicts of input <-> output feature id mapping: self.original2preprocessed and self.preprocessed2original
        """
        for _, transformer, feature_names in self.ct_preprocessor.transformers_:
            orig_feat_ids = np.where(np.in1d(self.original_feature_names, feature_names))[0]
            for step in generate_preprocessing_steps(transformer):
                output_dim_changed = False
                if isinstance(step, ErrorAnalyzerConstants.STEPS_THAT_CHANGE_OUTPUT_DIMENSION_WITH_OUTPUT_FEATURE_NAMES):
                    # It is assumed that for each pipeline, at most one step changes the feature's dimension
                    # For now, it can only be a OneHotEncoder step
                    self._update_feature_mapping_dict_using_output_names(step,
                                                                        feature_names,
                                                                        orig_feat_ids)
                    output_dim_changed = True
                    break
            if not output_dim_changed:
                self._update_feature_mapping_dict_using_input_names(feature_names, orig_feat_ids)

    def _update_feature_mapping_dict_using_input_names(self, transformer_feature_names, original_feature_ids):
        self.preprocessed_feature_names.extend(transformer_feature_names)
        for original_feat_id in original_feature_ids:
            idx = len(self.preprocessed2original)
            self.original2preprocessed[original_feat_id] = [idx]
            self.preprocessed2original[idx] = original_feat_id

    def _update_feature_mapping_dict_using_output_names(self, single_transformer, transformer_feature_names, original_feature_ids):
        out_feature_names = list(single_transformer.get_feature_names(input_features=transformer_feature_names))
        self.preprocessed_feature_names.extend(out_feature_names)
        for orig_id, orig_name in zip(original_feature_ids, transformer_feature_names):
            part_out_feature_names = [name for name in out_feature_names if orig_name + '_' in name]
            offset = len(self.preprocessed2original)
            for i in range(len(part_out_feature_names)):
                self.original2preprocessed[orig_id].append(offset + i)
                self.preprocessed2original[offset + i] = orig_id

    def _transform_feature_id(self, index):
        """
        Args:
            index: int

        Returns: index of output feature(s) generated by the requested feature.
        """
        return self.original2preprocessed[index]

    def transform(self, x):
        """Transform the input feature values according to the preprocessing pipeline.

        Args:
            x (array-like or dataframe of shape (number of samples, number of features)): input feature values.

        Return:
            numpy.ndarray: transformed feature values.
        """
        return self.ct_preprocessor.transform(x)

    def _get_feature_ids_related_to_transformer(self, transformer_feature_names):
        original_features = self.get_original_feature_names()
        original_feature_ids = np.where(np.in1d(original_features, transformer_feature_names))[0]
        preprocessed_feature_ids = []
        for i in original_feature_ids:
            preprocessed_feature_ids += self._transform_feature_id(i)
        return original_feature_ids, preprocessed_feature_ids

    @staticmethod
    def _inverse_single_step(single_step, step_output, transformer_feature_names):
        inverse_transform_function_available = getattr(single_step, "inverse_transform", None)
        if invert_transform_via_identity(single_step):
            logger.info("Reversing step using identity transformation on column(s): {}".format(single_step, ', '.join(transformer_feature_names)))
            return step_output
        if inverse_transform_function_available:
            logger.info("Reversing step using inverse_transform() method on column(s): {}".format(single_step, ', '.join(transformer_feature_names)))
            return single_step.inverse_transform(step_output)
        raise TypeError('The package does not support {} because it does not provide inverse_transform function.'.format(single_step))

    def inverse_transform(self, preprocessed_x):
        """Invert the preprocessing pipeline and inverse transform feature values.

        Args:
            preprocessed_x (numpy.ndarray or scipy sparse matrix): preprocessed feature values.

        Return:
            numpy.ndarray: feature values without preprocessing.

        """
        nr_original_features = len(self.get_original_feature_names())
        undo_prep_test_x = np.zeros((preprocessed_x.shape[0], nr_original_features), dtype='O')
        any_cat = np.vectorize(lambda x: self.is_categorical(x))

        for _, transformer, feature_names in reversed(self.ct_preprocessor.transformers_):
            original_feature_ids, preprocessed_feature_ids = self._get_feature_ids_related_to_transformer(feature_names)
            transformer_output = preprocessed_x[:, preprocessed_feature_ids]
            if issparse(transformer_output) and not np.any(any_cat(original_feature_ids)):
                transformer_output = transformer_output.todense()

            # TODO: could be simplified as sklearn.Pipeline implements inverse_transform
            for step in generate_preprocessing_steps(transformer, invert_order=True):
                transformer_input = PipelinePreprocessor._inverse_single_step(step, transformer_output, feature_names)
                transformer_output = transformer_input
            undo_prep_test_x[:, original_feature_ids] = transformer_input

        return undo_prep_test_x

    def is_categorical(self, index=None, name=None):
        """Check whether an unprocessed feature at a given index or with a given name is categorical.

        Args:
            index (int): feature index.
            name (str): feature name.

        Return:
            bool: True if the input feature is categorical, else False. If both index and name are provided, the index
                is retained.
        """
        if index is not None:
            name = self.original_feature_names[index]
        if name is not None:
            return name in self.categorical_features
        else:
            raise ValueError("Either the input index or its name should be specified.")

    def inverse_transform_feature_id(self, index):
        """Undo preprocessing of feature name.

        Transform the preprocessed feature name at given index back into the original unprocessed feature index.

        Args:
            index (int): feature index.

        Return:
            int : index of the unprocessed feature corresponding to the input preprocessed feature index.
        """
        return self.preprocessed2original[index]

    def get_top_ranked_feature_ids(self, feature_importances, max_nr_features):
        ranked_transformed_feature_ids = np.argsort(- feature_importances)
        if max_nr_features <= 0:
            max_nr_features += len(self.get_original_feature_names())

        ranked_feature_ids, seen = [], set()
        for idx in ranked_transformed_feature_ids:
            inverse_transformed_feature_id = self.inverse_transform_feature_id(idx)
            if inverse_transformed_feature_id not in seen:
                seen.add(inverse_transformed_feature_id)
                ranked_feature_ids.append(inverse_transformed_feature_id)
                if max_nr_features == len(ranked_feature_ids):
                    return ranked_feature_ids
        return ranked_feature_ids # should never be reached, but just in case

    def inverse_thresholds(self, tree):
        used_feature_mask = tree.feature >= 0
        feats_idx = tree.feature[used_feature_mask]
        thresholds = tree.threshold.astype('O')
        thresh = thresholds[used_feature_mask]
        n_cols = len(self.get_preprocessed_feature_names())

        dummy_x, indices= [], []
        for f, t in zip(feats_idx, thresh):
            row = [0]*n_cols
            row[f] = t
            dummy_x.append(row)
            indices.append(self.inverse_transform_feature_id(f))

        undo_dummy_x = self.inverse_transform(np.array(dummy_x))
        descaled_thresh = [undo_dummy_x[i, j] for i, j in enumerate(indices)]
        thresholds[used_feature_mask] = descaled_thresh
        return thresholds


class DummyPipelinePreprocessor(FeatureNameTransformer):

    def __init__(self, model_performance_predictor_features):
        super(DummyPipelinePreprocessor, self).__init__(
            original_features=model_performance_predictor_features,
            preprocessed_features=model_performance_predictor_features)

    def transform(self, x):
        """
        Args:
            x (array-like or dataframe of shape (number of samples, number of features)): input feature values.
        Returns:
            ndarray
        """
        if isinstance(x, pd.DataFrame):
            return x.values
        if isinstance(x, np.ndarray) or issparse(x):
            return x
        raise TypeError('x should be either a pandas dataframe, a numpy ndarray or a scipy sparse matrix')

    def is_categorical(self, index=None, name=None):
        return False

    def inverse_transform_feature_id(self, index):
        return index

    def inverse_transform(self, x):
        return x

    def get_top_ranked_feature_ids(self, feature_importances, max_nr_features):
        if max_nr_features == 0:
            return np.argsort(- feature_importances)
        return np.argsort(- feature_importances)[:max_nr_features]

    def inverse_thresholds(self, tree):
        return tree.threshold.astype('O')


class ErrorTree(object):

    def __init__(self, error_decision_tree):

        self._estimator = error_decision_tree
        self._leaf_ids = None
        self._impurity = None
        self._quantized_impurity = None
        self._difference = None
        self._total_error_fraction = None
        self._error_class_idx = None
        self._wrongly_predicted_leaves = None
        self._correctly_predicted_leaves = None

        self._check_error_tree()

    @property
    def estimator_(self):
        if self._estimator is None:
            raise NotFittedError("You should fit the ErrorAnalyzer first")
        return self._estimator

    @property
    def impurity(self):
        if self._impurity is None:
            self._impurity = self.correctly_predicted_leaves / (self.wrongly_predicted_leaves + self.correctly_predicted_leaves)
        return self._impurity

    @property
    def quantized_impurity(self):
        if self._quantized_impurity is None:
            purity_bins = np.linspace(0, 1., ErrorAnalyzerConstants.NUMBER_PURITY_LEVELS)
            self._quantized_impurity = np.digitize(self.impurity, purity_bins)
        return self._quantized_impurity

    @property
    def difference(self):
        if self._difference is None:
            self._difference = self.correctly_predicted_leaves - self.wrongly_predicted_leaves  # only negative numbers
        return self._difference

    @property
    def total_error_fraction(self):
        if self._total_error_fraction is None:
            self._total_error_fraction = self.wrongly_predicted_leaves / self.n_total_errors
        return self._total_error_fraction

    @property
    def error_class_idx(self):
        if self._error_class_idx is None:
            self._error_class_idx = np.where(self.estimator_.classes_ == ErrorAnalyzerConstants.WRONG_PREDICTION)[0][0]
        return self._error_class_idx

    @property
    def n_total_errors(self):
        return self.estimator_.tree_.value[0, 0, self.error_class_idx]

    @property
    def wrongly_predicted_leaves(self):
        if self._wrongly_predicted_leaves is None:
            self._wrongly_predicted_leaves = self.estimator_.tree_.value[self.leaf_ids, 0, self.error_class_idx]
        return self._wrongly_predicted_leaves

    @property
    def correctly_predicted_leaves(self):
        if self._correctly_predicted_leaves is None:
            self._correctly_predicted_leaves = self.estimator_.tree_.value[self.leaf_ids, 0, 1 - self.error_class_idx]
        return self._correctly_predicted_leaves

    @property
    def leaf_ids(self):
        if self._leaf_ids is None:
            self._leaf_ids = np.where(self.estimator_.tree_.feature < 0)[0]
        return self._leaf_ids

    def _check_error_tree(self):
        if self.estimator_.tree_.node_count == 1:
            logger.warning("The error tree has only one node, there will be problems when using it with ErrorVisualizer")