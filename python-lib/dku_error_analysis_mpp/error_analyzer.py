# -*- coding: utf-8 -*-
import numpy as np
from sklearn import tree
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from dku_error_analysis_mpp.kneed import KneeLocator
from dku_error_analysis_utils.dataframe_helpers import not_enough_data
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='Error Analysis Plugin | %(levelname)s - %(message)s')

PREDICTION_COLUMN = 'prediction'
ERROR_COLUMN = '__dku_error__'
WRONG_PREDICTION = "Wrong prediction"
CORRECT_PREDICTION = "Correct prediction"
MAX_DEPTH_GRID = [5, 10, 15, 20, 30, 50]

MIN_NUM_ROWS = 500  # heuristic choice
MAX_NUM_ROW = 100000  # heuristic choice

MPP_ACCURACY_TOLERANCE = 0.1


class ErrorAnalyzer:
    """
    ErrorAnalyzer analyzes the errors of a prediction models on a test set.
    It uses model predictions and ground truth target to compute the model errors on the test set.
    It then trains a Decision Tree on the same test set by using the model error as target.
    The nodes of the decision tree are different segments of errors to be studied individually.
    """

    def __init__(self, model_accessor):

        if model_accessor is not None:
            self._model_accessor = model_accessor
            self.target = self._model_accessor.get_target_variable()
            self.prediction_type = self._model_accessor.get_prediction_type()
        else:
            raise NotImplementedError('You need to precise a model accessor.')

        self._model_accessor = model_accessor
        self._error_df = None
        self.error_clf = None
        self._error_test_X = None
        self._error_test_Y = None

        self.features_in_model_performance_predictor = None

        self.mpp_accuracy_score = None
        self.primary_model_predicted_accuracy = None
        self.primary_model_true_accuracy = None

    def get_model_performance_predictor_test_df(self):
        return self._error_df

    def get_model_performance_predictor_features(self):
        return self.features_in_model_performance_predictor

    def get_model_performance_predictor(self):
        return self.error_clf

    def get_mpp_accuracy_score(self):
        return self.mpp_accuracy_score

    def get_primary_model_predicted_accuracy(self):
        return self.primary_model_predicted_accuracy

    def get_primary_model_true_accuracy(self):
        return self.primary_model_true_accuracy

    def fit(self):
        """
        Trains a Decision Tree to discriminate between samples that are correctly predicted or wrongly predicted
        (errors) by a primary model.
        """
        logger.info("Preparing the model performance predictor...")

        original_df = self._model_accessor.get_original_test_df()

        predictor = self._model_accessor.get_predictor()
        preprocessed_x, _, _, _, _ = predictor.preprocessing.preprocess(
            original_df,
            with_target=True,
            with_sample_weights=True)

        self._error_df = self.prepare_data_for_model_performance_predictor(original_df)

        error_y = np.array(self._error_df[ERROR_COLUMN])
        error_train_X, error_test_X, error_train_Y, error_test_Y = train_test_split(
            preprocessed_x,
            error_y,
            test_size=0.2
        )

        self._error_test_X = error_test_X  # we will use them later when compute metrics
        self._error_test_Y = error_test_Y
        self.features_in_model_performance_predictor = predictor.get_features()

        logger.info("Fitting the model performance predictor...")

        # entropy/mutual information is used to split nodes in Microsoft Pandora system
        criterion = 'entropy'

        dt_clf = tree.DecisionTreeClassifier(criterion=criterion, min_samples_leaf=1)
        parameters = {'max_depth': MAX_DEPTH_GRID}
        gs_clf = GridSearchCV(dt_clf, parameters, cv=5)
        gs_clf.fit(error_train_X, error_train_Y)
        self.error_clf = gs_clf.best_estimator_

        logger.info('Grid search selected max_depth = {}'.format(gs_clf.best_params_['max_depth']))

        self.compute_model_performance_predictor_metrics()

        return

    def prepare_data_for_model_performance_predictor(self, original_df):
        """
        Computes the errors of the primary model predictions and samples with max n = MAX_NUM_ROW
        :return: a dataframe with error target (correctly predicted vs wrongly predicted)
        """

        logger.info('Prepare data with model for model performance predictor')

        if self.target not in original_df:
            raise ValueError('The original dataset does not contain target "{}".'.format(self.target))

        if not_enough_data(original_df, min_len=MIN_NUM_ROWS):
            raise ValueError(
                'The original dataset is too small ({} rows) to have stable result, it needs to have at least {} rows'.format(
                    len(original_df), MIN_NUM_ROWS))

        logger.info("Rebalancing data:")
        number_of_rows = min(original_df.shape[0], MAX_NUM_ROW)
        logger.info(" - original dataset had %s rows. Selecting the first %s." % (original_df.shape[0], number_of_rows))

        df = original_df.head(number_of_rows)

        # get prediction, is there a way to get predictions from the model handler?
        if PREDICTION_COLUMN not in df.columns:
            df[PREDICTION_COLUMN] = self._model_accessor.predict(df)

        df[ERROR_COLUMN] = self._get_errors(df, prediction_column=PREDICTION_COLUMN)
        df[ERROR_COLUMN] = df[ERROR_COLUMN].replace({True: WRONG_PREDICTION, False: CORRECT_PREDICTION})

        selected_features = [ERROR_COLUMN] + self._model_accessor.get_selected_features()

        return df.loc[:, selected_features]

    def compute_model_performance_predictor_metrics(self):
        """
        MPP ability to predict primary model performance. Ideally primary_model_predicted_accuracy should be equal
        to primary_model_true_accuracy
        """

        y_true = self._error_test_Y
        y_pred = self.error_clf.predict(self._error_test_X)
        n_test_samples = y_pred.shape[0]
        self.mpp_accuracy_score = accuracy_score(y_true, y_pred)
        logger.info('Model Performance Predictor accuracy: {}'.format(self.mpp_accuracy_score))

        self.primary_model_predicted_accuracy = float(np.count_nonzero(y_pred == CORRECT_PREDICTION)) / n_test_samples
        self.primary_model_true_accuracy = float(np.count_nonzero(y_true == CORRECT_PREDICTION)) / n_test_samples

        logger.info('Primary model accuracy: {}'.format(self.primary_model_true_accuracy))
        logger.info('MPP predicted accuracy: {}'.format(self.primary_model_predicted_accuracy))

        difference_true_pred_accuracy = np.abs(self.primary_model_true_accuracy - self.primary_model_predicted_accuracy)
        if difference_true_pred_accuracy > MPP_ACCURACY_TOLERANCE:  # TODO: add message in UI?
            logger.info("Warning: the built MPP might not be representative of the primary model performances.")

    def _get_epsilon(self, difference, mode='rec'):
        """ compute epsilon to define errors in regression task """
        assert (mode in ['std', 'rec'])
        if mode == 'std':
            std_diff = np.std(difference)
            mean_diff = np.mean(difference)
            epsilon = mean_diff + std_diff
        elif mode == 'rec':
            n_points = 50
            epsilon_range = np.linspace(min(difference), max(difference), num=n_points)
            cdf_error = np.zeros_like(epsilon_range)
            n_samples = difference.shape[0]
            for i, epsilon in enumerate(epsilon_range):
                correct = difference <= epsilon
                cdf_error[i] = float(np.count_nonzero(correct)) / n_samples
            kneedle = KneeLocator(epsilon_range, cdf_error)
            epsilon = kneedle.knee
        return epsilon

    def _get_errors(self, test_df, prediction_column):
        """ compute errors of the primary model on the test set """
        assert (self.prediction_type in ["REGRESSION", "BINARY_CLASSIFICATION", "MULTICLASS"])
        if self.prediction_type == "REGRESSION":
            target = test_df[self.target]
            predictions = test_df[prediction_column]
            difference = np.abs(target - predictions)

            epsilon = self._get_epsilon(difference, mode='rec')

            error = difference > epsilon
            return error

        error = (test_df[self.target] != test_df[prediction_column])
        return np.array(error)

