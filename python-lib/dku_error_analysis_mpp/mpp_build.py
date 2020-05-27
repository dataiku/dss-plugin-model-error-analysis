# -*- coding: utf-8 -*-
import dataiku
import logging
import numpy as np
from sklearn import tree
from dku_error_analysis_mpp.kneed import KneeLocator
from dku_error_tree_parsing.tree_parser import TreeParser
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="MEA %(levelname)s - %(message)s")

PREDICT_CORRECT, PREDICT_WRONG = 0, 1

# Should be like the number of data segments the user is ready to review.
MAX_LEAF_NODES = 20


# compute epsilon to define errors in regression task
def get_epsilon(difference, mode='std'):
    assert(mode in ['std', 'rec'])
    if mode == 'std':
        std_diff = np.std(difference)
        mean_diff = np.mean(difference)
        epsilon = mean_diff + std_diff
    elif mode == 'rec':
        n_points = 50
        epsilon_range = np.linspace(np.min(difference), np.max(difference), num=n_points)
        cdf_error = np.zeros_like(epsilon_range)
        n_samples = difference.shape[0]
        for i, epsilon in enumerate(epsilon_range):
            correct = difference <= epsilon
            cdf_error[i] = float(np.count_nonzero(correct))/n_samples
        kneedle = KneeLocator(epsilon_range, cdf_error, S=1.0, curve='concave', direction='increasing')
        epsilon = kneedle.knee
    return epsilon


# compute errors of the primary model on the test set
def get_errors(task, test_df, target_column, prediction_column):
    assert (task in ["REGRESSION", "BINARY_CLASSIFICATION", "MULTICLASS"])
    if task == "REGRESSION":
        target = np.array(test_df[target_column])
        pred = np.array(test_df[prediction_column])
        difference = np.abs(target - pred)

        epsilon = get_epsilon(difference, mode='rec')

        error = difference > epsilon
        return error

    error = (test_df[target_column] != test_df[prediction_column])
    return np.array(error)


# get test set features and target needed to fit a DT as a Model Performance Predictor
def get_features_and_errors(model_handler, test_df):
    target_column = model_handler.get_target_variable()
    task = model_handler.get_prediction_type()
    pred = model_handler.get_predictor()

    if 'prediction' not in test_df.columns:
        test_df['prediction'] = pred.predict(test_df, with_probas=False)

    transformed_df, _, _, _, _ = pred.preprocessing.preprocess(
        test_df,
        with_target=True,
        with_sample_weights=True)

    error = get_errors(task, test_df, target_column, prediction_column='prediction')
    feature_names = pred.get_features()

    return transformed_df, error, feature_names


# fit a DT as a Model Performance Predictor
def build_mpp(transformed_df, error, grid_search=True):
    # fit a Decision Tree
    X = transformed_df
    Y = error
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=42)
    criterion = 'entropy'

    if grid_search:
        mpp_clf = tree.DecisionTreeClassifier(criterion=criterion, min_samples_leaf=1)
        parameters = {'max_depth': [5, 10, 15, 20, 30, 50]}
        gs_clf = GridSearchCV(mpp_clf, parameters)
        gs_clf.fit(X_train, Y_train)
        mpp_clf = gs_clf.best_estimator_

        logger.info('Grid search selected max_depth = {}'.format(gs_clf.best_params_['max_depth']))

    else:
        mpp_clf = tree.DecisionTreeClassifier(
            criterion=criterion,
            min_samples_leaf=1,
            max_leaf_nodes=MAX_LEAF_NODES,
            min_impurity_decrease=1e-7)
        mpp_clf.fit(X_train, Y_train)

    # MPP ability to predict primary model performance. Ideally mpp_score should be equal to ref_score
    ref_score = float(np.count_nonzero(Y_test == PREDICT_CORRECT)) / Y_test.shape[0]
    Y_pred = mpp_clf.predict(X_test)
    mpp_score = float(np.count_nonzero(Y_pred == PREDICT_CORRECT)) / Y_pred.shape[0]
    tolerance = 0.1

    logger.info('Primary model accuracy on left-out test set: {}'.format(ref_score))
    logger.info('MPP predicted accuracy: {}'.format(mpp_score))

    if np.abs(ref_score - mpp_score) > tolerance: #TODO: add message in UI?
        logger.info("Warning: the built MPP might not be representative of the primary model performances.")

    return mpp_clf


def get_error_dt(model_handler):
    test_df, res = model_handler.get_test_df()
    if not res:
        raise Exception("Error loading test set")
    transformed_df, error, feature_names = get_features_and_errors(model_handler, test_df)
    mpp_clf = build_mpp(transformed_df, error, grid_search=True)

    test_df['error'] = error
    test_df['error'] = test_df['error'].replace({True: "Wrong prediction", False: "Correct prediction"})

    return mpp_clf, test_df, transformed_df, feature_names


# rank features according to their correlation with the model performance
def rank_features_by_error_correlation(clf, feature_names, tree_parser, max_number_histograms=3):
    feature_idx_by_importance = np.argsort(-clf.feature_importances_)
    ranked_features = []
    for feature_idx in feature_idx_by_importance:
        preprocessed_name = feature_names[feature_idx]
        feature = tree_parser.get_preprocessed_feature_details(preprocessed_name)[1]
        if feature not in ranked_features:
            ranked_features.append(feature)
            if len(ranked_features) == max_number_histograms:
                return ranked_features
    return ranked_features
