# -*- coding: utf-8 -*-


class ErrorAnalyzerConstants(object):

    PREDICTION_COLUMN = 'prediction'
    ERROR_COLUMN = '__dku_error__'
    WRONG_PREDICTION = "Wrong prediction"
    CORRECT_PREDICTION = "Correct prediction"
    MAX_DEPTH_GRID = [5, 10, 15, 20, 30, 50]

    MIN_NUM_ROWS = 500  # heuristic choice
    MAX_NUM_ROW = 100000  # heuristic choice

    MPP_ACCURACY_TOLERANCE = 0.1
    CRITERION = 'entropy'
    NUMBER_EPSILON_VALUES = 50

    ERROR_TREE_COLORS = {CORRECT_PREDICTION: '#ed6547', WRONG_PREDICTION: '#fdc765'}
