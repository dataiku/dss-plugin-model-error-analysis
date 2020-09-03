import sys

def safe_str(val):
    if sys.version_info > (3, 0):
        return str(val)
    if isinstance(val, unicode):
        return val.encode("utf-8")
    return str(val)

def not_enough_data(df, min_len=1):
    """
    Compare length of dataframe to minimum lenght of the test data.
    Used in the relevance of the measure.

    :param df: Input dataframe
    :param min_len:
    :return:
    """
    return len(df) < min_len
class ErrorAnalyzerConstants(object):
    
    PREDICTION_COLUMN = 'prediction'
    ERROR_COLUMN = '__dku_error__'
    WRONG_PREDICTION = "Wrong prediction"
    CORRECT_PREDICTION = "Correct prediction"
    MAX_DEPTH_GRID = [5, 10, 15, 20, 30, 50]
    TEST_SIZE = 0.2

    MIN_NUM_ROWS = 500  # heuristic choice
    MAX_NUM_ROW = 100000  # heuristic choice

    MPP_ACCURACY_TOLERANCE = 0.1
    CRITERION = 'entropy'
    NUMBER_EPSILON_VALUES = 50

    ERROR_TREE_COLORS = {CORRECT_PREDICTION: '#ed6547', WRONG_PREDICTION: '#fdc765'}

    TOP_K_FEATURES = 3

    MPP_ACCURACY = 'mpp_accuracy_score'
    PRIMARY_MODEL_TRUE_ACCURACY = 'primary_model_true_accuracy'
    PRIMARY_MODEL_PREDICTED_ACCURACY = 'primary_model_predicted_accuracy'
    CONFIDENCE_DECISION = 'confidence_decision'

    NUMBER_PURITY_LEVELS = 10

    CATEGORICAL_OTHERS = 'Others'

def get_rgb_with_alpha(color_hex, alpha):
    rgb_color = lambda i: int(color_hex[i:i + 2], 16)
    color_rgb = [int(round(alpha * rgb_color(i) + (1 - alpha) * 255, 0)) for i in (1,3,5)]
    return '#{:02x}{:02x}{:02x}'.format(*color_rgb)
