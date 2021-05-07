import sys

def safe_str(val):
    if sys.version_info > (3, 0):
        return str(val)
    if isinstance(val, unicode):
        return val.encode("utf-8")
    return str(val)

def format_float(number, decimals):
    """
    Format a number to have the required number of decimals. Ensure no trailing zeros remain.
    Args:
        number (float or int): The number to format
        decimals (int): The number of decimals required
    Return:
        formatted (str): The number as a formatted string
    """
    formatted = ("{:." + str(decimals) + "f}").format(number).rstrip("0")
    if formatted.endswith("."):
        return formatted[:-1]
    return formatted

class DkuMEAConstants(object):
    ERROR_COLUMN = "__dku_error__"
    MAX_NUM_ROWS = 100000
