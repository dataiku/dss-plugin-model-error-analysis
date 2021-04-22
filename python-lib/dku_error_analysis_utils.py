import sys

def safe_str(val):
    if sys.version_info > (3, 0):
        return str(val)
    if isinstance(val, unicode):
        return val.encode("utf-8")
    return str(val)

class DkuMEAConstants(object):
    ERROR_COLUMN = "__dku_error__"
    MAX_NUM_ROWS = 100000
