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


def package_is_at_least(p, min_version):
    from distutils.version import LooseVersion
    return LooseVersion(p.__version__) >= LooseVersion(min_version)