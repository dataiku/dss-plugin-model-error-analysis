#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    Simple functions helpers
"""


def not_enough_data(df, min_len=1):
    """
        Compare length of dataframe to minimum lenght of the test data.
        Used in the relevance of the measure.
    :param df: Input dataframe
    :param min_len:
    :return:
    """
    return len(df) < min_len
