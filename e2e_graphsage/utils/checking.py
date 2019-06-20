"""
Functions some very common checks
"""
from six import integer_types, string_types, text_type
from collections import Mapping, Iterable


def is_integer(x):
    return isinstance(x, integer_types)


def is_positive_integer(x):
    return is_integer(x) and x > 0


def is_non_negative_integer(x):
    return is_integer(x) and x >= 0


def is_string(x):
    return isinstance(x, string_types)


def is_dict(x):
    return isinstance(x, Mapping)


def is_list_like(x):
    return isinstance(x, Iterable) and not is_string(x)
