from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

"""
label
==========

Module for string-like labels
"""

__all__ = ['prime_label', 'unprime_label', 'noprime_label', 'prime_level',
           'unique_label']

import uuid


class Label(str):
    """Wrapper class for priming labels"""

    def __new__(cls, value, **kwargs):
        return str.__new__(cls, value)

    def __init__(self, label, parent=None):
        self._parent = parent
        if isinstance(label, Label):
            # if input is a Label object copy its properties
            if parent is None:
                self._parent = label._parent

    @property
    def parent(self):
        """Return parent label"""
        return self._parent

    @property
    def origin(self):
        """Return origin label"""
        origin = self
        while hasattr(origin, "parent"):
            origin = origin.parent
        return origin

    @property
    def parents(self):
        """Return number of parents for label"""
        tmp = self
        level = 0
        while hasattr(tmp, "parent"):
            if tmp.parent is not None:
                tmp = tmp.parent
                level += 1
            else:
                break
        return level


def prime_label(label, prime="'"):
    """Put a prime on a label object"""
    return Label(str(label) + prime, parent=label)


def unprime_label(label, prime="'"):
    """Remove one prime from label object"""
    try:
        parent = label.parent
    except AttributeError:
        raise ValueError("label is not primed")
    if str(parent) + prime == label:
        return parent
    else:
        raise ValueError("label is not primed with \"" + prime + "\"")


def noprime_label(label):
    """Remove all primes from a label object"""
    try:
        return label.origin
    except AttributeError:
        return label


def prime_level(label):
    """Return number of primes on label object"""
    try:
        return label.parents
    except AttributeError:
        return 0


def unique_label():
    """Generate a long, random string that is very likely to be unique."""
    return str(uuid.uuid4())
