from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

"""
matrices
==========

Often used matrices
"""

import numpy as np


#
# Pauli spin 1/2 operators:
#
def sigmap():
    return np.matrix([[0., 1.], [0., 0.]])


def sigmam():
    return np.matrix([[0., 0.], [1., 0.]])


def sigmax():
    return sigmam() + sigmap()


def sigmay():
    return -1j * sigmap() + 1j * sigmam()


def sigmaz():
    return np.matrix([[1., 0.], [0., -1.]])


def destroy(dim):
    """
    Destruction (lowering) operator.

    Parameters
    ----------
    dim : int
        Dimension of Hilbert space.
    """
    return np.matrix(np.diag(np.sqrt(range(1, dim)), 1))


def create(dim):
    """
    Creation (raising) operator.

    Parameters
    ----------
    dim : int
        Dimension of Hilbert space.
    """
    return destroy(dim).getH()


def identity(dim):
    """
    Identity operator

    Parameters
    ----------
    dim : int
        Dimension of Hilbert space.

    """
    return np.matrix(np.identity(dim))


def basis(dim, i):
    """
    dim x 1 column vector with all zeros except a one at row i
    """
    vec = np.zeros(dim)
    vec[i] = 1.0
    return np.matrix(vec).T
