from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *


"""
test_canonical_form_mps_conversion
==========

Unit tests for converting between different types of canonical form MPS
"""

import os
import sys
import numpy.testing as testing
import pickle
import tncontract as tnc

pwd = os.path.dirname(__file__)

if sys.version_info > (3, 0):
    f = open(pwd + "/random_10site_mps.dat", "rb")
    psi = pickle.load(f, encoding="latin1")
    f.close()
else:
    f = open(pwd + "/random_10site_mps_py2.dat", "rb")
    psi = pickle.load(f)
    f.close()


def test_right_and_left_canonical_to_canonical():
    # SVD compress leaves MPS in right-canonical form
    # Test for unnormalized MPS
    psi.svd_compress(threshold=1e-12, normalise=False)
    psicr = tnc.onedim.right_canonical_to_canonical(psi, threshold=1e-12)
    testing.assert_almost_equal(psi.norm(), psicr.norm(), decimal=10)
    l, r, n = psicr.check_canonical_form(threshold=1e-10, print_output=False)
    testing.assert_equal(len(l), 0)
    testing.assert_equal(len(r), 0)
    testing.assert_equal(len(n), 1)
    psi.left_canonise()
    psicl = tnc.onedim.left_canonical_to_canonical(psi, threshold=1e-12)
    testing.assert_almost_equal(psi.norm(), psicr.norm(), decimal=10)
    l, r, n = psicr.check_canonical_form(threshold=1e-10, print_output=False)
    testing.assert_equal(len(l), 0)
    testing.assert_equal(len(r), 0)
    testing.assert_equal(len(n), 1)
    testing.assert_almost_equal(tnc.onedim.inner_product_mps(psicr, psicl),
                                psi.norm() ** 2, decimal=10)
    # Test for normalized MPS
    psi.svd_compress(threshold=1e-12, normalise=True)
    psicr = tnc.onedim.right_canonical_to_canonical(psi, threshold=1e-12)
    l, r, n = psicr.check_canonical_form(threshold=1e-10, print_output=False)
    testing.assert_equal(len(l), 0)
    testing.assert_equal(len(r), 0)
    testing.assert_equal(len(n), 0)
    psi.left_canonise()
    psicl = tnc.onedim.left_canonical_to_canonical(psi, threshold=1e-12)
    l, r, n = psicr.check_canonical_form(threshold=1e-10, print_output=False)
    testing.assert_equal(len(l), 0)
    testing.assert_equal(len(r), 0)
    testing.assert_equal(len(n), 0)
    testing.assert_almost_equal(tnc.onedim.inner_product_mps(psicr, psicl),
                                1.0, decimal=10)


def test_right_canonical_to_canonical_and_back():
    # SVD compress leaves MPS in right-canonical form
    psi.svd_compress(threshold=1e-12, normalise=True)
    psic = tnc.onedim.right_canonical_to_canonical(psi, threshold=1e-12)
    psir = tnc.onedim.canonical_to_right_canonical(psic)
    psil = tnc.onedim.canonical_to_left_canonical(psic)
    testing.assert_almost_equal(tnc.onedim.inner_product_mps(psi, psil), 1.0,
                                decimal=10)
    testing.assert_almost_equal(tnc.onedim.inner_product_mps(psi, psir), 1.0,
                                decimal=10)
    l, r = psir.check_canonical_form(threshold=1e-10, print_output=False)
    testing.assert_equal(r, 0)
    l, r = psil.check_canonical_form(threshold=1e-10, print_output=False)
    testing.assert_equal(l, len(psi) - 1)
