"""
test_canonical_form_mps_conversion
==========

Unit tests for converting between different types of canonical form MPS
"""

import tncontract as tnc
import numpy.testing as testing

def test_right_canonical_to_canonical_and_back():
    N = 100
    psi = tnc.onedim.init_mps_random(N, 5, 5)
    # SVD compress leaves MPS in right-canonical form
    psi.svd_compress(threshold=1e-12, normalise=True)
    # Convert to canonical form MPS
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
    testing.assert_equal(l, N-1)


