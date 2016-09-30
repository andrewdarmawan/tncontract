"""
tncontract
==========

A simple tensor-network library.

Available subpackages
---------------------
onedimensional
    Special purpose classes, methods and function for one dimensional
    tensor networs

twodimensional
    Special purpose classes, methods and function for two dimensional
    tensor networs
"""

from tncontract.version import __version__
from tncontract.tensor import (Tensor, contract, distance, matrix_to_tensor,
        tensor_to_matrix, random_tensor, tensor_product, tensor_svd,
        truncated_svd, unique_label, zeros_tensor)
from tncontract import onedimensional
from tncontract import twodimensional
