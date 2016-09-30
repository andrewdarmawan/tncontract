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
from tncontract.tensor import *
import tncontract.onedim
import tncontract.twodim
