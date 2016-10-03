"""
tncontract
==========

A simple tensor-network library.

Available subpackages
---------------------
onedim
    Special purpose classes, methods and function for one dimensional
    tensor networs

twodim
    Special purpose classes, methods and function for two dimensional
    tensor networs
"""

from tncontract.version import __version__
from tncontract.tensor import *
from tncontract.label import *
import tncontract.matrices
import tncontract.onedim
import tncontract.twodim
