"""
QuTiP / tncontract conversions.

Functionality for converting between `qutip.Qobj` and `Tensor`.

Requires `qutip`.
"""

import numpy as np

# import qutip as qt

import tncontract as tn


def qobj_to_tensor(qobj, labels=None):
    """
    Convert a `qutip.Qobj` object to a `Tensor`

    Parameters
    ----------
    qobj : `Qobj`
        Qobj to convert.
    labels : list, optional
        List of labels for the indices. Output labels followed by input labels.
        Defaults to `['physout1', ..., 'physoutn', 'physin1', ..., 'physinn']

    Notes
    -----
    Dummy indices of dimension one will be trimmed away.
    """

    data = qobj.data.toarray()

    if not len(np.shape(qobj.dims)) == 2:
        # wrong dims (not a ket, bra or operator)
        raise ValueError("qobj element not a ket/bra/operator")

    physout = qobj.dims[0]
    physin = qobj.dims[1]
    nsys = len(physout)
    if labels is None:
        physoutstr = ['physout'+str(k) for k in range(nsys)]
        physinstr = ['physin'+str(k) for k in range(nsys)]
    else:
        physoutstr = labels[:nsys]
        physinstr = labels[nsys:]

    t = tn.Tensor(np.reshape(data, physout+physin), physoutstr+physinstr)
    t.remove_all_dummy_indices()
    return t


def tensor_to_qobj(tens, input_labels, output_labels):
    """
    Convert a `Tensor` object to a `qutip.Qobj`

    Parameters
    ----------
    tens : `Tensor`
        Tensor to convert.
    input_labels : list
    output_labels : list
    """

    return

