"""
QuTiP / tncontract conversions.

Functionality for converting between `qutip.Qobj` and `Tensor`.

Requires `qutip`.
"""

import numpy as np

import qutip as qt

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

    output_dims = qobj.dims[0]
    input_dims = qobj.dims[1]
    nsys = len(output_dims)
    if labels is None:
        output_labels = ['physout'+str(k) for k in range(nsys)]
        input_labels = ['physin'+str(k) for k in range(nsys)]
    else:
        output_labels = labels[:nsys]
        input_labels = labels[nsys:]

    t = tn.matrix_to_tensor(data, output_dims, input_dims, output_labels,
            input_labels)
    t.remove_all_dummy_indices()
    return t


def tensor_to_qobj(tensor, output_labels):
    """
    Convert a `Tensor` object to a `qutip.Qobj`

    Parameters
    ----------
    tensor : `Tensor`
        Tensor to convert.
    input_labels : list
    output_labels : list
    """

    output_dims = []
    for label in output_labels:
        output_dims.append(tensor.index_dimension(label))

    input_labels=[x for x in tensor.labels if x not in output_labels]
    input_dims = []
    for label in input_labels:
        input_dims.append(tensor.index_dimension(label))

    data = tn.tensor_to_matrix(tensor, output_labels)
    dims = [output_dims, input_dims]
    return qt.Qobj(data, dims=dims)

