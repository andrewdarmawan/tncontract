from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *


"""
qutip_conv
==========

QuTiP / tncontract conversions.

Functionality for converting between `qutip.Qobj` and `Tensor`.

Requires `qutip`.
"""

import numpy as np

import qutip as qt

import tncontract as tn
import tncontract.onedim as onedim


def qobj_to_tensor(qobj, labels=None, trim_dummy=True):
    """
    Convert a `qutip.Qobj` object to a `Tensor`

    Parameters
    ----------
    qobj : Qobj
        Qobj to convert.
    labels : list, optional
        List of labels for the indices. Output labels followed by input labels.
        Defaults to `['out1', ..., 'outk', 'in1', ..., 'ink']
    trim_dummy : bool
        If true dummy indices of dimension one are trimmed away

    Returns
    ------
    Tensor
    """

    data = qobj.data.toarray()

    if not len(np.shape(qobj.dims)) == 2:
        # wrong dims (not a ket, bra or operator)
        raise ValueError("qobj element not a ket/bra/operator")

    output_dims = qobj.dims[0]
    input_dims = qobj.dims[1]
    nsys = len(output_dims)
    if labels is None:
        output_labels = ['out'+str(k) for k in range(nsys)]
        input_labels = ['in'+str(k) for k in range(nsys)]
    else:
        output_labels = labels[:nsys]
        input_labels = labels[nsys:]
    t = tn.matrix_to_tensor(data, output_dims+input_dims, output_labels+
            input_labels)
    if trim_dummy:
        t.remove_all_dummy_indices()
    return t


def tensor_to_qobj(tensor, output_labels, input_labels):
    """
    Convert a `Tensor` object to a `qutip.Qobj`

    Parameters
    ----------
    tensor : Tensor
        Tensor to convert.
    output_labels : list
        List of labels that will be the output indices for the `Qobj`.
        `None` can be used to insert a dummy index of dimension one.
    inpul_labels : list
        List of labels that will be the input indices for the `Qobj`.
        `None` can be used to insert a dummy index of dimension one.

    Returns
    -------
    Qobj

    Notes
    -----
    The `output_labels` and `input_labels` determines the tensor product
    structure of the resulting `Qobj`, inclding the order of the components.
    If the indices corresponding to `output_labels` have dimensions 
    [dim_out1, ..., dim_outk] and the indices corresponding to `input_labels` 
    have dimensions [dim_in1, ..., dim_inl], the `Qobj.dims` attribute will be
    `Qobj.dims = [[dim_out1, ..., dim_outk], [dim_in1, ..., dim_inl]]

    Examples
    --------
    Turn a rank-one vector into a ket `Qobj` (note the use of a `None` input
    label to get a well defined `Qobj`)
    >>> t = Tensor(np.array([1,0]), labels=['idx1'])
    >>> q = tensor_to_qobj(t, ['idx1'], [None])
    >>> print(q)
    Quantum object: dims = [[2], [1]], shape = [2, 1], type = ket
    Qobj data =
    [[ 1.]
     [ 0.]]
    """

    output_dims = []
    input_dims = []
    t = tensor.copy()

    if not isinstance(output_labels, list):
        output_labels=[output_labels]
    if not isinstance(input_labels, list):
        input_labels=[input_labels]
    # order the indices according to output_labels and input_labels
    for i, label in enumerate(output_labels+input_labels):
        if label is None:
            label = 'dummy'+str(i)
            t.add_dummy_index(label, i)
        t.move_index(label, i)
        if i < len(output_labels):
            output_dims.append(t.shape[i])
        else:
            input_dims.append(t.shape[i])

    output_labels_new = [l if l is not None else 'dummy'+str(i)
            for i,l in enumerate(output_labels)]

    data = tn.tensor_to_matrix(t, output_labels_new)
    dims = [output_dims, input_dims]
    return qt.Qobj(data, dims=dims)


def qobjlist_to_mpo(qobjlist):
    """
    Construct an MPO from a list of Qobj operators.

    Many-body operators are put in MPO form by exact SVD, and virtual "left"
    and "right" indices with bond dimension one are added between the elements
    of the list.
    """
    tensors = np.array([])
    for i, qobj in enumerate(qobjlist):
        if not len(np.shape(qobj.dims)) == 2:
            # wrong dims (not a ket, bra or operator)
            raise ValueError("qobj element not a ket/bra/operator")

        t = qobj_to_tensor(qobj, trim_dummy=False)

        # Add left and right indices with bonddim one
        t.add_dummy_index('left', -1)
        t.add_dummy_index('right', -1)

        # Break up many-body operators by SVDing
        tmp_mpo = onedim.tensor_to_mpo(t)

        tensors = np.concatenate((tensors, tmp_mpo.data))
    return onedim.MatrixProductOperator(tensors, left_label='left',
        right_label='right', physin_label='physin', physout_label='physout')


def qobjlist_to_mps(qobjlist):
    """
    Construct an MPS from a list of Qobj kets.

    Many-body states are put in MPS form by exact SVD, and virtual "left"
    and "right" indices with bond dimension one are added between the elements
    of the list.
    """
    mpo = qobjlist_to_mpo(qobjlist)
    tensors = mpo.data
    for t in tensors:
        # Remove dummy input labels
        t.remove_all_dummy_indices(labels=[mpo.physin_label])
        # Change physical label to the standard choice 'phys'
        t.replace_label(mpo.physout_label, 'phys')
    return onedim.MatrixProductState(tensors, left_label='left',
        right_label='right', phys_label='phys')
