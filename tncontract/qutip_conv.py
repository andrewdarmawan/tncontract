"""
QuTiP / tncontract conversions.

Functionality for converting between `qutip.Qobj` and `Tensor`.

Requires `qutip`.
"""

import numpy as np

import qutip as qt

import tncontract as tn


def qobj_to_tensor(qobj, labels=None, trim_dummy=True):
    """
    Convert a `qutip.Qobj` object to a `Tensor`

    Parameters
    ----------
    qobj : `Qobj`
        Qobj to convert.
    labels : list, optional
        List of labels for the indices. Output labels followed by input labels.
        Defaults to `['out1', ..., 'outk', 'in1', ..., 'ink']
    trim_dummy : bool
        If true dummy indices of dimension one are trimmed away
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
    t = tn.matrix_to_tensor(data, output_dims, input_dims, output_labels,
            input_labels)
    if trim_dummy:
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


def qobjlist_to_mpo(qobjlist):
    """
    Construct an MPO from a list of Qobj operators.

    Many-body operators are put in MPO form by exact SVD, and virtual "left"
    and "right" indices with bond dimension one are added between the elements
    of the list.
    """
    tensors = []
    for i, qobj in enumerate(qobjlist):
        if not len(np.shape(qobj.dims)) == 2:
            # wrong dims (not a ket, bra or operator)
            raise ValueError("qobj element not a ket/bra/operator")

        nsys = len(qobj.dims[0])
        t = qobj_to_tensor(qobj, trim_dummy=False)

        # Add left and right indices with bonddim one
        t.add_dummy_index('left', -1)
        t.add_dummy_index('right', -1)

        # Break up many-body operators by SVDing
        for k in range(nsys-1):
            U, S, V = tn.tensor_svd(t, ['out'+str(k), 'in'+str(k), 'left'])
            U.replace_label('svd_in', 'right')
            U.replace_label('out'+str(k), 'physout')
            U.replace_label('in'+str(k), 'physin')
            tensors.append(U)
            t = tn.contract(S, V, ['svd_in'], ['svd_out'])
            t.replace_label('svd_out', 'left')
        t.replace_label('out'+str(nsys-1), 'physout')
        t.replace_label('in'+str(nsys-1), 'physin')
        tensors.append(t)
    return tn.MatrixProductOperator(tensors, left_label='left',
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
        t.remove_all_dummy_indices(labels=['physin'])
        # Change physical label to the standard choice 'phys'
        t.replace_label(['physout'], ['phys'])
    return tn.MatrixProductState(tensors, left_label='left',
        right_label='right', phys_label='phys')
