"""
onedim_utils
==========

Module with various functions for MPS/MPOs.
"""

import numpy as np


from tncontract import tensor as tsr
from tncontract import onedim as od


def init_mps_random(nsites, physdim, bonddim=1, left_label='left',
        right_label='right', phys_label='phys'):
    """
    Create an MPS with `nsites` sites and random tensors with physical 
    dimensions given by `physdim` and bond dimensions given by
    `bonddim`. Open boundary conditions are used. The MPS is not normalized.

    Parameters
    ----------
    nsites : int
    physdim : int or list of ints
    bonddim : int or list of ints, optional
        The nth element of `bonddim` determines the right and left index of
        the tensors at sites n and n+1, respectively. The length of `bonddim`
        should be `nsites`-1. If `bonddim` is an int this is this is used for
        all bonds.
    left_label : str
    right_label : str
    phys_label : str
    """
    if not np.iterable(physdim):
        physdim = [physdim]*nsites
    if not np.iterable(bonddim):
        bonddim = [bonddim]*(nsites-1)
    bonddim = [1] + bonddim + [1]
    tensors = [tsr.Tensor(np.random.rand(physdim[i], bonddim[i], bonddim[i+1]),
        [phys_label, left_label, right_label]) for i in range(nsites)]
    return od.MatrixProductState(tensors, left_label=left_label,
            right_label=right_label, phys_label=phys_label)


def onebody_sum_mpo(terms, output_label=None):
    """
    Construct an MPO from a sum of onebody operators, using the recipe from
    the Supplemental Material of [1]_ (Eqs. (3) and (4))

    Parameters
    ---------
    terms : list
        A list containing the terms in the sum. Each term should be 2D 
        array-like, e.g., a rank-two Tensor or numpy array.
    output_label : str, optional
        Specify the label corresponding to the output index. Must be the same
        for each element of `terms`. If not specified the first index is taken 
        to be the output index.

    Returns
    ------
    MatrixProductOperator

    References
    ----------
    .. [1] E. Sanchez-Burillo et al., Phys. Rev. Lett. 113, 263604 (2014)
    """
    tensors = []
    for i, term1 in enumerate(terms):
        if output_label is not None:
            term = term1.copy()
            term.move_index(output_label, 0)
        else:
            term = term1
        if i==0:
            B = np.zeros(shape=term.shape+[2], dtype=complex)
            for k in range(term.shape[0]):
                for l in range(term.shape[1]):
                    B[k,l,:] = [term[k, l], k==l]
            tensors.append(tsr.Tensor(B, ['physout', 'physin', 'right']))
        elif i==len(terms)-1:
            B = np.zeros(shape=term.shape+[2], dtype=complex)
            for k in range(term.shape[0]):
                for l in range(term.shape[1]):
                    B[k,l,:] = [k==l, term[k, l]]
            tensors.append(tsr.Tensor(B, ['physout', 'physin', 'left']))
        else:
            B = np.zeros(shape=term.shape+[2,2], dtype=complex)
            for k in range(term.shape[0]):
                for l in range(term.shape[1]):
                    B[k,l,:,:] = [[k==l, 0], [term[k, l], k==l]]
            tensors.append(tsr.Tensor(B, ['physout', 'physin', 
                'left', 'right']))
    return od.MatrixProductOperator(tensors, left_label='left',
        right_label='right', physin_label='physin', physout_label='physout')


def expvals_mps(mps, oplist, output_label=None, canonised=None):
    """
    Return single site expectation values <op>_i for all i

    Parameters
    ----------
    mps : MatrixProductState
    oplist : list or Tensor
        List of rank-two tensors representing the operators at each site.
        If a single `Tensor` is given this will be used for all sites.
    output_label : str, optional
        Specify the label corresponding to the output index. Must be the same
        for each element of `terms`. If not specified the first index is taken 
        to be the output index.
    canonised : {'left', 'right', None}, optional
        Flag to specify theat `mps` is already in left or right canonical form.

    Returns
    ------
    array
        Complex array of same length as `mps` of expectation values.

    Notes
    -----
    `mps` will be in left canonical form after the function call.
    """
    N = len(mps)
    expvals = np.zeros(N, dtype=complex)
    if not np.iterable(oplist):
        oplist_new = [oplist]*N
    else:
        oplist_new = oplist

    if canonised == 'left':
        mps.reverse()
        oplist_new = oplist_new[::-1]
    elif canonised != 'right':
        mps.right_canonise()

    for k, op in enumerate(oplist_new):
        # compute exp value for site k
        A = mps[k]
        if output_label is None:
            out_label = op.labels[0]
            in_label = op.labels[1]
        else:
            out_label = output_label
            in_label = [x for x in op.labels if x is not out_label][0]
        Ad = A.copy()
        Ad.conjugate()
        exp = tsr.contract(A, op, mps.phys_label, in_label)
        exp = tsr.contract(Ad, exp, mps.phys_label, out_label)
        exp.contract_internal(mps.left_label, mps.left_label, index1=0,
                index2=1)
        exp.contract_internal(mps.right_label, mps.right_label, index1=0,
                index2=1)
        expvals[k] = exp.data
        # move orthogonality center along MPS
        mps.left_canonise(k, k+1)

    if canonised == 'left':
        mps.reverse()
        oplist_new = oplist_new[::-1]
        expvals = expvals[::-1]

    return expvals


