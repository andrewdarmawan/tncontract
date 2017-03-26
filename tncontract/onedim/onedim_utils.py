"""
onedim_utils
==========

Module with various functions for MPS/MPOs.
"""

__all__ = ['init_mps_random', 'init_mps_allzero', 'init_mps_logical',
        'onebody_sum_mpo', 'expvals_mps', 'ptrace_mps']


import numpy as np


from tncontract import tensor as tnc
from tncontract.onedim import onedim_core as onedim
# from tncontract import onedim as onedim


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
    tensors = [tnc.Tensor(np.random.rand(physdim[i], bonddim[i], bonddim[i+1]),
        [phys_label, left_label, right_label]) for i in range(nsites)]
    return onedim.MatrixProductState(tensors, left_label=left_label,
            right_label=right_label, phys_label=phys_label)


def init_mps_allzero(nsites, physdim, left_label='left',
        right_label='right', phys_label='phys'):
    """
    Create an MPS with `nsites` sites in the "all zero" state |00..0>.

    Parameters
    ----------
    nsites : int
    physdim : int or list of ints
    left_label : str
    right_label : str
    phys_label : str
    """
    if not np.iterable(physdim):
        physdim = [physdim]*nsites

    tensors = []
    for j in range(nsites):
        t = np.zeros(physdim[j])
        t[0] = 1.0
        t = tnc.Tensor(t.reshape(physdim[j], 1, 1), [phys_label, left_label,
            right_label])
        tensors.append(t)

    return onedim.MatrixProductState(tensors, left_label=left_label,
        right_label=right_label, phys_label=phys_label)


def init_mps_logical(nsites, basis_state, physdim, left_label='left',
        right_label='right', phys_label='phys'):
    """
    Create an MPS with `nsites` sites in the logical basis state |ijk..l>.

    Parameters
    ----------
    nsites : int
    basis_state : int or list of ints
        Site `i` will be in the state |`basis_state[i]`> (or simply
        |`basis_state`> if a single int is provided).
    physdim : int or list of ints
    left_label : str
    right_label : str
    phys_label : str
    """
    if not np.iterable(physdim):
        physdim = [physdim]*nsites

    tensors = []
    for j in range(nsites):
        t = np.zeros(physdim[j])
        t[basis_state[j]] = 1.0
        t = tnc.Tensor(t.reshape(physdim[j], 1, 1), [phys_label, left_label,
            right_label])
        tensors.append(t)

    return onedim.MatrixProductState(tensors, left_label=left_label,
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
            B = np.zeros(shape=term.shape+(2,), dtype=complex)
            for k in range(term.shape[0]):
                for l in range(term.shape[1]):
                    B[k,l,:] = [term[k, l], k==l]
            tensors.append(tnc.Tensor(B, ['physout', 'physin', 'right']))
        elif i==len(terms)-1:
            B = np.zeros(shape=term.shape+(2,), dtype=complex)
            for k in range(term.shape[0]):
                for l in range(term.shape[1]):
                    B[k,l,:] = [k==l, term[k, l]]
            tensors.append(tnc.Tensor(B, ['physout', 'physin', 'left']))
        else:
            B = np.zeros(shape=term.shape+(2,2), dtype=complex)
            for k in range(term.shape[0]):
                for l in range(term.shape[1]):
                    B[k,l,:,:] = [[k==l, 0], [term[k, l], k==l]]
            tensors.append(tnc.Tensor(B, ['physout', 'physin', 
                'left', 'right']))
    return onedim.MatrixProductOperator(tensors, left_label='left',
        right_label='right', physin_label='physin', physout_label='physout')


def expvals_mps(mps, oplist=[], sites=None, output_label=None, canonised=None):
    # TODO: Why canonised gives strange results?
    """
    Return single site expectation values <op>_i for all i

    Parameters
    ----------
    mps : MatrixProductState
    oplist : list or Tensor
        List of rank-two tensors representing the operators at each site.
        If a single `Tensor` is given this will be used for all sites.
    sites : int or list of ints, optional
        Sites for which to compute expectation values. If None all
        sites will be returned.
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
    if sites is None:
        sites = range(len(mps))
    if not np.iterable(sites):
        sites = [sites]

    N = len(sites)
    expvals = np.zeros(N, dtype=complex)

    if not isinstance(oplist, list):
        oplist_new = [oplist]*N
    else:
        oplist_new = oplist

    if canonised == 'left':
        mps.reverse()
        oplist_new = oplist_new[::-1]
    elif canonised != 'right':
        mps.right_canonise()

    center = 0
    for i, site in enumerate(sites):
        # Mover orthogonality center to site k
        mps.left_canonise(center, site)
        center = site

        # compute exp value for site k
        op = oplist_new[i]
        A = mps[site]
        if output_label is None:
            out_label = op.labels[0]
            in_label = op.labels[1]
        else:
            out_label = output_label
            in_label = [x for x in op.labels if x is not out_label][0]
        Ad = A.copy()
        Ad.conjugate()
        exp = tnc.contract(A, op, mps.phys_label, in_label)
        exp = tnc.contract(Ad, exp, mps.phys_label, out_label)
        exp.contract_internal(mps.left_label, mps.left_label, index1=0,
                index2=1)
        exp.contract_internal(mps.right_label, mps.right_label, index1=0,
                index2=1)
        expvals[i] = exp.data

    if canonised == 'left':
        mps.reverse()
        oplist_new = oplist_new[::-1]
        expvals = expvals[::-1]

    return expvals

def ptrace_mps(mps, sites=None, canonised=None):
    # TODO: Why canonised gives strange results?
    """
    Return single site reduced density matrix rho_i for all i in sites.

    Parameters
    ----------
    mps : MatrixProductState
    sites : int or list of ints, optional
        Sites for which to compute the reduced density matrix. If None all
        sites will be returned.
    canonised : {'left', 'right', None}, optional
        Flag to specify theat `mps` is already in left or right canonical form.

    Returns
    ------
    list
        List of same length as `mps` with rank-two tensors representing the
        reduced density matrices.

    Notes
    -----
    `mps` will be in left canonical form after the function call.
    """
    rho_list = []

    if canonised == 'left':
        mps.reverse()
    elif canonised != 'right':
        mps.right_canonise()

    if sites is None:
        sites = range(len(mps))
    if not np.iterable(sites):
        sites = [sites]

    center = 0
    for site in sites:
        # Mover orthogonality center to site k
        mps.left_canonise(center, site)
        center = site

        A = mps[center]
        Ad = A.copy()
        Ad.conjugate()
        Ad.prime_label(mps.phys_label)

        rho = tnc.contract(A, Ad, [mps.left_label, mps.right_label],
                [mps.left_label, mps.right_label])
        rho_list.append(rho)

    if canonised == 'left':
        mps.reverse()
        rho_list = rho_list[::-1]

    return rho_list


