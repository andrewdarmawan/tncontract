import numpy as np

from .. import tensor as tsr

class OneDimensionalTensorNetwork():
    """A one dimensional tensor network specified by a 1D array of tensors (a 
    list or 1D numpy array) where each tensor has a left and a right index.
    Need to specify which labels correspond to these using arguments 
    left_label, right_label."""
    def __init__(self, tensors, left_label, right_label):
        self.left_label=left_label
        self.right_label=right_label
        #Copy input tensors to the data attribute
        self.data=np.array([x.copy() for x in tensors])
        #Every tensor will have three indices corresponding to "left", "right"
        #and "phys" labels. If only two are specified for left and right 
        #boundary tensors (for open boundary conditions) an extra dummy index 
        #of dimension 1 will be added. 
        for x in self.data:
            if left_label not in x.labels: x.add_dummy_index(left_label)
            if right_label not in x.labels: x.add_dummy_index(right_label)

   #Add container emulation
    def __iter__(self):
        return self.data.__iter__()
    def __len__(self):
        return self.data.__len__()
    def __getitem__(self, key):
        return self.data.__getitem__(key)
    def __setitem__(self, key, value):
        self.data.__setitem__(key, value)

    def copy(self):
        """Alternative the standard copy method, returning a
        OneDimensionalTensorNetwork that is not
        linked in memory to the previous ones."""
        return OneDimensionalTensorNetwork([x.copy() for x in self], 
                self.left_label, self.right_label)

    def reverse(self):
        self.data=self.data[::-1]
        temp=self.left_label
        self.left_label=self.right_label
        self.right_label=temp

    def swap_gate(self, i):
        """
        Apply a swap gate swapping all "physical" (i.e., non-"left" and
        non-"right") indices for site i and i+1 of a
        OneDimensionalTensorNetwork.

        Notes
        -----
        The swap is implemented as described
        in Y.-Y. Shi et al, Phys. Rev. A 74, 022320 (2006).
        """
        A = self[i]
        B = self[i+1]
        A_phys_labels = [l for l in A.labels if l!=self.left_label and
                l!=self.right_label]
        B_phys_labels = [l for l in B.labels if l!=self.left_label and
                l!=self.right_label]
        A.prime_label(A_phys_labels)
        t = tsr.contract(A, B, self.right_label, self.left_label)
        U, S, V = tsr.tensor_svd(t, [self.left_label] + B_phys_labels)
        U.replace_label('svd_in', 'right')
        self[i] = U
        V.unprime_label(A_phys_labels)
        SV = tsr.contract(S, V, ['svd_in'], ['svd_out'])
        SV.replace_label('svd_out', 'left')
        self[i+1] = SV

    def leftdim(self, site):
        """Return left index dimesion for site"""
        return self.data[site].index_dimension(self.left_label)

    def rightdim(self, site):
        """Return right index dimesion for site"""
        return self.data[site].index_dimension(self.right_label)

    def bonddims(self):
        """Return list of all bond dimensions"""
        if self.nsites == 0:
            return []
        bonds = [self.leftdim(0)]
        for i in range(self.nsites):
            bonds.append(self.rightdim(i))
        return bonds

    @property
    def nsites(self):
        return len(self.data)


class MatrixProductState(OneDimensionalTensorNetwork):
    """Matrix product state"is a list of tensors, each having and index 
    labelled "phys" and at least one of the indices "left", "right"
    Input is a list of tensors, with three up to three index labels, If the 
    labels aren't already specified as "left", "right", "phys" need to specify
    which labels correspond to these using arguments left_label, right_label, 
    phys_label. The tensors input will be copied, and will not point in memory
    to the original ones."""

    def __init__(self, tensors, left_label, right_label, phys_label):
        OneDimensionalTensorNetwork.__init__(self, tensors, left_label, 
                right_label)
        self.phys_label=phys_label

    def __repr__(self):
        return ("MatrixProductState(tensors=%r, left_label=%r, right_label=%r,"
            "phys_label=%r)" % (self.data, self.left_label, self.right_label, 
                    self.phys_label))

    def __str__(self):
        return ("MatrixProductState object: " +
              "sites = " + str(len(self)) + 
              ", left_label = " + self.left_label + 
              ", right_label = " + self.right_label + 
              ", phys_label = " + self.phys_label)

    def copy(self):
        """Return an MPS that is not linked in memory to the original."""
        return MatrixProductState([x.copy() for x in self], self.left_label, 
                self.right_label, self.phys_label)

    def left_canonise(self, start=0, end=-1, chi=0, threshold=10**-14, 
            normalise=False):
        """
        Perform left canonisation of MPS. 
        
        Left canonisation refers to putting the MatrixProductState in a form
        where the tensors are isometric maps from the left and physical 
        indices to the right index. This is achieved using successive
        singular-value decompositions and exploiting the gauge freedom of the
        MPS. For more details, see U. Schollwock, Ann. Phys. 326 (2011) 96-192.
        If no arguments are supplied, every tensor will be put in this form, 
        i.e. the MPS will be put in left-canonical form. Canonisation of 
        a segment of also possible by specifying the `start` and `end` 
        parameters. Truncating singular values can be performed by specifying 
        `chi` and `threshold`. If `normalise`=True and the entire MPS is to be
        left-canonised, the resulting MPS will represent a normalised state. If
        only a segment of the MPS is to be left canonised, then `normalise`
        will have no effect (the resulting state will have same norm as input).

        Parameters
        ----------
        start : int
        end : int
            The segment of the MPS to be left canonised. All tensors from 
            `start` to `end`-1 will be left canonised. `end`=-1 implies that
            the MPS will be canonised to the right boundary.
        chi : int
            Maximum number of singular values of each tensor to keep after
            performing singular-value decomposition.
        threshold : float
            Lower bound on the magnitude of singular values to keep. Singular
            values less than or equal to this value will be truncated.
        normalise : bool
            False value indicates resulting state will have same norm as
            original. True value indicates that, if the entire MPS is to be
            left canonised, it will be divided by a factor such that it is
            normalised (have norm=1). Has no effect if only a segment of the 
            MPS is to be left canonised (resulting state will have the same
            norm as input).
        """
        N=len(self)
        if end==-1:
            end=N

        #At each step will divide by a constant so that the largest singular 
        #value of S is 1. Will store the product of these constants in `norm`
        norm=1
        for i in range(start,end):
            if i==N-1:
                #The final SVD has no right index, so S and V are just scalars.
                #S is the norm of the state. 
                if normalise==True and start==0: #Whole chain is canonised
                    self[i].data=self[i].data/np.linalg.norm(self[i].data)
                else:
                    self[i].data=self[i].data*norm
                return
            else:
                U,S,V = tsr.tensor_svd(self[i], [self.phys_label, 
                    self.left_label])

            #Truncate to threshold and to specified chi
            singular_values=np.diag(S.data)
            largest_singular_value=singular_values[0]
            #Normalise S
            singular_values=singular_values/largest_singular_value
            norm*=largest_singular_value

            singular_values_to_keep = singular_values[singular_values > 
                    threshold]
            if chi:
                singular_values_to_keep = singular_values_to_keep[:chi]
            S.data=np.diag(singular_values_to_keep)
            #Truncate corresponding singular index of U and V
            U.data=U.data[:,:,0:len(singular_values_to_keep)]
            V.data=V.data[0:len(singular_values_to_keep)]

            U.replace_label("svd_in", self.right_label)
            self[i]=U
            self[i+1]=tsr.contract(V, self[i+1], self.right_label, 
                    self.left_label)
            self[i+1]=tsr.contract(S, self[i+1], ["svd_in"], ["svd_out"])
            self[i+1].replace_label("svd_out", self.left_label)

            #Reabsorb normalisation factors into next tensor
            #Note if i==N-1 (end of chain), this will not be reached 
            #and normalisation factors will be taken care of in the earlier 
            #block.
            if i==end-1:
                self[i+1].data*=norm

    def right_canonise(self, start=0, end=-1, chi=0, threshold=10**-14, 
            normalise=False):
        """Perform right canonisation of MPS. Identical to `left_canonise`
        except that process is mirrored (i.e. canonisation is performed from
        right to left). `start` and `end` specify the interval to be canonised.

        Notes
        -----
        The first tensor to be canonised is `end`-1 and the
        final tensor to be canonised is `start`""" 

        self.reverse()
        N=len(self)
        if end==-1:
            end=N
        self.left_canonise(start=N-end, end=N-start, chi=chi,
                threshold=threshold, normalise=normalise)
        self.reverse()

    def replace_left_right_phys_labels(self, new_left_label=None, 
            new_right_label=None, new_phys_label=None):
        """
        Replace left, right, phys labels in every tensor, and update the 
        MatrixProductState attributes left_label, right_label and phys_label. 
        If new label is None, the label will not be replaced. 
        """
        old_labels=[self.left_label, self.right_label, self.phys_label]
        new_labels=[new_left_label, new_right_label, new_phys_label]
        #Replace None entries with old label (i.e. don't replace)
        for i, x in enumerate(new_labels):
            if x==None:
                new_labels[i]=old_labels[i]
        for tensor in self.data:
            tensor.replace_label(old_labels, new_labels)

        self.left_label=new_left_label
        self.right_label=new_right_label
        self.phys_label=new_phys_label

    def standard_labels(self, suffix=""):
        """
        Overwrite self.labels, self.left_label, self.right_label, 
        self.phys_label with standard labels "left", "right", "phys"
        """
        self.replace_left_right_phys_labels(new_left_label="left"+suffix, 
                new_right_label="right"+suffix, new_phys_label="phys"+suffix)

    def check_canonical_form(self, threshold=10**-14, print_output=True):
        """Determines which tensors in the MPS are left canonised, and which 
        are right canonised. Returns the index of the first tensor (starting 
        from left) that is not left canonised, and the first tensor (starting 
        from right) that is not right canonised. If print_output=True, will 
        print useful information concerning whether a given MPS is in a 
        canonical form (left, right, mixed).""" 
        mps_cc=mps_complex_conjugate(self)
        first_site_not_left_canonised=len(self)-1
        for i in range(len(self)-1): 
            I=tsr.contract(self[i], mps_cc[i], 
                    [self.phys_label, self.left_label], 
                    [mps_cc.phys_label, mps_cc.left_label])
            #Check if tensor is left canonised.
            if np.linalg.norm(I.data-np.identity(I.data.shape[0])) > threshold:
                first_site_not_left_canonised=i
                break
        first_site_not_right_canonised=0
        for i in range(len(self)-1,0, -1): 
            I=tsr.contract(self[i], mps_cc[i], 
                    [self.phys_label, self.right_label], 
                    [mps_cc.phys_label, mps_cc.right_label])
            #Check if tensor is right canonised.
            right_canonised_sites=[]
            if np.linalg.norm(I.data-np.identity(I.data.shape[0])) > threshold:
                first_site_not_right_canonised=i
                break
        if print_output:
            if first_site_not_left_canonised==first_site_not_right_canonised:
                if first_site_not_left_canonised==len(self)-1:
                    if abs(np.linalg.norm(self[-1].data)-1) > threshold:
                        print("MPS in left canonical form (unnormalised)")
                    else:
                        print("MPS in left canonical form (normalised)")
                elif first_site_not_left_canonised==0:
                    if abs(np.linalg.norm(self[0].data)-1) > threshold:
                        print("MPS in right canonical form (unnormalised)")
                    else:
                        print("MPS in right canonical form (normalised)")
                else:
                    print("MPS in mixed canonical form with orthogonality "
                            "centre at site "+
                            str(first_site_not_right_canonised))
            else:
                if first_site_not_left_canonised==0:
                    print("No tensors left canonised")
                else:
                    print("Tensors left canonised up to site "+
                            str(first_site_not_left_canonised))
                if first_site_not_right_canonised==len(self)-1:
                    print("No tensors right canonised")
                else:
                    print("Tensors right canonised up to site "+
                            str(first_site_not_right_canonised))
        return (first_site_not_left_canonised, first_site_not_right_canonised)

    def svd_compress(self, chi, threshold=10**-15, normalise=False):
        """Simply right canonise the left canonical form according to 
        Schollwock"""
        self.left_canonise(threshold=threshold, normalise=normalise)
        self.right_canonise(chi=chi, threshold=threshold, normalise=normalise)

    def physdim(self, site):
        """Return physical index dimesion for site"""
        return self.data[site].index_dimension(self.phys_label)

    def apply_gate(self, gate, firstsite, gate_outputs=None, gate_inputs=None,
            chi=0, threshold=1e-15):
        """
        Apply Tensor `gate` on sites `firstsite`, `firstsite`+1, ...,
        `firstsite`+`nsites`-1, where `nsites` is the length of gate_inputs.
        The physical index of the nth site is contracted with the nth label of 
        `gate_inputs`. After the contraction the MPS is put back into the 
        original form by SVD, and the nth sites physical index is given by the 
        nth label of `gate_outputs` (but relabeled to `self.phys_label` to
        preserve the original MPS form).

        Parameters
        ----------
        gate : Tensor
            Tensor representing the multisite gate.
        firstsite : int
            First site of MPS involved in the gate
        gate_outputs : list of str, optional
            Output labels corresponding to the input labels given by
            `gate_inputs`. Must have the same length as `gate_inputs`.
            If `None` the first half of `gate.labels` will be taken as output
            labels.
        gate_inputs : list of str, optional
            Input labels. The first index of the list is contracted with
            `firstsite`, the second with `firstsite`+1 etc.
            If `None` the second half of `gate.labels` will be taken as input
            labels.
        chi : int, optional
            Maximum number of singular values of each tensor to keep after
            performing singular-value decomposition.
        threshold : float
            Lower bound on the magnitude of singular values to keep. Singular
            values less than or equal to this value will be truncated.

        Notes
        -----
        At the end of the gate all physical indices are relabeled to
        `self.phys_label`.

        Only use this for gates acting on small number of sites.
        """
        # Set gate_outputs and gate_inputs to default values if not given
        if gate_outputs is None and gate_inputs is None:
            gate_outputs = gate.labels[:int(len(gate.labels)/2)]
            gate_inputs = gate.labels[int(len(gate.labels)/2):]
        elif gate_outputs is None:
            gate_outputs =[x for x in gate.labels if x not in gate_inputs]
        elif gate_inputs is None:
            gate_inputs =[x for x in gate.labels if x not in gate_outputs]

        nsites = len(gate_inputs)
        if len(gate_outputs) != nsites:
            raise ValueError("len(gate_outputs) != len(gate_inputs)")

        # contract the sites first
        t = contract_virtual_indices(self, firstsite, firstsite+nsites,
                periodic_boundaries=False)

        # contract all physical indices with gate input indices
        t = tsr.contract(t, gate, self.phys_label, gate_inputs)

        # split big tensor into MPS form by exact SVD
        mps = tensor_to_mps(t, mps_phys_label=self.phys_label,
                left_label=self.left_label, right_label=self.right_label,
                chi=chi, threshold=threshold)
        self.data[firstsite:firstsite+nsites] = mps.data


class MatrixProductOperator(OneDimensionalTensorNetwork):
    #TODO currently assumes open boundaries
    """Matrix product operator "is a list of tensors, each having and index 
    labelled "phys" and at least one of the indices "left", "right"
    Input is a list of tensors, with three up to three index labels, If the 
    labels aren't already specified as "left", "right", "physin", "physout" 
    need to specify which labels correspond to these using 
    arguments left_label, right_label, physin_label and physout_label. """
    def __init__(self, tensors, left_label, right_label, physout_label, 
            physin_label):
        OneDimensionalTensorNetwork.__init__(self, tensors, left_label, 
                right_label)
        self.physout_label=physout_label
        self.physin_label=physin_label

    def __repr__(self):
        return ("MatrixProductOperator(tensors=%r, left_label=%r,"
                " right_label=%r, physout_label=%r, phsin_labe=%r)" 
                % (self.data, self.left_label, self.right_label,
                    self.physout_label, self.physin_label))

    def __str__(self):
        return ("MatrixProductOperator object: " +
              "sites = " + str(len(self)) +
              ", left_label = " + self.left_label +
              ", right_label = " + self.right_label +
              ", physout_label = " + self.physout_label +
              ", physin_label = " + self.physin_label)

    ###TODO replace copy method

    def physoutdim(self, site):
        """Return output physical index dimesion for site"""
        return self.data[site].index_dimension(self.physout_label)

    def physindim(self, site):
        """Return input physical index dimesion for site"""
        return self.data[site].index_dimension(self.physin_label)


def tensor_to_mps(tensor, phys_labels=None, mps_phys_label='phys',
        left_label='left', right_label='right', chi=0, threshold=1e-15):
    """
    Split a tensor into MPS form by exact SVD

    Parameters
    ----------
    tensor : Tensor
    phys_labels list of str, optional
        Can be used to specify the order of the physical indices for the MPS.
    mps_phys_label : str
        Physical labels of the resulting MPS will be renamed to this value.
    left_label : str
        Label for index of `tensor` that will be regarded as the leftmost index
        of the resulting MPS if it exists (must be unique).
        Also used as `left_label` for the resulting MPS.
    right_label : str
        Label for index of `tensor` that will be regarded as the rightmost
        index of the resulting MPS if it exists (must be unique).
        Also used as `right_label` for the resulting MPS.
    chi : int, optional
        Maximum number of singular values of each tensor to keep after
        performing singular-value decomposition.
    threshold : float
        Lower bound on the magnitude of singular values to keep. Singular
        values less than or equal to this value will be truncated.
    """
    if phys_labels is None:
        phys_labels =[x for x in tensor.labels if x not in
                [left_label, right_label]]

    nsites = len(phys_labels)
    V = tensor.copy()
    mps = []
    for k in range(nsites-1):
        U, V, _ = tsr.truncated_svd(V, [left_label]*(left_label in V.labels)
                +[phys_labels[k]], chi=chi, threshold=threshold)
        U.replace_label('svd_in', right_label)
        U.replace_label(phys_labels[k], mps_phys_label)
        mps.append(U)
        #t = tsr.contract(S, V, ['svd_in'], ['svd_out'])
        V.replace_label('svd_out', left_label)
    V.replace_label(phys_labels[nsites-1], mps_phys_label)
    mps.append(V)
    return MatrixProductState(mps, phys_label=mps_phys_label,
            left_label=left_label, right_label=right_label)


def tensor_to_mpo(tensor, physout_labels=None, physin_labels=None,
        mpo_physout_label='physout', mpo_physin_label='physin',
        left_label='left', right_label='right', chi=0, threshold=1e-15):
    """
    Split a tensor into MPO form by exact SVD

    Parameters
    ----------
    tensor : Tensor
    physout_labels : list of str, optional
        The output physical indices for the MPO. First site of MPO has output
        index corresponding to physout_labels[0] etc.
        If `None` the first half of `tensor.labels` will be taken as output
        labels.
    physin_labels : list of str, optional
        The input physical indices for the MPO. First site of MPO has input
        index corresponding to physin_labels[0] etc.
        If `None` the second half of `tensor.labels` will be taken as input
        labels.
    mpo_phys_label : str
        Physical input labels of the resulting MPO will be renamed to this.
    mpo_phys_label : str
        Physical output labels of the resulting MPO will be renamed to this.
    left_label : str
        Label for index of `tensor` that will be regarded as the leftmost index
        of the resulting MPO if it exists (must be unique).
        Also used as `left_label` for the resulting MPO.
    right_label : str
        Label for index of `tensor` that will be regarded as the rightmost
        index of the resulting MPO if it exists (must be unique).
        Also used as `right_label` for the resulting MPO.
    chi : int, optional
        Maximum number of singular values of each tensor to keep after
        performing singular-value decomposition.
    threshold : float
        Lower bound on the magnitude of singular values to keep. Singular
        values less than or equal to this value will be truncated.
    """
    # Set physout_labels and physin_labels to default values if not given
    phys_labels =[x for x in tensor.labels if x not in
            [left_label, right_label]]
    if physout_labels is None and physin_labels is None:
        physout_labels = phys_labels[:int(len(phys_labels)/2)]
        physin_labels = phys_labels[int(len(phys_labels)/2):]
    elif physout_labels is None:
        physout_labels =[x for x in phys_labels if x not in physin_labels]
    elif physin_labels is None:
        physin_labels =[x for x in phys_labels if x not in physout_labels]

    nsites = len(physin_labels)
    if len(physout_labels) != nsites:
        raise ValueError("len(physout_labels) != len(physin_labels)")

    V = tensor.copy()
    mpo = []
    for k in range(nsites-1):
        U, V, _ = tsr.truncated_svd(V, [left_label]*(left_label in V.labels)
                +[physout_labels[k], physin_labels[k]],
                chi=chi, threshold=threshold)
        U.replace_label('svd_in', right_label)
        U.replace_label(physout_labels[k], mpo_physout_label)
        U.replace_label(physin_labels[k], mpo_physin_label)
        mpo.append(U)
        V.replace_label('svd_out', left_label)
    V.replace_label(physout_labels[nsites-1], mpo_physout_label)
    V.replace_label(physin_labels[nsites-1], mpo_physin_label)
    mpo.append(V)
    return MatrixProductOperator(mpo, physout_label=mpo_physout_label,
            physin_label=mpo_physin_label, left_label=left_label,
            right_label=right_label)

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
    return MatrixProductState(tensors, left_label=left_label,
            right_label=right_label, phys_label=phys_label)


def contract_multi_index_tensor_with_one_dim_array(tensor, array, label1, 
        label2):
    """Will contract a one dimensional tensor array of length N 
    with a single tensor with N indices with label1.
    All virtual indices are also contracted. 
    Each tensor in array is assumed to have an index with label2. 
    Starting from the left, the label2 index of each tensor 
    is contracted with the first uncontracted label1 index of tensor
    until every tensor in array is incorporated.
    It is assumed that only the indices to be contracted have the labels label1 
    label2."""

    #To avoid possible label conflicts, rename labels temporarily 
    temp_label=0 
    tensor.replace_label(label1, temp_label)

    C=tsr.contract(tensor, array[0], temp_label, label2, index_list1=[0])
    for i in range(1, len(array)):
        #TODO make this work
        C=tsr.contract(C, array[i], [array.right_label, temp_label], 
                [array.left_label, label2], index_list1=[0,1])

    #Contract boundaries of array
    C.contract_internal(array.right_label, array.left_label)
    #Restore original labelling to tensor
    tensor.replace_label(temp_label, label1)
    return C

def contract_virtual_indices(array_1d, start=0, end=None, 
        periodic_boundaries=True):
    """
    Return a Tensor by contracting all virtual indices of a segment of a
    OneDimensionalTensorNetwork.

    Params
    -----
    array_1d : OneDimensionalTensorNetwork
    start : int
        First site of segment to be contracted
    end : int
        Last site of segment to be contracted
    periodic_boundaries : bool
        If `True` leftmost and rightmost virtual indices are contracted.
    """
    C=array_1d[start]
    for x in array_1d[start+1:end]:
        C=tsr.contract(C, x, array_1d.right_label, array_1d.left_label)
    if periodic_boundaries:
        # Contract left and right boundary indices (periodic boundaries)
        # Note that this will simply remove boundary indices of dimension one.
        C.contract_internal(array_1d.right_label, array_1d.left_label) 
    return C

def left_canonical_form_mps(orig_mps, chi=0, threshold=10**-14, 
        normalise=False):
    """Computes left canonical form of an MPS using algorithm in 
    Schollwock 2011, 
    by taking successive SVDs.
    Compression is usually not done at this stage, but it can by specifying 
    the optional chi argument.
    Setting chi=0 means no compression."""
    """Possible to speed up using a QR rather than SVD decomposition"""
    mps=orig_mps.copy()
    mps.left_canonise(chi=chi, threshold=threshold, normalise=normalise)
    return mps

def reverse_mps(mps):
    return MatrixProductState([x.copy() for x in reversed(mps)], 
            mps.right_label, mps.left_label, mps.phys_label)

def right_canonical_form_mps(orig_mps, chi=0, threshold=10**-14, 
        normalise=False):
    """Identical to left canonical form but starting from right"""
    #TODO replace with call to right canonise method
    #mps=[x.copy() for x in orig_mps] #Dont want to modify the original mps
    mps=reverse_mps(orig_mps)
    mps=left_canonical_form_mps(mps, chi=chi, threshold=threshold, 
            normalise=normalise)
    return reverse_mps(mps)

def check_canonical_form_mps(mps, threshold=10**-14, print_output=True):
    mps.check_canonical_form(threshold=threshold,
            print_output=print_output)
    
def svd_compress_mps(orig_mps, chi, threshold=10**-15, normalise=False):
    """Simply right canonise the left canonical form according to Schollwock"""
    mps=left_canonical_form_mps(orig_mps, threshold=threshold, 
            normalise=normalise)
    return right_canonical_form_mps(mps, chi=chi, threshold=threshold, 
            normalise=normalise)

def variational_compress_mps(mps, chi, max_iter=20):
    #TODO I think I have implemented this very inefficiently. Fix.
    """Take an mps represented as a list of tensors with labels: left, right, 
    phys and return an mps in the same form with bond dimension chi. 
    (using dmrg)"""
    #TODO check check frob norm difference to previous iteration and stop 
    #when the update becomes too small. First make a guess at optimal form 
    #using svd_truncate_mps, as above.
    n=len(mps)

    #Start with a good guess for the new mps, using the canonical form
    new_mps=svd_compress_mps(mps, chi, threshold=10**-15, normalise=False)

    #Loop along the chain from left to right, optimizing each tensor 
    for k in range(max_iter):
        for i in range(n):
            #Define complex conjugate of mps
            new_mps_cc=[x.copy() for x in new_mps]
            for x in new_mps_cc:
                x.conjugate()
                x.replace_label("left", "left_cc")
                x.replace_label("right", "right_cc")

            #Computing A (quadratic component)

            #Right contraction
            if i!=n-1:
                right_boundary=tsr.contract(new_mps[-1], new_mps_cc[-1], 
                        ["phys"], ["phys"] )
                for j in range(n-2, i, -1):
                    right_boundary=tsr.contract(right_boundary, new_mps[j], 
                            ["left"], ["right"])
                    right_boundary=tsr.contract(right_boundary, new_mps_cc[j],
                            ["left_cc", "phys"], ["right_cc", "phys"])

            #Left contraction
            if i!=0:
                left_boundary=tsr.contract(new_mps[0], new_mps_cc[0], 
                        ["phys"], ["phys"] )
                for j in range(1,i):
                    left_boundary=tsr.contract(left_boundary, new_mps[j], 
                            ["right"], ["left"])
                    left_boundary=tsr.contract(left_boundary, new_mps_cc[j], 
                            ["right_cc", "phys"], ["left_cc", "phys"])

            #Combine left and right components to form A

            #Dimension of the physical index
            phys_index=new_mps[i].labels.index("phys")
            phys_index_dim=new_mps[i].data.shape[phys_index]
            I_phys=tsr.Tensor(data=np.identity(phys_index_dim), 
                    labels=["phys_cc", "phys"])
            #Define the A matrix
            if i==0:
                A=tsr.tensor_product(right_boundary, I_phys)
            elif i==n-1:
                A=tsr.tensor_product(left_boundary, I_phys)
            else:
                A=tsr.tensor_product(left_boundary, right_boundary)
                A=tsr.tensor_product(A, I_phys)

            #Compute linear component "b"
            #Right contraction
            if i!=n-1:
                right_boundary=tsr.contract(mps[-1], new_mps_cc[-1], 
                        ["phys"], ["phys"] )
                for j in range(n-2, i, -1):
                    right_boundary=tsr.contract(right_boundary, mps[j], 
                            ["left"], ["right"])
                    right_boundary=tsr.contract(right_boundary, new_mps_cc[j], 
                            ["left_cc", "phys"], ["right_cc", "phys"])
            #Left contraction
            if i!=0:
                left_boundary=tsr.contract(mps[0], new_mps_cc[0], 
                        ["phys"], ["phys"] )
                for j in range(1,i):
                    left_boundary=tsr.contract(left_boundary, mps[j], 
                            ["right"], ["left"])
                    left_boundary=tsr.contract(left_boundary, new_mps_cc[j], 
                            ["right_cc", "phys"], ["left_cc", "phys"])

            #Connect left and right boundaries to form b
            if i==0:
                b=tsr.contract(mps[0], right_boundary, ["right"], ["left"])
            elif i==n-1:
                b=tsr.contract(left_boundary, mps[-1], ["right"], ["left"])
            else:
                b=tsr.contract(left_boundary, mps[i], ["right"], ["left"])
                b=tsr.contract(b, right_boundary, ["right"], ["left"])

            #Put indices in correct order, convert to matrices, and 
            #solve linear equation
            if i==0:
                #row indices
                A.move_index("left_cc", 0)
                A.move_index("phys_cc", 1)
                #column indices
                A.move_index("left", 2)
                A.move_index("phys", 3)
                old_shape=A.data.shape
                A_matrix=np.reshape(A.data, (np.prod(old_shape[0:2]),
                    np.prod(old_shape[2:4])))

                b.move_index("left_cc", 0)
                b_vector=b.data.flatten()
                minimum=np.linalg.solve(A_matrix, b_vector)
                updated_tensor=tsr.Tensor(data=np.reshape(minimum, 
                    (old_shape[0], old_shape[1])), labels=["right", "phys"])
            elif i==n-1:
                A.move_index("right_cc", 0)
                A.move_index("phys_cc", 1)
                A.move_index("right", 2)
                A.move_index("phys", 3)
                old_shape=A.data.shape
                A_matrix=np.reshape(A.data, (np.prod(old_shape[0:2]),
                    np.prod(old_shape[2:4])))

                b.move_index("right_cc", 0)
                b_vector=b.data.flatten()
                minimum=np.linalg.solve(A_matrix, b_vector)

                updated_tensor=tsr.Tensor(data=np.reshape(minimum, 
                    (old_shape[0], old_shape[1])), labels=["left", "phys"])
            else:
                A.move_index("right_cc", 0)
                A.move_index("left_cc", 1)
                A.move_index("phys_cc", 2)
                A.move_index("right", 3)
                A.move_index("left", 4)
                A.move_index("phys", 5)
                old_shape=A.data.shape
                A_matrix=np.reshape(A.data, 
                        (np.prod(old_shape[0:3]),np.prod(old_shape[3:6])))

                b.move_index("right_cc", 0)
                b.move_index("left_cc", 1)
                b_vector=b.data.flatten()
                minimum=np.linalg.solve(A_matrix, b_vector)

                updated_tensor=tsr.Tensor(data=np.reshape(minimum, 
                    (old_shape[0], old_shape[1], old_shape[2])), 
                        labels=["left", "right", "phys"])
                updated_tensor.consolidate_indices()
            new_mps[i]=updated_tensor

    return new_mps


def mps_complex_conjugate(mps):
    """Will take complex conjugate of every entry of every tensor in mps, 
    and append label_suffix to every label"""
    new_mps=mps.copy()
    for x in new_mps.data: 
        x.conjugate()
    return new_mps

def inner_product_one_dimension(array1, array2, label1, label2):
    """"""
    pass

def inner_product_mps(mps_bra, mps_ket, complex_conjugate_bra=True, 
        return_whole_tensor=False):
    """Inner product of two mps.
    They must have the same physical index dimensions"""
    if complex_conjugate_bra:
        mps_bra_cc=mps_complex_conjugate(mps_bra)
    else:
        #Just copy without taking complex conjugate
        mps_bra_cc=mps_bra.copy()

    #Temporarily relabel so no conflicts 
    mps_ket_old_labels=[mps_ket.left_label, mps_ket.right_label, 
            mps_ket.phys_label]
    mps_ket.standard_labels()
    #Suffix to distinguish from mps_ket labels
    mps_bra_cc.standard_labels(suffix="_cc") 

    left_boundary=tsr.contract(mps_bra_cc[0], mps_ket[0], 
            mps_bra_cc.phys_label, mps_ket.phys_label)
    for i in range(1,len(mps_ket)):
        left_boundary=tsr.contract(left_boundary, mps_bra_cc[i], 
                mps_bra_cc.right_label, mps_bra_cc.left_label)
        left_boundary=tsr.contract(left_boundary, mps_ket[i], 
                [mps_ket.right_label, mps_bra_cc.phys_label], 
                [mps_ket.left_label, mps_ket.phys_label])

    #Restore labels of mps_ket
    mps_ket.replace_left_right_phys_labels(new_left_label=mps_ket_old_labels[0]
            , new_right_label=mps_ket_old_labels[1],
            new_phys_label=mps_ket_old_labels[2])

    left_boundary.remove_all_dummy_indices()
    if return_whole_tensor:
        return left_boundary
    else:
        return left_boundary.data

def frob_distance_squared(mps1, mps2):
    ip=inner_product_mps
    return ip(mps1, mps1) + ip(mps2, mps2) - 2*np.real(ip(mps1, mps2))

def contract_mps_mpo(mps, mpo):
    """Will contract the physical index of mps with the physin index of mpo.
    Left and right indices will be combined. The resulting MPS will have the 
    same left and right labels as mps and the physical label will be 
    mpo.physout_label"""
    N=len(mps)
    new_mps=[]
    for i in range(N):
        new_tensor=tsr.contract(mps[i], mpo[i], mps.phys_label, 
                mpo.physin_label)
        new_tensor.consolidate_indices()
        new_mps.append(new_tensor)
    new_mps=MatrixProductState(new_mps, mps.left_label, mps.right_label, 
            mpo.physout_label)
    return new_mps


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
    return MatrixProductOperator(tensors, left_label='left',
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


