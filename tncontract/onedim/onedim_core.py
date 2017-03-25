"""
onedim_core
==========

Core module for onedimensional tensor networks
"""

__all__ = ['MatrixProductState', 'MatrixProductStateCanonical',
        'MatrixProductOperator', 'OneDimensionalTensorNetwork',
        'check_canonical_form_mps',
        'contract_mps_mpo', 'contract_multi_index_tensor_with_one_dim_array',
        'contract_virtual_indices', 'frob_distance_squared',
        'inner_product_mps', 'ladder_contract', 'left_canonical_form_mps',
        'mps_complex_conjugate', 'reverse_mps', 'right_canonical_form_mps',
        'svd_compress_mps', 'variational_compress_mps', 'tensor_to_mpo',
        'tensor_to_mps',
        'right_canonical_to_canonical',
        ]

import numpy as np


from tncontract import tensor as tsr
from tncontract.label import unique_label


class OneDimensionalTensorNetwork():
    """
    A one-dimensional tensor network. MatrixProductState and
    MatrixProductOperator are subclasses of this class. 

    An instance of `OneDimensionalTensorNetwork` contains a one-dimensional
    array of tensors in its `data` attribute. This one dimensional array is
    specified in the `tensors` argument when initialising the array. Each
    tensor in `data` requires a left index and a right index. The right index
    is taken to be contracted with the left index of the next tensor in the
    array, while the left index is taken to be contracted with the right index
    of the previous tensor in the array. All left indices are assumed to have
    the same label, and likewise for the right indices. They are specified in
    the initialisation of array (by default they are assumed to be "left" and
    "right" respectively) and will be stored in the attributes `left_label` and
    `right_label` of the OneDimensionalTensorNetwork instance. 
    
    """
    def __init__(self, tensors, left_label="left", right_label="right"):
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

   #Container emulation
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

    def complex_conjugate(self):
        """Will complex conjugate every entry of every tensor in array."""
        for x in self.data: 
            x.conjugate()

    def swap_gate(self, i, threshold=1e-15):
        """
        Apply a swap gate swapping all "physical" (i.e., non-"left" and
        non-"right") indices for site i and i+1 of a
        OneDimensionalTensorNetwork.

        Parameters
        ----------
        i : int
        threshold : float
            Lower bound on the magnitude of singular values to keep. Singular
            values less than or equal to this value will be truncated.

        Notes
        -----
        The swap is implemented by SVD as described
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
        U, V, _ = tsr.truncated_svd(t, [self.left_label] + B_phys_labels,
                chi=0, threshold=threshold, absorb_singular_values='both')
        U.replace_label('svd_in', self.right_label)
        self[i] = U
        V.unprime_label(A_phys_labels)
        V.replace_label('svd_out', self.left_label)
        self[i+1] = V

    def replace_labels(self, old_labels, new_labels):
        """Run `Tensor.replace_label` method on every tensor in `self` then
        replace `self.left_label` and `self.right_label` appropriately."""

        if not isinstance(old_labels, list):
            old_labels=[old_labels]
        if not isinstance(new_labels, list):
            new_labels=[new_labels]

        for x in self.data:
            x.replace_label(old_labels, new_labels)

        if self.left_label in old_labels:
            self.left_label = new_labels[old_labels.index(self.left_label)]
        if self.right_label in old_labels:
            self.right_label = new_labels[old_labels.index(self.right_label)]

    def standard_virtual_labels(self, suffix=""):
        """Replace `self.left_label` with "left"+`suffix` and 
        `self.right_label` with "right"+`suffix`."""

        self.replace_labels([self.left_label, self.right_label], 
                ["left"+suffix, "right"+suffix])

    def unique_virtual_labels(self):
        """Replace `self.left_label` and `self.right_label` with unique labels
        generated by tensor.unique_label()."""

        self.replace_labels([self.left_label, self.right_label], 
                [unique_label(), unique_label()])

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

    def __init__(self, tensors, left_label="left", right_label="right",
            phys_label="phys"):
        OneDimensionalTensorNetwork.__init__(self, tensors,
                left_label=left_label, right_label=right_label)
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

    def left_canonise(self, start=0, end=-1, chi=None, threshold=1e-14, 
            normalise=False, qr_decomposition=False):
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
        qr_decomposition : bool
            True specifies that a QR decomposition is performed rather than an
            SVD (which may improve performance). No truncation of singular
            values is possible with a QR decomposition, thus `chi` and
            `threshold` arguments are ignored.
        """
        N=len(self)
        if end==-1:
            end=N

        if qr_decomposition:
            for i in range(start,end):
                if i==N-1:
                    #The final QR has no right index, so R are just
                    #scalars. S is the norm of the state. 
                    if normalise==True and start==0: #Whole chain is canonised
                        self[i].data=self[i].data/np.linalg.norm(self[i].data)
                    return
                else:
                    qr_label=unique_label()
                    Q,R = tsr.tensor_qr(self[i], [self.phys_label, 
                        self.left_label], qr_label=qr_label)

                #Replace tensor at site i with Q
                Q.replace_label(qr_label+"in", self.right_label)
                self[i]=Q

                #Absorb R into next tensor
                self[i+1]=tsr.contract(R, self[i+1], self.right_label, 
                        self.left_label)
                self[i+1].replace_label(qr_label+"out", self.left_label)

        else:
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
                    svd_label=unique_label()
                    U,S,V = tsr.tensor_svd(self[i], [self.phys_label, 
                        self.left_label], svd_label=svd_label)

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

                U.replace_label(svd_label+"in", self.right_label)
                self[i]=U
                self[i+1]=tsr.contract(V, self[i+1], self.right_label, 
                        self.left_label)
                self[i+1]=tsr.contract(S, self[i+1], [svd_label+"in"], 
                        [svd_label+"out"])
                self[i+1].replace_label(svd_label+"out", self.left_label)

                #Reabsorb normalisation factors into next tensor
                #Note if i==N-1 (end of chain), this will not be reached 
                #and normalisation factors will be taken care of in the earlier 
                #block.
                if i==end-1:
                    self[i+1].data*=norm

    def right_canonise(self, start=0, end=-1, chi=None, threshold=1e-14, 
            normalise=False, qr_decomposition=False):
        """Perform right canonisation of MPS. Identical to `left_canonise`
        except that process is mirrored (i.e. canonisation is performed from
        right to left). `start` and `end` specify the interval to be canonised.

        Notes
        -----
        The first tensor to be canonised is `end`-1 and the final tensor to be
        canonised is `start`""" 

        self.reverse()
        N=len(self)
        if end==-1:
            end=N
        self.left_canonise(start=N-end, end=N-start, chi=chi,
                threshold=threshold, normalise=normalise,
                qr_decomposition=qr_decomposition)
        self.reverse()

    def replace_labels(self, old_labels, new_labels):
        """run `tensor.replace_label` method on every tensor in `self` then
        replace `self.left_label`, `self.right_label` and `self.phys_label` 
        appropriately."""

        if not isinstance(old_labels, list):
            old_labels=[old_labels]
        if not isinstance(new_labels, list):
            new_labels=[new_labels]

        for x in self.data:
            x.replace_label(old_labels, new_labels)

        if self.left_label in old_labels:
            self.left_label = new_labels[old_labels.index(self.left_label)]
        if self.right_label in old_labels:
            self.right_label = new_labels[old_labels.index(self.right_label)]
        if self.phys_label in old_labels:
            self.phys_label = new_labels[old_labels.index(self.phys_label)]

    def standard_labels(self, suffix=""):
        """
        overwrite self.labels, self.left_label, self.right_label, 
        self.phys_label with standard labels "left", "right", "phys"
        """
        self.replace_labels([self.left_label, self.right_label,
            self.phys_label], ["left"+suffix, "right"+suffix, "phys"+suffix])

    def check_canonical_form(self, threshold=1e-14, print_output=True):
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

    def svd_compress(self, chi=None, threshold=1e-15, normalise=False,
            reverse=False):
        """Compress MPS to a given bond dimension `chi` or to a minimum
        singular value `threshold` using SVD compression as described in U.
        Schollwock, Ann. Phys. 326 (2011) 96-192. This is achieved by
        performing two successive canonisations. If `reverse` is False,
        canonisation is first performed from left to right (with QR
        decomposition) then the resulting state is canonised from right to left
        (using SVD decomposition). The resulting MPS is in right canonical
        form.  If `reverse` is True this is mirrored, resulting in a state in
        left canonical form. """
        if reverse:
            self.reverse()
        self.left_canonise(normalise=False, qr_decomposition=True)
        #Normalise the state temporarily
        norm=self.norm(canonical_form="left")
        self[-1].data/=norm
        self.right_canonise(chi=chi, threshold=threshold, normalise=False)
        if normalise==False:
            self[0].data*=norm
        if reverse:
            self.reverse()

    def variational_compress(self, chi, max_iter=10, initial_guess=None,
            tolerance=1e-15, normalise=False):
        """Compress MPS to a given bond dimension `chi` or to the same bond
        dimensions as an optional input MPS `initial_guess` using an iterative
        compression procedure described in U. Schollwock, Ann. Phys. 326 (2011)
        96-192. The algorithm will start from an initial guess for the target
        MPS, either computed with the `svd_compress` method or, if supplied,
        with the `initial_guess` keyword argument.  It will sweep over the
        chain, successively optimising individual tensors until convergence.
        The output MPS will be in right canonical form. Should be more
        accurate, albeit slower, than `svd_compress` method. 
        
        Parameters
        ----------

        chi : int
            Bond dimension of resulting MPS.

        max_iter : int
            Maximum number of full updates to perform, where a full update
            consists of a sweep from  right to left, then left to right. If
            convergence is not reached after `max_iter` full updates, an error
            will be returned.

        initial_guess : MatrixProductState
            Starting point for variational algorithm. Output MPS will have the
            same bond dimension as `initial_guess`. If not provided, an SVD
            compression of the input MPS will be computed and used as the
            starting point. 

        tolerance : float
            After a full update is completed, the difference in norm with the
            target state for the last two sweeps is computed. The algorithm
            will be regarded as having converged and will stop if this
            difference is less than `tolerance`.
        """
        if initial_guess == None:
            mps=self.copy()
            #Make sure state is in left canonical form to start
            mps.svd_compress(chi=chi, reverse=True)
        else:
            mps=initial_guess
            #Put state in left canonical form
            mps.left_canonise(qr_decomposition=True)

        #Give mps1 unique labels
        mps.replace_labels([mps.left_label, mps.right_label, mps.phys_label], 
                [unique_label(), unique_label(), unique_label()])

        le_label=unique_label()
        left_environments = ladder_contract(mps, self, mps.phys_label,
                self.phys_label, return_intermediate_contractions=True,
                right_output_label=le_label, complex_conjugate_array1=True)

        def variational_sweep(mps1, mps2, left_environments):
            """Iteratively update mps1, to minimise frobenius distance to mps2
            by sweeping from right to left. Expects mps1 to be in right
            canonical form."""

            #Get the base label of left_environments
            le_label=left_environments[0].labels[0][:-1]
            #Generate some unique labels to avoid conflicts
            re_label=unique_label()
            lq_label=unique_label()

            right_environments=[]
            norms=[mps1[-1].norm()]
            for i in range(mps2.nsites-1, 0, -1):

                #Optimise the tensor at site i by contracting with left and 
                #right environments
                updated_tensor=tsr.contract(mps2[i], left_environments[i-1],
                    mps2.left_label, le_label+"2")
                if i!=mps2.nsites-1:
                    updated_tensor=tsr.contract(updated_tensor, 
                            right_environment, mps2.right_label, re_label+"2")
                    updated_tensor.replace_label(re_label+"1", 
                            mps1.right_label)
                updated_tensor.replace_label([le_label+"1", mps2.phys_label]
                        , [mps1.left_label, mps1.phys_label])

                #Right canonise the tensor at site i using LQ decomposition
                #Absorb L into tensor at site i-1
                L, Q = tsr.tensor_lq(updated_tensor, mps1.left_label,
                        lq_label=lq_label)
                Q.replace_label(lq_label+"out", mps1.left_label)
                L.replace_label(lq_label+"in", mps1.right_label)
                mps1[i]=Q
                mps1[i-1]=tsr.contract(mps1[i-1], L, mps1.right_label, 
                        mps1.left_label)

                #Compute norm of mps
                #Taking advantage of canonical form
                norms.append(mps1[i-1].norm())

                #Compute next column of right_environment
                if i==mps2.nsites-1:
                    right_environment=tsr.contract(tsr.conjugate(mps1[i]), 
                            mps2[i], mps1.phys_label, self.phys_label)
                    right_environment.remove_all_dummy_indices(
                            labels=[mps1.right_label, mps2.right_label])
                else:
                    right_environment.contract(tsr.conjugate(mps1[i]), 
                            re_label+"1", mps1.right_label)
                    right_environment.contract(mps2[i], [mps1.phys_label,
                        re_label+"2"], [self.phys_label, self.right_label])

                right_environment.replace_label([mps1.left_label,
                    mps2.left_label], [re_label+"1", re_label+"2"])
                right_environments.append(right_environment.copy())

                #At second last site, compute final tensor
                if i==1:
                    updated_tensor=tsr.contract(mps2[0], right_environment,
                            mps2.right_label, re_label+"2")
                    updated_tensor.replace_label([mps2.phys_label, 
                        re_label+"1"],
                            [mps1.phys_label, mps1.right_label])
                    mps1[0]=updated_tensor

            return right_environments, np.array(norms)

        for i in range(max_iter):
            left_environments, norms1 = variational_sweep(mps, self, 
                    left_environments)
            mps.reverse()
            self.reverse()
            le_label=left_environments[0].labels[0][:-1]
            left_environments, norms2 = variational_sweep(mps, self, 
                    left_environments)
            mps.reverse()
            self.reverse()
            #Compute differences between norms of successive updates in second
            #sweep. As shown in U. Schollwock, Ann. Phys. 326 (2011) 96-192,
            #these quantities are equivalent to the differences between the
            #frobenius norms between the target state and the variational
            #state.
            if np.all(np.abs(norms2[1:]-norms2[:-1])/norms2[1:] < tolerance):
                mps.replace_labels([mps.left_label, mps.right_label,
                    mps.phys_label], [self.left_label, self.right_label,
                        self.phys_label])
                if normalise==True:
                    mps[-1].data/=mps.norm(canonical_form="left")
                return mps
            elif i==max_iter-1: #Has reached the last iteration
                raise RuntimeError("variational_compress did not converge.")

    def physdim(self, site):
        """Return physical index dimesion for site"""
        return self.data[site].index_dimension(self.phys_label)

    def norm(self, canonical_form=False):
        """Return norm of mps.

        Parameters
        ----------

        canonical_form : str
            If `canonical_form` is "left", the state will be assumed to be in
            left canonical form, if "right" the state will be assumed to be in
            right canonical form. In these cases the norm can be read off the 
            last tensor (much more efficient). 
        """

        if canonical_form=="left":
            return np.linalg.norm(self[-1].data)
        elif canonical_form=="right":
            return np.linalg.norm(self[0].data)
        else:
            return np.sqrt(inner_product_mps(self, self))

    def apply_gate(self, gate, firstsite, gate_outputs=None, gate_inputs=None,
            chi=None, threshold=1e-15, canonise='left'):
        """
        Apply Tensor `gate` on sites `firstsite`, `firstsite`+1, ...,
        `firstsite`+`nsites`-1, where `nsites` is the length of gate_inputs.
        The physical index of the nth site is contracted with the nth label of 
        `gate_inputs`. After the contraction the MPS is put back into the 
        original form by SVD, and the nth sites physical index is given 
        by the nth label of `gate_outputs` (but relabeled to `self.phys_label` 
        to preserve the original MPS form).

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
        threshold : float
            Lower bound on the magnitude of singular values to keep. Singular
            values less than or equal to this value will be truncated.
        chi : int
            Maximum number of singular values of each tensor to keep after
            performing singular-value decomposition.
        canonise : str {'left', 'right'}
            Direction in which to canonise the sites after applying gate.

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
        if canonise == 'right':
            phys_labels = gate_outputs[::-1]
            left_label = 'right'
            right_label = 'left'
        else:
            phys_labels = gate_outputs
            left_label = 'left'
            right_label = 'right'
        mps = tensor_to_mps(t, phys_labels=phys_labels,
                mps_phys_label=self.phys_label, left_label=left_label,
                right_label=right_label, chi=chi, threshold=threshold)
        if canonise == 'right':
            mps.reverse()
        self.data[firstsite:firstsite+nsites] = mps.data


class MatrixProductStateCanonical(OneDimensionalTensorNetwork):
    """
    Matrix product state in canonical form with every other tensor assumed to
    be a diagonal matrix of singular values. The site numbering is

    0     1      2      3     ... N-2   N-1

    Lambda Gamma Lambda Gamma ... Gamma Lambda

    where the Gammas are rank three tensors and the Lambdas diagonal matrices 
    of singular values. The left-mots and right-most Lambda matrices are 
    trivial one-by-one matrices inserted for convenience.

    Convenient for TEBD type algorithms.

    See U. Schollwock, Ann. Phys. 326 (2011) 96-192 section 4.6.
    """

    def __init__(self, tensors, left_label="left", right_label="right",
            phys_label="phys"):
        OneDimensionalTensorNetwork.__init__(self, tensors,
                left_label=left_label, right_label=right_label)
        self.phys_label=phys_label

    def __repr__(self):
        return ("MatrixProductStateCanonical(tensors=%r, left_label=%r,"
            "right_label=%r, phys_label=%r)" % (self.data, self.left_label,
                self.right_label, self.phys_label))

    def __str__(self):
        return ("MatrixProductStateCanonical object: " +
              "sites (incl. singular value sites)= " + str(len(self)) + 
              ", left_label = " + self.left_label + 
              ", right_label = " + self.right_label + 
              ", phys_label = " + self.phys_label)

    def copy(self):
        """Return an MPS that is not linked in memory to the original."""
        return MatrixProductStateCanonical([x.copy() for x in self],
                self.left_label, self.right_label, self.phys_label)

    def physdim(self, site):
        """Return physical index dimesion for site"""
        return self.data[site].index_dimension(self.phys_label)

    def replace_labels(self, old_labels, new_labels):
        """run `tensor.replace_label` method on every tensor in `self` then
        replace `self.left_label`, `self.right_label` and `self.phys_label` 
        appropriately."""

        if not isinstance(old_labels, list):
            old_labels=[old_labels]
        if not isinstance(new_labels, list):
            new_labels=[new_labels]

        for x in self.data:
            x.replace_label(old_labels, new_labels)

        if self.left_label in old_labels:
            self.left_label = new_labels[old_labels.index(self.left_label)]
        if self.right_label in old_labels:
            self.right_label = new_labels[old_labels.index(self.right_label)]
        if self.phys_label in old_labels:
            self.phys_label = new_labels[old_labels.index(self.phys_label)]

    def standard_labels(self, suffix=""):
        """
        overwrite self.labels, self.left_label, self.right_label, 
        self.phys_label with standard labels "left", "right", "phys"
        """
        self.replace_labels([self.left_label, self.right_label,
            self.phys_label], ["left"+suffix, "right"+suffix, "phys"+suffix])

    def physical_site(self, n):
        """ Return position of n'th physical (pos=2*n+1)"""
        return 2*n+1

    def singular_site(self, n):
        """ Return position of n'th singular value site (pos=2*n)"""
        return 2*n

    def apply_gate(self, gate, firstsite, gate_outputs=None, gate_inputs=None,
            chi=None, threshold=1e-15):
        """
        Apply two-site gate to `physical_site(firstsite)` and 
        `physical_site(firstsite+1)`, and perform optimal compression, assuming 
        canonical form.
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

        # contract the MPS sites first
        start = self.physical_site(firstsite)-1
        end = self.physical_site(firstsite+1)+1
        t = contract_virtual_indices(self, start, end,
                periodic_boundaries=False)

        # contract all physical indices with gate input indices
        t = tsr.contract(t, gate, self.phys_label, gate_inputs)

        # split big tensor into MPS form by exact SVD
        U, S, V = tsr.truncated_svd(t, [gate_outputs[0], self.left_label],
                chi=chi, threshold=threshold, absorb_singular_values=None)
        U.replace_label(["svd_in", gate_outputs[0]],
                [self.right_label, self.phys_label])
        V.replace_label(["svd_out", gate_outputs[1]],
                [self.left_label, self.phys_label])
        S1_inv = self[start].copy()
        S1_inv.inv()
        S2_inv = self[end].copy()
        S2_inv.inv()
        S.replace_label(["svd_out", "svd_in"], [self.left_label,
            self.right_label])
        self[start+1] = S1_inv[self.right_label,]*U[self.left_label,]
        self[start+2] = S
        self[start+3] = V[self.right_label,]*S2_inv[self.left_label,]



class MatrixProductOperator(OneDimensionalTensorNetwork):
    #TODO currently assumes open boundaries
    """Matrix product operator "is a list of tensors, each having and index 
    labelled "phys" and at least one of the indices "left", "right"
    Input is a list of tensors, with three up to three index labels, If the 
    labels aren't already specified as "left", "right", "physin", "physout" 
    need to specify which labels correspond to these using 
    arguments left_label, right_label, physin_label and physout_label. """
    def __init__(self, tensors, left_label="left", right_label="right", 
            physout_label="physout", physin_label="physin"):
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

    C=tsr.contract(tensor, array[0], temp_label, label2, index_slice1=[0])
    for i in range(1, len(array)):
        #TODO make this work
        C=tsr.contract(C, array[i], [array.right_label, temp_label], 
                [array.left_label, label2], index_slice1=[0,1])

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

def left_canonical_form_mps(orig_mps, chi=0, threshold=1e-14, 
        normalise=False):
    """
    Computes left canonical form of an MPS

    See also
    --------
    Tensor.left_canonise()
    """
    mps=orig_mps.copy()
    mps.left_canonise(chi=chi, threshold=threshold, normalise=normalise)
    return mps

def right_canonical_form_mps(orig_mps, chi=0, threshold=1e-14, 
        normalise=False):
    """Computes left canonical form of an MPS"""

    mps=orig_mps.copy()
    mps.right_canonise(chi=chi, threshold=threshold, normalise=normalise)
    return mps

def reverse_mps(mps):
    return MatrixProductState([x.copy() for x in reversed(mps)], 
            mps.right_label, mps.left_label, mps.phys_label)

def check_canonical_form_mps(mps, threshold=1e-14, print_output=True):
    mps.check_canonical_form(threshold=threshold,
            print_output=print_output)
    
def svd_compress_mps(orig_mps, chi, threshold=1e-15, normalise=False):
    """Simply right canonise the left canonical form according to Schollwock"""
    mps=left_canonical_form_mps(orig_mps, threshold=threshold, 
            normalise=normalise)
    return right_canonical_form_mps(mps, chi=chi, threshold=threshold, 
            normalise=normalise)

def variational_compress_mps(mps, chi, max_iter=10, initial_guess=None,
        tolerance=1e-15):
    return mps.variational_compress(chi, max_iter=max_iter,
            initial_guess=initial_guess, tolerance=tolerance)
   
def mps_complex_conjugate(mps):
    """Will take complex conjugate of every entry of every tensor in mps, 
    and append label_suffix to every label"""
    new_mps=mps.copy()
    for x in new_mps.data: 
        x.conjugate()
    return new_mps

def ladder_contract(array1, array2, label1, label2, start=0, end=None,
        complex_conjugate_array1=False, left_output_label="left",
        right_output_label="right", return_intermediate_contractions=False): 
    """
    Contract two one-dimensional tensor networks. Indices labelled `label1` in
    `array1` and indices labelled `label2` in `array2` are contracted pairwise
    and all virtual indices are contracted.  The contraction pattern
    resembles a ladder when represented graphically. 

    Parameters
    ----------

    array1 : OneDimensionalTensorNetwork
    array2 : OneDimensionalTensorNetwork
        The one-dimensional networks to be contracted.

    label1 : str
    label2 : str
        The index labelled `label1` is contracted with the index labelled
        `label2` for every site in array.

    start : int
    end : int
        The endpoints of the interval to be contracted. The leftmost tensors
        involved in the contraction are `array1[start]` and `array2[start]`,
        while the rightmost tensors are `array2[end]` and `array2[end]`. 

    complex_conjugate_array1 : bool
        Whether the complex conjugate of `array1` will be used, rather than
        `array1` itself. This is useful if, for instance, the two arrays are
        matrix product states and the inner product is to be taken (Note that
        inner_product_mps could be used in this case). 

    right_output_label : str
        Base label assigned to right-going indices of output tensor.
        Right-going indices will be assigned labels `right_output_label`+"1"
        and `right_output_label`+"2" corresponding, respectively, to `array1`
        and `array2`.

    left_output_label : str
        Base label assigned to left-going indices of output tensor. Left-going
        indices will be assigned labels `left_output_label`+"1" and
        `left_output_label`+"2" corresponding, respectively, to `array1` and
        `array2`.

    return_intermediate_contractions : bool
        If true, a list of tensors is returned. If the contraction is performed
        from left to right (see Notes below), the i-th entry contains the
        contraction up to the i-th contracted pair. If contraction is performed
        from right to left, this order is reversed (so the last entry
        corresponds to the contraction of the right-most pair tensors, which
        are first to be contracted).

    Returns
    -------
    tensor : Tensor
       Tensor obtained by contracting the two arrays. The tensor may have left
       indices, right indices, both or neither depending on the interval
       specified. 

    intermediate_contractions : list 
        If `return_intermediate_contractions` is true a list
        `intermediate_contractions` is returned containing a list of tensors
        corresponding to contraction up to a particular column.

    Notes
    -----
    If the interval specified contains the left open boundary, contraction is
    performed from left to right. If not and if interval contains right
    boundary, contraction is performed from right to left. If the interval
    does not contain either boundary, contraction is performed from left to
    right.
    """

    #If no end specified, will contract to end
    if end==None:
        end=min(array1.nsites, array2.nsites)-1 #index of the last site

    if end < start:
        raise ValueError("Badly defined interval (end before start).")

    a1=array1.copy()
    a2=array2.copy()

    if complex_conjugate_array1: 
        a1.complex_conjugate()

    #Give all contracted indices unique labels so no conflicts with other 
    #labels in array1, array2
    a1.unique_virtual_labels()
    a2.unique_virtual_labels()
    rung_label=unique_label()
    a1.replace_labels(label1, rung_label)
    a2.replace_labels(label2, rung_label)

    intermediate_contractions=[]
    if start==0: #Start contraction from left
        for i in range(0, end+1):
            if i==0:
                C=tsr.contract(a1[0], a2[0], rung_label, rung_label)
            else:
                C.contract(a1[i], a1.right_label, a1.left_label)
                C.contract(a2[i], [a2.right_label, rung_label], 
                        [a2.left_label, rung_label])

            if return_intermediate_contractions:
                t=C.copy()
                t.replace_label([a1.right_label, a2.right_label], 
                        [right_output_label+"1", right_output_label+"2"])
                #Remove dummy indices except the right indices
                t.remove_all_dummy_indices(labels=[x for x in t.labels if x
                    not in [right_output_label+"1", right_output_label+"2"]])
                intermediate_contractions.append(t)

        C.replace_label([a1.right_label, a2.right_label], 
                [right_output_label+"1", right_output_label+"2"])
        C.remove_all_dummy_indices()

    elif end==a1.nsites-1 and end==a2.nsites-1: #Contract from the right
        for i in range(end, start-1, -1):
            if i==end:
                C=tsr.contract(a1[end], a2[end], rung_label, rung_label)
            else:
                C.contract(a1[i], a1.left_label, a1.right_label)
                C.contract(a2[i], [a2.left_label, rung_label], 
                        [a2.right_label, rung_label])

            if return_intermediate_contractions:
                t=C.copy()
                t.replace_label([a1.left_label, a2.left_label], 
                        [left_output_label+"1", left_output_label+"2"])
                #Remove dummy indices except the left indices
                t.remove_all_dummy_indices(labels=[x for x in t.labels if x
                    not in [left_output_label+"1", left_output_label+"2"]])
                intermediate_contractions.insert(0,t)

        C.replace_label([a1.left_label, a2.left_label], 
                [left_output_label+"1", left_output_label+"2"])
        C.remove_all_dummy_indices()

    else: 
        #When an interval does not contain a boundary, contract in pairs first
        #then together
        for i in range(start, end+1):
            t=tsr.contract(a1[i], a2[i], rung_label, rung_label)
            if i==start:
                C=t
            else:
                C.contract(t, [a1.right_label, a2.right_label], 
                        [a1.left_label, a2.left_label])

            if return_intermediate_contractions:
                t=C.copy()
                t.replace_label([a1.right_label, a2.right_label, a1.left_label, 
                    a2.left_label], [right_output_label+"1", 
                        right_output_label+"2", left_output_label+"1", 
                        left_output_label+"2"])
                #Remove dummy indices except the left and right indices
                t.remove_all_dummy_indices(labels=[x for x in t.labels if x
                    not in [right_output_label+"1", right_output_label+"2", 
                        left_output_label+"1", left_output_label+"2"]])
                t.remove_all_dummy_indices()
                intermediate_contractions.append(t)

        C.replace_label([a1.right_label, a2.right_label, a1.left_label, 
            a2.left_label], [right_output_label+"1", right_output_label+"2", 
                left_output_label+"1", left_output_label+"2"])
        C.remove_all_dummy_indices()

    if return_intermediate_contractions:
        return intermediate_contractions
    else:
        return C

def inner_product_mps(mps_bra, mps_ket, complex_conjugate_bra=True, 
        return_whole_tensor=False):
    """Compute the inner product of two MatrixProductState objects."""
    t=ladder_contract(mps_bra, mps_ket, mps_bra.phys_label, mps_ket.phys_label,
            complex_conjugate_array1=complex_conjugate_bra)
    if return_whole_tensor:
        return t
    else:
        return t.data
    
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

    Notes
    -----
    The resulting MPS is left-canonised.
    """
    if phys_labels is None:
        phys_labels =[x for x in tensor.labels if x not in
                [left_label, right_label]]

    nsites = len(phys_labels)
    V = tensor.copy()
    mps = []
    for k in range(nsites-1):
        U, V, _ = tsr.truncated_svd(V, [left_label]*(left_label in V.labels)
                +[phys_labels[k]], chi=chi, threshold=threshold,
                absorb_singular_values='right')
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


def right_canonical_to_canonical(mps, chi=None, threshold=1e-14,
        normalise=False):
    """
    Turn an MPS in right canonical form into an MPS in canonical form
    """
    N=len(mps)

    #At each step will divide by a constant so that the largest singular 
    #value of S is 1. Will store the product of these constants in `norm`
    norm=1
    S_prev = tsr.Tensor([[1.0]], labels=[mps.left_label, mps.right_label])
    S_prev_inv = S_prev.copy()
    tensors = [S_prev, mps[0]]
    svd_label=unique_label()
    for i in range(N):
        if i==N-1:
            #The final SVD has no right index, so S and V are just scalars.
            #S is the norm of the state. 
            G = S_prev_inv[mps.right_label,]*tensors[-1][mps.left_label,]
            tensors[-1] = S_prev
            tensors.append(G)
            if normalise==True:
                tensors[-1].data=tensors[-1].data/np.linalg.norm(
                        tensors[-1].data)
            else:
                tensors[-1].data=tensors[-1].data*norm
            tensors.append(tsr.Tensor([[1.0]], labels=[mps.left_label,
                mps.right_label]))
        else:
            # Construct B = Gamma Lambda
            U,S,V = tsr.tensor_svd(tensors[-1], [mps.phys_label, 
                mps.left_label], svd_label=svd_label)

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
            U.replace_label(svd_label+"in", mps.right_label)

            G = S_prev_inv[mps.right_label,]*U[mps.left_label,]
            if i > 0:
                tensors[-1] = S_prev
                tensors.append(G)
            else:
                tensors[-1] = G
            # Store S and S^{-1} for next iteration
            S_prev = S.copy()
            S_prev.replace_label([svd_label+"out"], mps.left_label)
            S_prev.replace_label([svd_label+"in"], mps.right_label)
            S_prev_inv = S_prev.copy()
            S_prev_inv.data = np.diag(1./singular_values_to_keep)

            tensors.append(V[mps.right_label,]*mps[i+1][mps.left_label,])
            tensors[-1]=S[svd_label+"in",]*tensors[-1][svd_label+"out",]
            tensors[-1].replace_label(svd_label+"out", mps.left_label)

            #Reabsorb normalisation factors into next tensor
            if i==N-1:
                tensors[-1].data*=norm

    # Construct empty MPS in canonical form
    return MatrixProductStateCanonical(tensors,
            left_label=mps.left_label, right_label=mps.right_label,
            phys_label=mps.phys_label)


