from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

"""
square_lattice
==========

Core module for square lattice tensor networks
"""

import numpy as np

import tncontract.onedim as od
import tncontract as tn


class SquareLatticeTensorNetwork():
    """Base class for square lattices, e.g. square-lattice PEPS and PEPO.
    The argument "tensors" is a two-dimensional array (a list of lists or 2D 
    numpy array) of Tensor objects.
    Each tensor is expected to have have four indices: up, down, left, right. 
    The labels corresponding these indices are specified using the required 
    arguments : left_label, right_label, up_label, down_label
    If the state has open boundaries, the edge indices of tensors on the 
    boundary should have dimension 1. If not, the tensors will be put in this 
    form."""

    def __init__(self, tensors, up_label="up", right_label="right",
                 down_label="down", left_label="left",
                 copy_data=True):
        self.up_label = up_label
        self.right_label = right_label
        self.down_label = down_label
        self.left_label = left_label

        if copy_data:
            # Creates copies of tensors in memory
            copied_tensors = []
            for row in tensors:
                copied_tensors.append([x.copy() for x in row])
            self.data = np.array(copied_tensors)
        else:
            # This will not create copies of tensors in memory
            # (just link to originals)
            self.data = np.array(tensors)

        # Every tensor will have four indices corresponding to
        # "left", "right" and "up", "down" labels.
        for i, x in np.ndenumerate(self.data):
            if left_label not in x.labels: x.add_dummy_index(left_label)
            if right_label not in x.labels: x.add_dummy_index(right_label)
            if up_label not in x.labels: x.add_dummy_index(up_label)
            if down_label not in x.labels: x.add_dummy_index(down_label)

    # Add container emulation
    def __iter__(self):
        return self.data.__iter__()

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, key):
        return self.data.__getitem__(key)

    def __setitem__(self, key, value):
        self.data.__setitem__(key, value)

    def copy(self):
        """Return a copy of SquareLatticeTensorNetwork that is not linked in
        memory to the original."""
        return SquareLatticeTensorNetwork(self.data, 
                up_label=self.up_label, right_label=self.right_label, 
                down_label=self.down_label, left_label=self.left_label,
                copy_data=True)
    @property
    def shape(self):
        return self.data.shape

    def is_left_right_periodic(self):
        """Check whether state is periodic by checking whether the left 
        indices of the first column have dimension greater than one"""
        for x in self[:, 0]:
            if x.index_dimension(self.left_label) > 1:
                return True
        return False

    def can_contract(self):
        """Check whether the virtual indices of the tensor network can be 
        contracted, based on bond dimensions."""
        rows, cols = self.data.shape
        left_unmatched=[]
        up_unmatched=[]
        for i in range(1,rows):
            for j in range(1,cols):
                #Check all horizontal and vertical bonds
                if (self[i,j].index_dimension(self.up_label)!=
                        self[i-1,j].index_dimension(self.down_label)):
                    left_unmatched.append((i,j))
                if (self[i,j].index_dimension(self.left_label)!=
                        self[i,j-1].index_dimension(self.right_label)):
                    up_unmatched.append((i,j))
        if len(left_unmatched) == 0 and len(up_unmatched) == 0:
            return True
        else:
            print("Unmatched bonds found between the following sites:")
            for k in left_unmatched:
                print("("+str(k[0]-1)+", "+ str(k[1])+")"+" and "+str(k))
            for k in up_unmatched:
                print("("+str(k[0])+", "+ str(k[1]-1)+")"+" and "+str(k))
            return False

    def exact_contract(self, until_column=-1):
        """Will perform exact contraction of all virtual indices of the square
        lattice, starting from the top-left, contracting the whole first 
        column, then contracting one column at a time."""
        rows, cols = self.data.shape
        mpo = column_to_mpo(self, 0)
        C = od.contract_virtual_indices(mpo)
        for i in range(1, cols):
            if i == until_column + 1:
                # Return the contraction early
                return C
            mpo = column_to_mpo(self, i)
            C = od.contract_multi_index_tensor_with_one_dim_array(C, mpo,
                                            self.right_label, self.left_label)
            C.remove_all_dummy_indices([self.left_label, self.up_label, 
                                                self.down_label])
        return C

    def mps_contract(self, chi, compression_type="svd", until_column=-1, 
            max_iter=10, tolerance=1e-14, return_all_columns = False):
        """Approximately contract a square lattice tensor network using MPS 
        evolution and compression. Will contract from left to right.
        If `return_all_columns` is true, will return a list of MPS 
        corresponding to the contraction up to each column.
        """

        nrows, ncols = self.shape

        if return_all_columns:
            column_list=[]

        # Divide matrix product state by its norm after each compression
        # but keep these factors in the variable `norm`
        norm = 1
        for col in range(ncols - 1):
            if col == 0:
                mps_to_compress = column_to_mpo(self, 0)
            else:
                column_mpo = column_to_mpo(self, col)
                mps_to_compress = od.contract_mps_mpo(compressed_mps,
                                                      column_mpo)

            if compression_type == "svd":
                compressed_mps = od.svd_compress_mps(mps_to_compress, chi,
                                                     normalise=False)
                # Normalise MPS (although keep normalisation factor in `norm`)
                mps_norm = compressed_mps.norm(canonical_form="right")
                #Return 0 if the norm of the MPS is zero
                if mps_norm==0.0: 
                    return 0.0
                compressed_mps[0].data = compressed_mps[0].data / mps_norm
                norm *= mps_norm
            elif compression_type == "variational":
                compressed_mps = mps_to_compress.variational_compress(
                    chi, max_iter=max_iter, tolerance=tolerance)
                # Normalise MPS (although keep normalisation factor in `norm`)
                mps_norm = compressed_mps.norm(canonical_form="left")
                #Return 0 if the norm of the MPS is zero
                if mps_norm==0.0: 
                    return 0.0
                compressed_mps[-1].data = compressed_mps[-1].data / mps_norm
                norm *= mps_norm

            if return_all_columns:
                mps_copy=compressed_mps.copy()
                mps_copy[0].data *= norm
                column_list.append(mps_copy)

            if col == until_column:
                if return_all_columns:
                    return column_list
                elif compression_type == "svd":
                    compressed_mps[0].data *= norm
                    return compressed_mps
                elif compression_type == "variational":
                    compressed_mps[-1].data *= norm
                    return compressed_mps

        # For final column, compute contraction exactly
        final_column_mps = column_to_mpo(self, ncols - 1)
        full_contraction = od.inner_product_mps(compressed_mps, 
               final_column_mps, return_whole_tensor=True, 
               complex_conjugate_bra=False) * norm
        if return_all_columns:
            column_list.append(full_contraction)
            return column_list
        else:
            return full_contraction

    def col_to_1D_array(self, col):
        """
        Will extract column col from square_tn (which is assumed to be a 
        SquareLatticeTensorNetwork object), and convert the column into a
        MatrixProductState object (if first or last column without periodic 
        boundary conditions) or a MatrixProductOperator object. 
        """
        new_data = self[:, col].copy()
        return od.OneDimensionalTensorNetwork(new_data,
                                              left_label=self.up_label,
                                              right_label=self.down_label)

    def fliplr(self):
        """
        Returns left-right mirror image of TN. Note: will not modify labels of
        constituent tensors, but will switch the `left_label` and `right_label`
        attributes of `SquareLatticeTensorNetwork`. 
        """
        mirror_tn=self.copy()
        mirror_data=np.fliplr(mirror_tn)
        mirror_tn.data=mirror_data
        mirror_tn.right_label=self.left_label
        mirror_tn.left_label=self.right_label

        return mirror_tn
        



class SquareLatticePEPS(SquareLatticeTensorNetwork):
    def __init__(self, tensors, up_label="up", right_label="right",
                 down_label="down", left_label="left", phys_label="phys", 
                 copy_data=True):
        SquareLatticeTensorNetwork.__init__(self, tensors, up_label,
                                            right_label, down_label, left_label, 
                                            copy_data=copy_data)
        self.phys_label = phys_label

    def copy(self):
        """Return a copy of SquareLatticePEPS that is not linked in
        memory to the original."""
        return SquareLatticePEPS(self.data, 
                up_label=self.up_label, right_label=self.right_label, 
                down_label=self.down_label, left_label=self.left_label,
                phys_label=self.phys_label, copy_data=True)

    def outer_product(self, physin_label="physin", physout_label="physout"):
        """
        Take the outer product of this PEPS with itself, returning a PEPO. 
        The outer product of each  tensor in the PEPS is taken and 
        virtual indices are consolidated. Returns an instance of SquareLatticePEPO."""
        tensor_array=[]
        for row in range(self.shape[0]):
            new_row=[]
            for col in range(self.shape[1]):
                #This takes the outer product of two tensors
                #Without contracting any indices
                outer = tn.contract(self[row,col], self[row,col], [], []) 
                #Replace the first physical label with physin label
                outer.labels[outer.labels.index(self.phys_label)]=physin_label
                #Replace the second physical label with physin label
                outer.labels[outer.labels.index(self.phys_label)]=physout_label

                #Consolidate indices
                outer.consolidate_indices(labels=[self.left_label, 
                    self.right_label, self.up_label, self.down_label])

                new_row.append(outer)
            tensor_array.append(new_row)

        return SquareLatticePEPO(tensor_array, up_label=self.up_label, 
                down_label=self.down_label, right_label=self.right_label,
                left_label=self.left_label, physin_label=physin_label,
                physout_label=physout_label)

    #Alias for outer_product
    density_operator = outer_product

def inner_product_peps(peps_ket, peps_bra, exact_contract="True", 
        complex_conjugate_bra=True, compression_type="svd", chi=2,  
        max_iter=10, tolerance=1e-14, contract_virtual=True):
    new_tensors=[] #Tensors formed by contracting the physical indices of peps_ket and bra
    for i in range(peps_ket.shape[0]):
        new_row=[]
        for j in range(peps_ket.shape[1]):
            t=(tn.tensor.conjugate(peps_bra[i,j])["phys"]*
                    peps_ket[i,j]["phys"])
            t.consolidate_indices()
            new_row.append(t)
        new_tensors.append(new_row)
    
    ip=SquareLatticeTensorNetwork(new_tensors)
    if not contract_virtual:
        return ip

    if exact_contract:
        return ip.exact_contract()
    else:
        return ip.mps_contract(chi, compression_type=compression_type, 
                    max_iter=max_iter, tolerance=tolerance)

def outer_product_peps(peps1, peps2, physin_label="physin", 
        physout_label="physout"):
    """Return the outer product of two PEPS networks i.e. if `peps1` and
    `peps2` correspond to two PEPS |a> and |b>  then outer_product_peps(peps1,
    peps2) returns the density operator corresponding to |a><b|, where "physin"
    is the physical index associated with <b|" and "physout" is associated with
    |a>.  Assumes that input PEPS are the same size. The output physin label
    replaces the phys label of `peps2` and the output label physout replaces
    the phys label  of `peps1`."""
    #TODO input PEPS must have the same left right up down labels. Check this
    #TODO careful for conflicting phys labels of peps1 and peps2
    if peps1.shape != peps2.shape:
        raise ValueError("Peps input do not have same dimension.")
    tensor_array=[]
    for row in range(peps1.shape[0]):
        new_row=[]
        for col in range(peps1.shape[1]):
            #This takes the outer product of two tensors
            #Without contracting any indices
            outer = tn.contract(peps1[row,col], 
                    tn.tensor.conjugate(peps2[row,col]), [], []) 
            #Replace the physical label of peps1 with  physout label
            outer.labels[outer.labels.index(peps1.phys_label)]=physout_label
            #Replace the physical label of peps2 with physin label
            outer.labels[outer.labels.index(peps2.phys_label)]=physin_label

            #Consolidate indices
            outer.consolidate_indices(labels=[peps1.left_label, 
                peps1.right_label, peps1.up_label, peps1.down_label])

            new_row.append(outer)
        tensor_array.append(new_row)

    return SquareLatticePEPO(tensor_array, up_label=peps1.up_label, 
                down_label=peps1.down_label, right_label=peps1.right_label,
                left_label=peps1.left_label, physin_label=physin_label,
                physout_label=physout_label)

class SquareLatticePEPO(SquareLatticeTensorNetwork):
    def __init__(self, tensors, up_label="up", right_label="right",
                 down_label="down", left_label="left", physin_label="physin",
                 physout_label="physout", copy_data=True):
        SquareLatticeTensorNetwork.__init__(self, tensors, up_label,
                                            right_label, down_label, left_label, copy_data=copy_data)
        self.physin_label = physin_label
        self.physout_label = physout_label

    def copy(self):
        """Return a copy of SquareLatticePEPO that is not linked in
        memory to the original."""
        return SquareLatticePEPO(self.data, 
                up_label=self.up_label, right_label=self.right_label, 
                down_label=self.down_label, left_label=self.left_label,
                physin_label=self.physin_label, 
                physout_label=self.physout_label, copy_data=True)

    def trace(self):
        """Contract the physin and physout indices of every tensor. Returns
        an instance of SquareLatticeTensorNetwork."""
        tensor_array=[]
        for i in range(self.shape[0]):
            row=[]
            for j in range(self.shape[1]):
                tmp=self[i,j].copy()
                tmp.trace(self.physin_label, self.physout_label)
                row.append(tmp)
            tensor_array.append(row)
        return SquareLatticeTensorNetwork(tensor_array, up_label=self.up_label,
                down_label=self.down_label, right_label=self.right_label, 
                left_label=self.left_label)

def apply_pepo_to_peps(peps, pepo):
    nrows, ncols = peps.shape

    new_tensors=[]
    for i in range(nrows):
        new_row=[]
        for j in range(ncols):
            new_tensor=peps[i,j][peps.phys_label]*pepo[i,j][pepo.physin_label]
            new_tensor.replace_label(pepo.physout_label, peps.phys_label)
            new_tensor.consolidate_indices()
            new_row.append(new_tensor)
        new_tensors.append(new_row)
    return SquareLatticePEPS(new_tensors, 
                up_label=peps.up_label, right_label=peps.right_label, 
                down_label=peps.down_label, left_label=peps.left_label,
                phys_label=peps.phys_label)

def column_to_mpo(square_tn, col):
    """
    Will extract column col from square_tn (which is assumed to be a 
    SquareLatticeTensorNetwork object), and convert the column into a
    MatrixProductState object (if first or last column without periodic 
    boundary conditions) or a MatrixProductOperator object. 
    """
    new_data = square_tn[:, col].copy()
    if col == 0 or col == square_tn.shape[1] - 1:
        if col == 0:
            new_mps = od.MatrixProductState(new_data, square_tn.up_label,
                                            square_tn.down_label, square_tn.right_label)
            for x in new_mps.data:
                x.remove_all_dummy_indices(square_tn.left_label)
        else:  # Last column
            new_mps = od.MatrixProductState(new_data, square_tn.up_label,
                                            square_tn.down_label, square_tn.left_label)
            for x in new_mps.data:
                x.remove_all_dummy_indices(square_tn.right_label)
        return new_mps
    else:
        return od.MatrixProductOperator(new_data, square_tn.up_label,
                                        square_tn.down_label, square_tn.right_label,
                                        square_tn.left_label)
