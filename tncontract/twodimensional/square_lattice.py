import numpy as np

from ..onedimensional import one_dimension as od

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
    def __init__(self, tensors, up_label, right_label, down_label, left_label, 
            copy_data=True):
        self.up_label=up_label
        self.right_label=right_label
        self.down_label=down_label
        self.left_label=left_label

        if copy_data:
            #Creates copies of tensors in memory
            copied_tensors=[]
            for row in tensors:
                copied_tensors.append([x.copy() for x in row])
            self.data=np.array(copied_tensors)
        else:
            #This will not create copies of tensors in memory 
            #(just link to originals)
            self.data=np.array(tensors)

        #Every tensor will have four indices corresponding to 
        #"left", "right" and "up", "down" labels. 
        for i,x in np.ndenumerate(self.data):
            if left_label not in x.labels: x.add_dummy_index(left_label)
            if right_label not in x.labels: x.add_dummy_index(right_label)
            if up_label not in x.labels: x.add_dummy_index(up_label)
            if down_label not in x.labels: x.add_dummy_index(down_label)

    #Add container emulation
    def __iter__(self):
        return self.data.__iter__()
    def __len__(self):
        return self.data.__len__()
    def __getitem__(self, key):
        return self.data.__getitem__(key)
    def __setitem__(self, key, value):
        self.data.__setitem__(key, value)

    @property
    def shape(self):
        return self.data.shape

    def is_left_right_periodic(self):
        """Check whether state is periodic by checking whether the left 
        indices of the first column have dimension greater than one"""
        for x in self[:,0]:
            if x.index_dimension(self.left_label)>1:
                return True
        return False

    def exact_contract(self, until_column=-1):
        """Will perform exact contraction of all virtual indices of the square
        lattice, starting from the top-left, contracting the whole first 
        column, then contracting one column at a time."""
        rows, cols = self.data.shape
        mpo=column_to_mpo(self, 0)
        C=od.contract_virtual_indices(mpo)
        for i in range(1, cols):
            if i==until_column+1:
                #Return the contraction early
                return C
            mpo=column_to_mpo(self, i)
            C=od.contract_multi_index_tensor_with_one_dim_array(C, mpo, 
                    self.right_label, self.left_label)
            C.remove_all_dummy_indices()
        return C

    def mps_contract(self, chi, compression_type="svd", normalise=False, 
            until_column=-1):
        """Approximately contract a square lattice tensor network using MPS 
        evolution and compression. Will contract from left to right.
        If normalise is set to True, the normalise argument to 
        svd_compress_mps will be set to true."""

        nrows, ncols = self.shape

        for col in range(ncols-1):
            if col==0:
                mps_to_compress = column_to_mpo(self, 0, to_mps=True)
            else:
                column_mpo=column_to_mpo(self, col)
                mps_to_compress = od.contract_mps_mpo(compressed_mps, 
                        column_mpo)

            if compression_type=="svd":
                compressed_mps = od.svd_compress_mps(mps_to_compress, chi, 
                        normalise=normalise)
            elif compression_type=="variational":
                compressed_mps = od.variational_compress_mps(mps_to_compress, 
                        chi, max_iter=10)

            if col == until_column:
                return compressed_mps

        #For final column, compute contraction exactly
        final_column_mps=column_to_mpo(self, ncols-1, to_mps=True)
        return od.inner_product_mps(compressed_mps, final_column_mps, 
                return_whole_tensor=True, complex_conjugate_bra=False)

class SquareLatticePEPS(SquareLatticeTensorNetwork):
    def __init__(self, tensors, up_label, right_label, down_label, left_label,
            phys_label):
        SquareLatticeTensorNetwork.__init__(self, tensors, up_label, 
                right_label, down_label, left_label)
        self.phys_label=phys_label

def column_to_mpo(square_tn, col, to_mps=False):
    """
    Will extract column col from square_tn (which is assumed to be a 
    SquareLatticeTensorNetwork object), and convert the column into a
    MatrixProductState object (if first or last column without periodic 
    boundary conditions) or a MatrixProductOperator object. 
    If to_mps==True and col refers to either the first or last column, 
    a MatrixProductState object is returned instead .
    
    """
    new_data=square_tn[:,col].copy()
    if to_mps and (col==0 or col==square_tn.shape[1]-1):
        if col==0:
            new_mps=od.MatrixProductState(new_data, square_tn.up_label, 
                    square_tn.down_label, square_tn.right_label)
            for x in new_mps.data:
                x.remove_all_dummy_indices(square_tn.left_label)
        else: #Last column
            new_mps=od.MatrixProductState(new_data, square_tn.up_label, 
                    square_tn.down_label, square_tn.left_label)
            for x in new_mps.data:
                x.remove_all_dummy_indices(square_tn.right_label)
        return new_mps
    else:
        return od.MatrixProductOperator(new_data, square_tn.up_label, 
            square_tn.down_label, square_tn.right_label, 
            square_tn.left_label)


