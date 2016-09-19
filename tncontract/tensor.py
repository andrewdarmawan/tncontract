import numpy as np

class Tensor():
    def __init__(self, data=None, labels=[]):
        self.labels=labels
        self.data=data

    def __repr__(self):
        return "Tensor(data=%r, labels=%r)" % (self.data, self.labels)

    def __str__(self):
        return ("Tensor object: "
                "shape = " + str(self.shape) +
                ", labels = " + str(self.labels) + "\n" +
                "Tensor data = \n" + str(self.data))

    def replace_label(self, old_labels, new_labels):
        """
        Takes two lists old_labels, new_labels as arguments. If a label in self.labels is in old_labels,
        it is replaced with the respective label in new_labels.
        """

        #If either argument is not a list, convert to list with single entry
        if not isinstance(old_labels, list):
            old_labels=[old_labels]
        if not isinstance(new_labels, list):
            new_labels=[new_labels]

        for i,label in enumerate(self.labels):
            if label in old_labels:
                self.labels[i]=new_labels[old_labels.index(label)]

    def prime_label(self, labels):
        """
        Add suffix "_p" to any label of the form "label" or "label_p_p..._p"
        for all `label` in `labels`

        See also
        -------
        unprime_label
        """
        if not isinstance(labels, list):
            labels=[labels]
        for i, label in enumerate(self.labels):
            for unprimedlabel in labels:
                if label.startswith(unprimedlabel):
                    primes = label[len(unprimedlabel):]
                    if primes == '_p'*int(len(primes)/2):
                        self.labels[i] += '_p'

    def unprime_label(self, labels):
        """
        Remove the last "_p" from any label of the form "label_p_p.._p"
        for all `label` in `labels`

        Examples
        --------
        >>> t = Tensor(np.array([1,0]), labels=['idx'])
        >>> t.prime_label('idx')
        >>> print(t)
        Tensor object: shape = (2,), labels = ['idx_p']
        Tensor data =
        [1 0]
        >>> t.prime_label('idx')
        >>> print(t)
        Tensor object: shape = (2,), labels = ['idx_p_p']
        Tensor data =
        [1 0]
        >>> t.unprime_label('idx')
        >>> print(t)
        Tensor object: shape = (2,), labels = ['idx_p']
        Tensor data =
        [1 0]
        >>> t.unprime_label('idx')
        >>> print(t)
        Tensor object: shape = (2,), labels = ['idx']
        Tensor data =
        [1 0]
        """
        if not isinstance(labels, list):
            labels=[labels]
        for i, label in enumerate(self.labels):
            for unprimedlabel in labels:
                if label.startswith(unprimedlabel):
                    primes = label[len(unprimedlabel):]
                    if primes == '_p'*int(len(primes)/2):
                        self.labels[i]=label[:-2]

    def contract_internal(self, label1, label2, index1=0, index2=0):
        """By default will contract the first index with label1 with the 
        first index with label2. index1 and index2 can be specified to contract 
        indices that are not the first with the specified label."""

        label1_indices=[i for i,x in enumerate(self.labels) if x==label1]
        label2_indices=[i for i,x in enumerate(self.labels) if x==label2]

        index_to_contract1=label1_indices[index1]
        index_to_contract2=label2_indices[index2]

        self.data=np.trace(self.data, axis1=index_to_contract1, axis2=index_to_contract2)

        #The following removes the contracted indices from the list of labels 
        self.labels=[label for j,label in enumerate(self.labels) 
            if j not in [index_to_contract1, index_to_contract2]]

    def consolidate_indices(self):
        """Combines all indices with the same label into a single label.
        Also puts labels in alphabetical order (and reshapes data accordingly). 
        """
        labels_unique=sorted(set(self.labels))
        for p,label in enumerate(labels_unique):
            indices = [i for i,j in enumerate(self.labels) if j == label]
            #Put all of these indices together
            for k,q in enumerate(indices):
                self.data=np.rollaxis(self.data,q,p+k)
            #Total dimension of all indices with label
            total_dim=self.data.shape[p]
            for r in range(1,len(indices)):
                total_dim=total_dim*self.data.shape[p+r]
            #New shape after consolidating all indices with label into 
            #one at position p
            new_shape=list(self.data.shape[0:p])+[total_dim]+list(self.data.shape[p+len(indices):])
            self.data=np.reshape(self.data,tuple(new_shape))

            #Update self.labels
            #Remove all instances of label from self.labels
            self.labels=[x for x in self.labels if x != label]
            #Reinsert label at position p
            self.labels.insert(p, label)
    def sort_labels(self):
        self.consolidate_indices()

    def copy(self):
        """Creates a copy of the tensor that does not point to the original"""
        """Never use A=B in python as modifying A will modify B"""
        return Tensor(data=self.data.copy(), labels=self.labels.copy())

    def move_index(self, label, position):
        """Change the order of the indices by moving the first index with label to position, 
        possibly shifting other indices forward or back in the process. """
        index = self.labels.index(label)

        #Move label in list
        self.labels.pop(index)
        self.labels.insert(position, label)

        #To roll axis of self.data
        #Not 100% sure why, but need to add 1 when rolling an axis backward
        if position <= index:
            self.data=np.rollaxis(self.data,index,position)
        else:
            self.data=np.rollaxis(self.data,index,position+1)

    def conjugate(self):
        self.data=self.data.conjugate()

    def inv(self):
        self.data=np.linalg.inv(self.data)

    def add_suffix_to_labels(self,suffix):
        """Warning: by changing the labels, e.g. with this method, 
        the MPS will no longer be in the correct form for various MPS functions."""
        new_labels=[]
        for label in self.labels:
            new_labels.append(label+suffix)
        self.labels=new_labels

    def add_dummy_index(self, label, position=0):
        """Add an additional index to the tensor with dimension 1, and label specified by the index "label"
        The position argument specifies where the index will be inserted. """
        #Will insert an axis of length 1 in the first position
        self.data=self.data[np.newaxis,:]
        self.labels.insert(0, label)
        self.move_index(label, position)

    def remove_dummy_index(self, label):
        """Remove the first dummy index (that is, an index of dimension 1) 
        with specified label."""
        if not self.index_dimension('label' == 1):
            # index not a dummy index
            raise ValueError("Index specified is not a dummy index.")
        self.move_index(label, 0)
        self.labels=self.labels[1:]
        self.data=self.data[0]

    def remove_all_dummy_indices(self, labels=None):
        """Removes all dummy indices (i.e. indices with dimension 1)
        which have labels specified by the labels argument. None
        for the labels argument implies all labels."""
        orig_shape=self.shape
        for i,x in enumerate(self.labels):
            if labels!= None:
                if x in labels and orig_shape[i]==1:
                    self.move_index(x, 0)
                    self.labels=self.labels[1:]
                    self.data=self.data[0]
            elif orig_shape[i]==1:
                self.move_index(x, 0)
                self.labels=self.labels[1:]
                self.data=self.data[0]

    def index_dimension(self, label):
        """Will return the dimension of the first index with label=label"""
        index = self.labels.index(label)
        return self.data.shape[index]

    def to_matrix(self, output_labels):
        """
        Convert tensor to a matrix regarding output_labels as output 
        (row index) and the remaining indices as input (column index).
        """
        return tensor_to_matrix(self, output_labels)

    @property
    def shape(self):
        return self.data.shape

def contract(tensor1, tensor2, label_list1, label_list2, index_list1=None, index_list2=None):
    """Contract two different tensors according to the specified labels"""
    """Indices to contract are specified by label_list1 and label_list2"""
    """Will find all the indices of tensor1 data with label label_list1[0],"""
    """then append indices with label given by "label_list1[1] etc."""

    #If the input label_list is not a list, convert to list with one entry
    if not isinstance(label_list1, list):
        label_list1=[label_list1]
    if not isinstance(label_list2, list):
        label_list2=[label_list2]

    tensor1_indices=[]
    for label in label_list1:
        #Append all indices to tensor1_indices with label
        tensor1_indices.extend([i for i,x in enumerate(tensor1.labels) if x==label])

    tensor2_indices=[]
    for label in label_list2:
        #Append all indices to tensor1_indices with label
        tensor2_indices.extend([i for i,x in enumerate(tensor2.labels) if x==label])
        
    #Replace the index -1 with the len(tensor1_indeces), to refer to the last element in the list
    if index_list1 != None:
        index_list1=[x if x!=-1 else len(tensor1_indices)-1 for x in index_list1]
    if index_list2 != None:
        index_list2=[x if x!=-1 else len(tensor2_indices)-1 for x in index_list2]
   
    #Select some subset or permutation of these indices if specified
    #If no list is specified, contract all indices with the specified labels
    #If an empty list is specified, no indices will be contracted
    if index_list1 != None:
        tensor1_indices=[j for i,j in enumerate(tensor1_indices) if i in index_list1]
    if index_list2 != None:
        tensor2_indices=[j for i,j in enumerate(tensor2_indices) if i in index_list2]

    
    #Contract the two tensors
    C=Tensor(np.tensordot(tensor1.data, tensor2.data, (tensor1_indices, tensor2_indices)))
    #The following removes the contracted indices from the list of labels 
    #and concatenates them
    new_tensor1_labels=[i for j,i in enumerate(tensor1.labels) 
            if j not in tensor1_indices]
    new_tensor2_labels=[i for j,i in enumerate(tensor2.labels) 
            if j not in tensor2_indices]
    C.labels=new_tensor1_labels+new_tensor2_labels

    return C

def tensor_product(tensor1, tensor2):
    """Take tensor product of two tensors without contracting any indices"""
    return contract(tensor1, tensor2, [], [])

def tensor_to_matrix(tensor, output_labels):
    """
    Convert a tensor to a matrix regarding output_labels as output (row index)
    and the remaining indices as input (column index).
    """
    t = tensor.copy()
    # Move labels in output_labels first and reshape accordingly
    total_output_dimension=1
    for i,label in enumerate(output_labels):
        t.move_index(label, i)
        total_output_dimension*=t.data.shape[i]

    total_input_dimension=int(np.product(t.data.shape)/total_output_dimension)
    return np.reshape(t.data,(total_output_dimension, total_input_dimension))

def matrix_to_tensor(matrix, output_dims, input_dims, output_labels,
        input_labels):
    """
    Convert a matrix to a tensor. The row index is divided into indices
    with dimensions and labels specified in output_dims and output_labels,
    respectively, and similarly the column index is divided into indices as
    specified by input_dims and input_labels.
    """
    return Tensor(np.reshape(matrix, tuple(output_dims)+tuple(input_dims)), output_labels+input_labels)

def tensor_svd(tensor, input_labels):
    """Compute the singular value decomposition of the matrix obtained from the tensor by regarding
    input_labels as input and the remaining indices as the output"""
    t=tensor.copy()

    #Move labels in input_labels to the beginning of list, and reshape data accordingly
    total_input_dimension=1
    for i,label in enumerate(input_labels):
        t.move_index(label, i)
        total_input_dimension*=t.data.shape[i]

    output_labels=[x for x in t.labels if x not in input_labels]

    old_shape=t.data.shape
    total_output_dimension=int(np.product(t.data.shape)/total_input_dimension)
    data_matrix=np.reshape(t.data,(total_input_dimension, total_output_dimension))

    u,s,v=np.linalg.svd(data_matrix, full_matrices=False)

    #Define tensors according to svd 
    n_input_indices=len(input_labels)

    #New shape original index labels as well as svd index
    U_shape=list(old_shape[0:n_input_indices])
    U_shape.append(u.shape[1])
    U=Tensor(data=np.reshape(u, U_shape), labels=input_labels+["svd_in"])
    V_shape=list(old_shape)[n_input_indices:]
    V_shape.insert(0,v.shape[0])
    V=Tensor(data=np.reshape(v, V_shape), labels=["svd_out"]+output_labels)

    S=Tensor(data=np.diag(s), labels=["svd_out", "svd_in"])

    return U, S, V

def truncated_svd(tensor, input_labels, chi, threshold=10**-15, absorb_singular_values="right"):
    """Will perform svd of a tensor, as in tensor_svd, and provide approximate decomposition by truncating
    all but the largest k singular values then absorbing S into U, V or both. Truncation is performed  
    by specifying the parameter chi (number of singular values to keep)."""

    U,S,V=tensor_svd(tensor, input_labels)
    
    #Slice columns from U, rows from V
    U.data=U.data[... , 0:chi]
    V.data=V.data[0:chi,:]
    V.data=V.data[0:chi,...]

    truncated_evals=np.diag(S.data)[chi:]
    S.data=S.data[0:chi, 0:chi]

    #Absorb singular values S into either V or U
    #or take the square root of S and absorb into both (default)
    if absorb_singular_values=="left":
        U_new=contract(U, S, ["svd_in"], ["svd_out"])
        V_new=V
    elif absorb_singular_values=="right":
        V_new=contract(S, V, ["svd_in"], ["svd_out"])
        U_new=U
    else:
        sqrtS=S.copy()
        sqrtS.data=np.sqrt(sqrtS.data)
        U_new=contract(U, sqrtS, ["svd_in"], ["svd_out"])
        V_new=contract(sqrtS, V, ["svd_in"], ["svd_out"])
    return U_new, V_new, truncated_evals


