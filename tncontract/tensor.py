from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

__all__ = ['Tensor', 'contract', 'distance', 'matrix_to_tensor',
           'tensor_to_matrix', 'random_tensor', 'tensor_product', 'tensor_svd',
           'truncated_svd', 'zeros_tensor']

import copy
import warnings
import numpy as np
import scipy as sp

from tncontract import label as lbl


class Tensor():
    """
    A single tensor containing a numpy array and a list of labels.
    
    A tensor, for our purposes, is a multi-index array of complex numbers.
    Tensors can be contracted with other tensors to form new tensors. A basic
    contraction requires specification of two indices, either from the same
    tensor of from a pair of different tensors. 

    The `Tensor` class contains a multi-dimensional ndarray (stored in the
    `data` attribute), and list of labels (stored in the `labels` attribute)
    where each label in `labels` corresponds to an axis of `data`.  Labels are
    assumed to be strings. The order of the labels in `labels` should agree
    with the order of the axes in `data` such that the first label corresponds
    to the first axis and so on, and the length of labels should equal the
    number of axes in `data`. Functions and methods that act on Tensor objects
    should update `labels` whenever `data` is changed and vice versa, such that
    a given label always corresponds to the same axis. For instance, if two
    axes are swapped in `data` the corresponding labels should be swapped in
    `labels`. The exceptions being when labels are explicitly changed e.g. when
    using the `replace_label` method. 

    Attributes
    ----------

    data : ndarray
        A multi-dimensional array of numbers. 
    labels : list
        A list of strings which label the axes of data. `label[i]` is the label
        for the `i`-1th axis of data.
    """

    def __init__(self, data, labels=None, base_label="i"):
        labels = [] if labels is None else labels
        self.data = np.array(data)

        if len(labels) == 0:
            self.assign_labels(base_label=base_label)
        else:
            self.labels = labels

    def __repr__(self):
        return "Tensor(data=%r, labels=%r)" % (self.data, self.labels)

    def __str__(self):
        array_str = str(self.data)
        lines = array_str.splitlines()
        if len(lines) > 20:
            lines = lines[:20] + ["...",
                                  "Printed output of large array was truncated.\nString "
                                  "representation of full data array returned by "
                                  "tensor.data.__str__()."]
            array_str = "\n".join(lines)

        # Specify how index information is printed
        lines = []
        for i, label in enumerate(self.labels):
            lines.append("   " + str(i) + ". (dim=" + str(self.shape[i]) + ") " +
                         str(label) + "\n")
        indices_str = "".join(lines)

        return ("Tensor object: \n" +
                "Data type: " + str(self.data.dtype) + "\n"
                                                       "Number of indices: " + str(len(self.data.shape)) + "\n"
                                                                                                           "\nIndex labels:\n" + indices_str +
                # "shape = " + str(self.shape) +
                # ", labels = " + str(self.labels) + "\n" +(
                "\nTensor data = \n" + array_str)

    def __eq__(self, other):
        if isinstance(other, Tensor):
            return (np.array_equal(self.data, other.data)
                    and self.labels == other.labels)
        else:
            return False

    def __neq__(self, other):
        return not self.__eq__(other)

    def __mul__(self, other):
        """
        Multiplcation with Tensor on left (i.e. `self*other`).
        Returns a copy of `self` with `data` attribute multiplied by
        `other`.
        """
        try:
            out = self.copy()
            out.data = out.data * other
            return out
        except TypeError:
            raise TypeError("unsupported operand type(s) *: for '"
                            + self.__class__.__name__ + "' and '"
                            + other.__class__.__name__ + "'")

    def __rmul__(self, other):
        """
        Multiplcation with Tensor on right (i.e. `other*self`).
        Returns a copy of `self` with `data` attribute multiplied by
        `other`.
        """
        try:
            out = self.copy()
            out.data = other * out.data
            return out
        except TypeError:
            raise TypeError("unsupported operand type(s) *: for '"
                            + self.__class__.__name__ + "' and '"
                            + other.__class__.__name__ + "'")

    def __add__(self, other):
        """
        Return a new tensor with the same labels as Tensor objects `self` (and `other`)  
        with a data array equal to the sum of the data arrays of `self` and 
        `other`. Requires that the labels of `self` and `other` are the same and,
        that their corresponding indices have the same dimension. 
        """
        try: 
            a=self.copy()
            b=other.copy()
            a.consolidate_indices()
            b.consolidate_indices()
            return Tensor(a.data+b.data, labels=a.labels)
        except:
            raise TypeError("Can only add together tensors with the same"+
                    " indices: labels and dimensions of each index must match.")

    def __getitem__(self, *args):
        """Used to allow convenient shorthand for defining tensor
        contraction."""
        return ToContract(self, *args)

    # Define functions for getting and setting labels
    def get_labels(self):
        return self._labels

    def set_labels(self, labels):
        if len(labels) == len(self.data.shape):
            self._labels = list(labels)
        else:
            raise ValueError("Labels do not match shape of data.")

    labels = property(get_labels, set_labels)

    def assign_labels(self, base_label="i"):
        """Assign labels to all of the indices of `Tensor`. The i-th axis will
        be assigned the label `base_label`+"i-1"."""
        self.labels = [base_label + str(i) for i in range(len(self.data.shape))]

    def replace_label(self, old_labels, new_labels):
        """
        Takes two lists old_labels, new_labels as arguments. If a label in 
        self.labels is in old_labels, it is replaced with the respective label 
        In new_labels.
        """

        # If either argument is not a list, convert to list with single entry
        if not isinstance(old_labels, list):
            old_labels = [old_labels]
        if not isinstance(new_labels, list):
            new_labels = [new_labels]

        for i, label in enumerate(self.labels):
            if label in old_labels:
                self.labels[i] = new_labels[old_labels.index(label)]

    def prime_label(self, labels=None):
        """
        Add a prime (') to all `label` in `labels`

        Parameters
        ----------
        labels : str or list of str, optional
            Labels to prime. If None all labels of the tensor will be primed.

        See also
        -------
        unprime_label
        """
        if labels is None:
            labels = self.labels
        elif not isinstance(labels, list):
            labels = [labels]
        for i, label in enumerate(self.labels):
            for noprime in labels:
                if lbl.noprime_label(label) == noprime:
                    self.labels[i] = lbl.prime_label(self.labels[i])

    def unprime_label(self, labels=None):
        """
        Remove the last prime (') from all `label` in `labels`

        Parameters
        ----------
        labels : str or list of str, optional
            Labels to unprime. If None all labels of the tensor will be 
            unprimed.

        Examples
        --------
        >>> t = Tensor(np.array([1,0]), labels=["idx"])
        >>> t.prime_label("idx")
        >>> print(t)
        Tensor object: shape = (2,), labels = ["idx'"]
        >>> t.prime_label("idx")
        >>> print(t)
        Tensor object: shape = (2,), labels = ["idx''"]
        >>> t.unprime_label("idx")
        >>> print(t)
        Tensor object: shape = (2,), labels = ["idx'"]
        >>> t.unprime_label("idx")
        >>> print(t)
        Tensor object: shape = (2,), labels = ["idx"]
        """
        if labels is None:
            labels = self.labels
        elif not isinstance(labels, list):
            labels = [labels]
        for i, label in enumerate(self.labels):
            for noprime in labels:
                if lbl.noprime_label(label) == noprime:
                    self.labels[i] = lbl.unprime_label(self.labels[i])

    def fuse_indices(self, indices_to_fuse, new_label,
                     preserve_relative_order=True):
        """Fuse multiple indices into a single index. If
        `preserve_relative_order` is True, the relative order of the fused
        indices will be preserved. Otherwise, the order will follow the order
        in the `indices_to_fuse` argument.
        
        Examples
        --------
        In this example we fuse a pair of indices to a single index, 
        then split them again. We start with a random rank-5 tensor.
        >>> t=random_tensor(2,3,4,5,6, labels=["a","b","c","d","a"])
        >>> t_orig=t.copy()

        Fuse indices "b" and "d" to a new index called "new_index".
        >>> t.fuse_indices(["b","d"], "new_index")
        >>> print(t)
        Tensor object: 
        Data type: float64
        Number of indices: 4
        Index labels:
        0. (dim=15) new_index
        1. (dim=2) a
        2. (dim=4) c
        3. (dim=6) a

        Split the "new_index" index into two indices "b" and "d" with 
        dimensions 3 and 5 respectively. 
        >>> t.split_index("new_index", (3,5), ["b","d"])
        >>> print(t)
        Tensor object: 
        Data type: float64
        Number of indices: 5
        Index labels:
        0. (dim=3) b
        1. (dim=5) d
        2. (dim=2) a
        3. (dim=4) c
        4. (dim=6) a

        The resulting tensor is identical to the original (up to a reordering 
        of the indices). 
        >>> distance(t, t_orig)
        0.0
        """
        indices = [i for i, x in enumerate(self.labels) if x in indices_to_fuse]
        # Move the indices to fuse to position zero
        self.move_indices(indices_to_fuse, 0,
                          preserve_relative_order=preserve_relative_order)
        # Compute the total dimension of the new index
        total_dim = 1
        for i, x in enumerate(self.labels):
            if x in indices_to_fuse:
                total_dim *= self.data.shape[i]
            else:
                new_labels = [new_label] + self.labels[i:]
                new_shape = (total_dim,) + self.data.shape[i:]
                break

        self.data = np.reshape(self.data, new_shape)
        self.labels = new_labels

    def split_index(self, label, new_dims, new_labels):
        """
        Split a single index into multiple indices.

        See also
        --------
        fuse_indices
        """
        if len(new_dims) != len(new_labels):
            raise ValueError("Length of new_dims must equal length of "
                             "new_labels")

        new_dims = tuple(new_dims)
        i = self.labels.index(label)
        new_shape = self.data.shape[:i] + new_dims + self.data.shape[i + 1:]
        new_labels = self.labels[:i] + new_labels + self.labels[i + 1:]

        self.data = np.reshape(self.data, new_shape)
        self.labels = new_labels

    def contract_internal(self, label1, label2, index1=0, index2=0):
        """By default will contract the first index with label1 with the 
        first index with label2. index1 and index2 can be specified to contract
        indices that are not the first with the specified label."""

        label1_indices = [i for i, x in enumerate(self.labels) if x == label1]
        label2_indices = [i for i, x in enumerate(self.labels) if x == label2]

        index_to_contract1 = label1_indices[index1]
        index_to_contract2 = label2_indices[index2]

        self.data = np.trace(self.data, axis1=index_to_contract1, axis2=
        index_to_contract2)

        # The following removes the contracted indices from the list of labels
        self.labels = [label for j, label in enumerate(self.labels)
                       if j not in [index_to_contract1, index_to_contract2]]

    # aliases for contract_internal
    trace = contract_internal
    tr = contract_internal

    def consolidate_indices(self, labels=[]):
        """Combines all indices with the same label into a single label.
        If `labels` keyword argument is non-empty, only labels in `labels` will 
        be consolidated. Puts labels in alphabetical order (and reshapes data 
        accordingly) if `labels` is empty.
        """
        labels_unique = sorted(set(self.labels))
        if len(labels) !=0:
            #If `labels` is set, only consolidate indices in `labels`
            labels_unique=[x for x in labels_unique if x in labels]
        for p, label in enumerate(labels_unique):
            indices = [i for i, j in enumerate(self.labels) if j == label]
            # Put all of these indices together
            for k, q in enumerate(indices):
                self.data = np.rollaxis(self.data, q, p + k)
            # Total dimension of all indices with label
            total_dim = self.data.shape[p]
            for r in range(1, len(indices)):
                total_dim = total_dim * self.data.shape[p + r]
            # New shape after consolidating all indices with label into
            # one at position p
            new_shape = (list(self.data.shape[0:p]) + [total_dim] +
                         list(self.data.shape[p + len(indices):]))
            self.data = np.reshape(self.data, tuple(new_shape))

            # Update self.labels
            # Remove all instances of label from self.labels
            new_labels = [x for x in self.labels if x != label]
            # Reinsert label at position p
            new_labels.insert(p, label)
            self.labels = new_labels

    def sort_labels(self):
        self.consolidate_indices()

    def copy(self):
        """Creates a copy of the tensor that does not point to the original"""
        """Never use A=B in python as modifying A will modify B"""
        return Tensor(data=self.data.copy(), labels=copy.copy(self.labels))

    def move_index(self, label, position):
        """Change the order of the indices by moving the first index with label
        `label` to position `position`, possibly shifting other indices forward
        or back in the process. """
        index = self.labels.index(label)
        # Move label in list
        self.labels.pop(index)
        self.labels.insert(position, label)

        # To roll axis of self.data
        # Not 100% sure why, but need to add 1 when rolling an axis backward
        if position <= index:
            self.data = np.rollaxis(self.data, index, position)
        else:
            self.data = np.rollaxis(self.data, index, position + 1)

    def move_indices(self, labels, position,
                     preserve_relative_order=False):
        """Move indices with labels in `labels` to consecutive positions
        starting at `position`. If `preserve_relative_order`==True, the
        relative order of the moved indices will be identical to their order in
        the original tensor. If not, the relative order will be determined by
        the order in the `labels` argument.
        
        Examples
        --------
        First initialise a random tensor.
        >>> from tncontract import random_tensor
        >>> t=random_tensor(2,3,4,5,6, labels=["a", "b", "c", "b", "d"])

        Now we move the indices labelled "d", "b" and "c" to position 0 (i.e.
        the beginning). When preserve_relative_order is True, the relative 
        order of these indices is identical to the original tensor.
        >>> t.move_indices(["d","b","c"], 0, preserve_relative_order=True)
        >>> print(t)
        Tensor object: 
        Data type: float64
        Number of indices: 5
        Index labels:
           0. (dim=3) b
           1. (dim=4) c
           2. (dim=5) b
           3. (dim=6) d
           4. (dim=2) a

        If, on the other hand, preserve_relative_order is False, the order of 
        the indices is determined by the order in which they appear in the 
        `labels` argument of `move_indices`. In this case, "d" comes first 
        then the "b" indices then "c". 
        >>> t=random_tensor(2,3,4,5,6, labels=["a", "b", "c", "b", "d"])
        >>> t.move_indices(["d","b","c"], 0, preserve_relative_order=False)
        >>> print(t)
        Tensor object: 
        Data type: float64
        Number of indices: 5
        Index labels:
           0. (dim=6) d
           1. (dim=3) b
           2. (dim=5) b
           3. (dim=4) c
           4. (dim=2) a
  
        """

        if not isinstance(labels, list):
            labels = [labels]

        if preserve_relative_order:
            orig_labels = self.labels.copy()
            n_indices_to_move = 0
            for label in orig_labels:
                if label in labels:
                    # Move label to end of list
                    self.move_index(label, len(self.labels) - 1)
                    n_indices_to_move += 1
        else:
            # Remove duplicates
            unique_labels = []
            for label in labels:
                if label not in unique_labels:
                    unique_labels.append(label)
            labels = unique_labels

            n_indices_to_move = 0
            for label in labels:
                for i in range(self.labels.count(label)):
                    # Move label to end of list
                    self.move_index(label, len(self.labels) - 1)
                    n_indices_to_move += 1

        if position + n_indices_to_move > len(self.labels):
            raise ValueError("Specified position too far right.")

        # All indices to move are at the end of the array
        # Now put put them in desired place
        for j in range(n_indices_to_move):
            old_index = len(self.labels) - n_indices_to_move + j
            label = self.labels[old_index]
            # Move label in list
            self.labels.pop(old_index)
            self.labels.insert(position + j, label)
            # Reshape accordingly
            self.data = np.rollaxis(self.data, old_index, position + j)

    def conjugate(self):
        self.data = self.data.conjugate()

    def inv(self):
        self.data = np.linalg.inv(self.data)

    def add_suffix_to_labels(self, suffix):
        """Warning: by changing the labels, e.g. with this method, 
        the MPS will no longer be in the correct form for various MPS functions
        ."""
        new_labels = []
        for label in self.labels:
            new_labels.append(label + suffix)
        self.labels = new_labels

    def add_dummy_index(self, label, position=0):
        """Add an additional index to the tensor with dimension 1, and label 
        specified by the index "label". The position argument specifies where 
        the index will be inserted. """
        # Will insert an axis of length 1 in the first position
        self.data = self.data[np.newaxis, :]
        self.labels.insert(0, label)
        self.move_index(label, position)

    def remove_all_dummy_indices(self, labels=None):
        """Removes all dummy indices (i.e. indices with dimension 1)
        which have labels specified by the labels argument. None
        for the labels argument implies all labels."""
        orig_shape = self.shape
        for i, x in enumerate(self.labels):
            if labels != None:
                if x in labels and orig_shape[i] == 1:
                    self.move_index(x, 0)
                    self.data = self.data[0]
                    self.labels = self.labels[1:]
            elif orig_shape[i] == 1:
                self.move_index(x, 0)
                self.data = self.data[0]
                self.labels = self.labels[1:]

    def index_dimension(self, label):
        """Will return the dimension of the first index with label=label"""
        index = self.labels.index(label)
        return self.data.shape[index]

    def to_matrix(self, row_labels):
        """
        Convert tensor to a matrix regarding row_labels as row index 
        (output) and the remaining indices as column index (input).
        """
        return tensor_to_matrix(self, row_labels)

    def pad_index(self, label, inc, before=False):
        """
        Increase the dimension of first index with `label` by `inc` by padding
        with zeros.

        By default zeros are appended after the last edge of the axis in
        question, e.g., [1,2,3] -> [1,2,3,0..0]. If `before=True` the zeros
        will be padded before the first edge of the index instead,
        e.g., [1,2,3] -> [0,..,0,1,2,3].

        See also
        --------
        numpy.pad
        """
        if before:
            npad = ((inc, 0),)
        else:
            npad = ((0, inc),)
        index = self.labels.index(label)
        npad = ((0, 0),) * (index) + npad + ((0, 0),) * (self.rank - index - 1)
        self.data = np.pad(self.data, npad, mode='constant', constant_values=0)

    def contract(self, *args, **kwargs):
        """
        A method that calls the function `contract`, passing `self` as the
        first argument.
        
        See also
        --------
        contract (function)

        """
        t = contract(self, *args, **kwargs)
        self.data = t.data
        self.labels = t.labels

    @property
    def shape(self):
        return self.data.shape

    @property
    def rank(self):
        return len(self.shape)

    def norm(self):
        """Return the frobenius norm of the tensor, equivalent to taking the
        sum of absolute values squared of every element. """
        return np.linalg.norm(self.data)


class ToContract():
    """A simple class that contains a Tensor and a list of indices (labels) of
    that tensor which are to be contracted with another tensor. Used in
    __mul__, __rmul__ for convenient tensor contraction."""

    def __init__(self, tensor, labels):
        self.tensor = tensor
        self.labels = labels

    def __mul__(self, other):
        # If label argument is not a tuple, simply use that as the argument to
        # contract function. Otherwise convert to a list.
        if not isinstance(self.labels, tuple):
            labels1 = self.labels
        else:
            labels1 = list(self.labels)
        if not isinstance(other.labels, tuple):
            labels2 = other.labels
        else:
            labels2 = list(other.labels)
        return contract(self.tensor, other.tensor, labels1, labels2)

        # Tensor constructors


def random_tensor(*args, **kwargs):
    """Construct a random tensor of a given shape. Entries are generated using
    `numpy.random.rand`."""
    labels = kwargs.pop("labels", [])
    base_label = kwargs.pop("base_label", "i")
    return Tensor(np.random.rand(*args), labels=labels, base_label=base_label)


def zeros_tensor(*args, **kwargs):
    """Construct a tensor of a given shape with every entry equal to zero."""
    labels = kwargs.pop("labels", [])
    dtype = kwargs.pop("dtype", np.float)
    base_label = kwargs.pop("base_label", "i")
    return Tensor(np.zeros(*args, dtype=dtype), labels=labels,
                  base_label=base_label)


def contract(tensor1, tensor2, labels1, labels2, index_slice1=None,
             index_slice2=None):
    """
    Contract the indices of `tensor1` specified in `labels1` with the indices
    of `tensor2` specified in `labels2`. 
    
    This is an intuitive wrapper for numpy's `tensordot` function.  A pairwise
    tensor contraction is specified by a pair of tensors `tensor1` and
    `tensor2`, a set of index labels `labels1` from `tensor1`, and a set of
    index labels `labels2` from `tensor2`. All indices of `tensor1` with label
    in `labels1` are fused (preserving order) into a single label, and likewise
    for `tensor2`, then these two fused indices are contracted. 

    Parameters
    ----------
    tensor1, tensor2 : Tensor
        The two tensors to be contracted.

    labels1, labels2 : str or list
        The indices of `tensor1` and `tensor2` to be contracted. Can either be
        a single label, or a list of labels. 

    Examples
    --------
    Define a random 2x2 tensor with index labels "spam" and "eggs" and a random
    2x3x2x4 tensor with index labels 'i0', 'i1', etc. 

    >>> A = random_tensor(2, 2, labels = ["spam", "eggs"])
    >>> B = random_tensor(2, 3, 2, 4)
    >>> print(B)
    Tensor object: shape = (2, 3, 2, 4), labels = ['i0', 'i1', 'i2', 'i3']
    
    Contract the "spam" index of tensor A with the "i2" index of tensor B.
    >>> C = contract(A, B, "spam", "i2")
    >>> print(C)
    Tensor object: shape = (2, 2, 3, 4), labels = ['eggs', 'i0', 'i1', 'i3']

    Contract the "spam" index of tensor A with the "i0" index of tensor B and
    also contract the "eggs" index of tensor A with the "i2" index of tensor B.

    >>> D = contract(A, B, ["spam", "eggs"], ["i0", "i2"])
    >>> print(D)
    Tensor object: shape = (3, 4), labels = ['i1', 'i3']

    Note that the following shorthand can be used to perform the same operation
    described above.
    >>> D = A["spam", "eggs"]*B["i0", "i2"]
    >>> print(D)
    Tensor object: shape = (3, 4), labels = ['i1', 'i3']

    Returns
    -------
    C : Tensor
        The result of the tensor contraction. Regarding the `data` and `labels`
        attributes of this tensor, `C` will have all of the uncontracted
        indices of `tensor1` and `tensor2`, with the indices of `tensor1`
        always coming before those of `tensor2`, and their internal order
        preserved. 

    """

    # If the input labels is not a list, convert to list with one entry
    if not isinstance(labels1, list):
        labels1 = [labels1]
    if not isinstance(labels2, list):
        labels2 = [labels2]

    tensor1_indices = []
    for label in labels1:
        # Append all indices to tensor1_indices with label
        tensor1_indices.extend([i for i, x in enumerate(tensor1.labels)
                                if x == label])

    tensor2_indices = []
    for label in labels2:
        # Append all indices to tensor1_indices with label
        tensor2_indices.extend([i for i, x in enumerate(tensor2.labels)
                                if x == label])

    # Replace the index -1 with the len(tensor1_indeces),
    # to refer to the last element in the list
    if index_slice1 is not None:
        index_slice1 = [x if x != -1 else len(tensor1_indices) - 1 for x
                        in index_slice1]
    if index_slice2 is not None:
        index_slice2 = [x if x != -1 else len(tensor2_indices) - 1 for x
                        in index_slice2]

    # Select some subset or permutation of these indices if specified
    # If no list is specified, contract all indices with the specified labels
    # If an empty list is specified, no indices will be contracted
    if index_slice1 is not None:
        tensor1_indices = [j for i, j in enumerate(tensor1_indices)
                           if i in index_slice1]
    if index_slice2 is not None:
        tensor2_indices = [j for i, j in enumerate(tensor2_indices)
                           if i in index_slice2]

    # Contract the two tensors
    try:
        C = Tensor(np.tensordot(tensor1.data, tensor2.data,
                                (tensor1_indices, tensor2_indices)))
    except ValueError as e:
        # Print more useful info in case of ValueError.
        # Check if number of indices are equal
        if not len(tensor1_indices) == len(tensor2_indices):
            raise ValueError('Number of indices in contraction '
                    'does not match.')
        # Check if indices have equal dimensions
        for i in range(len(tensor1_indices)):
            d1 = tensor1.data.shape[tensor1_indices[i]]
            d2 = tensor2.data.shape[tensor2_indices[i]]
            if d1 != d2:
                raise ValueError(labels1[i] + ' with dim=' + str(d1) +
                                       ' does not match ' + labels2[i] +
                                       ' with dim=' + str(d2))
        # Check if indices exist
        for i in range(len(labels1)):
            if not labels1[i] in tensor1.labels:
                raise ValueError(labels1[i] + 
                        ' not in list of labels for tensor1')
            if not labels2[i] in tensor2.labels:
                raise ValueError(labels2[i] + 
                        ' not in list of labels for tensor2')
        # Re-raise exception
        raise e

    # The following removes the contracted indices from the list of labels
    # and concatenates them
    new_tensor1_labels = [i for j, i in enumerate(tensor1.labels)
                          if j not in tensor1_indices]
    new_tensor2_labels = [i for j, i in enumerate(tensor2.labels)
                          if j not in tensor2_indices]
    C.labels = new_tensor1_labels + new_tensor2_labels

    return C


def tensor_product(tensor1, tensor2):
    """Take tensor product of two tensors without contracting any indices"""
    return contract(tensor1, tensor2, [], [])


def distance(tensor1, tensor2):
    """
    Will compute the Frobenius distance between two tensors, specifically the
    distance between the flattened data arrays in the 2 norm. 

    Notes
    -----
    The `consolidate_indices` method will be run first on copies of the tensors 
    to put the data in the same shape. `tensor1` and `tensor2` should have the
    same labels, and same shape after applying `consolidate_indices`, otherwise
    an error will be raised.
    """
    t1 = tensor1.copy()
    t2 = tensor2.copy()
    t1.consolidate_indices()
    t2.consolidate_indices()

    if t1.labels == t2.labels:
        return np.linalg.norm(t1.data - t2.data)
    else:
        raise ValueError("Input tensors must have the same labels.")


def tensor_to_matrix(tensor, row_labels):
    """
    Convert a tensor to a matrix regarding row_labels as row index (output)
    and the remaining indices as column index (input).
    """
    t = tensor.copy()
    # Move labels in row_labels first and reshape accordingly
    total_row_dimension = 1
    for i, label in enumerate(row_labels):
        t.move_index(label, i)
        total_row_dimension *= t.data.shape[i]

    total_column_dimension = int(np.product(t.data.shape) / total_row_dimension)
    return np.reshape(t.data, (total_row_dimension, total_column_dimension))


def matrix_to_tensor(matrix, shape, labels=None):
    """
    Convert a matrix to a tensor by reshaping to `shape` and giving labels
    specifid by `labels`
    """
    labels = [] if labels is None else labels
    return Tensor(np.reshape(np.array(matrix), shape), labels)


def tensor_svd(tensor, row_labels, svd_label="svd_",
               absorb_singular_values=None):
    """
    Compute the singular value decomposition of `tensor` after reshaping it 
    into a matrix.

    Indices with labels in `row_labels` are fused to form a single index 
    corresponding to the rows of the matrix (typically the left index of a
    matrix). The remaining indices are fused to form the column indices. An SVD
    is performed on this matrix, yielding three matrices u, s, v, where u and
    v are unitary and s is diagonal with positive entries. These three
    matrices are then reshaped into tensors U, S, and V. Contracting U, S and V
    together along the indices labelled by `svd_label` will yeild the original
    input `tensor`.

    Parameters
    ----------
    tensor : Tensor
        The tensor on which the SVD will be performed.
    row_labels : list
        List of labels specifying the indices of `tensor` which will form the
        rows of the matrix on which the SVD will be performed.
    svd_label : str
        Base label for the indices that are contracted with `S`, the tensor of
        singular values. 
    absorb_singular_values : str, optional
        If "left", "right" or "both", singular values will be absorbed into
        U, V, or the square root into both, respectively, and only U and V
        are returned.

    Returns
    -------
    U : Tensor
        Tensor obtained by reshaping the matrix u obtained by SVD as described 
        above. Has indices labelled by `row_labels` corresponding to the
        indices labelled `row_labels` of `tensor` and has one index labelled 
        `svd_label`+"in" which connects to S.
    V : Tensor
        Tensor obtained by reshaping the matrix v obtained by SVD as described 
        above. Indices correspond to the indices of `tensor` that aren't in 
        `row_labels`. Has one index labelled  `svd_label`+"out" which connects
        to S.
    S : Tensor
        Tensor with data consisting of a diagonal matrix of singular values.
        Has two indices labelled `svd_label`+"out" and `svd_label`+"in" which
        are contracted with with the `svd_label`+"in" label of U and the
        `svd_label`+"out" of V respectively.

    Examples
    --------
    >>> a=random_tensor(2,3,4, labels = ["i0", "i1", "i2"])
    >>> U,S,V = tensor_svd(a, ["i0", "i2"])
    >>> print(U)
    Tensor object: shape = (2, 4, 3), labels = ['i0', 'i2', 'svd_in']
    >>> print(V)
    Tensor object: shape = (3, 3), labels = ['svd_out', 'i1']
    >>> print(S)
    Tensor object: shape = (3, 3), labels = ['svd_out', 'svd_in']
    
    Recombining the three tensors obtained from SVD, yeilds a tensor very close
    to the original.

    >>> temp=tn.contract(S, V, "svd_in", "svd_out")
    >>> b=tn.contract(U, temp, "svd_in", "svd_out")
    >>> tn.distance(a,b)
    1.922161284937472e-15
    """

    t = tensor.copy()

    # Move labels in row_labels to the beginning of list, and reshape data
    # accordingly
    total_input_dimension = 1
    for i, label in enumerate(row_labels):
        t.move_index(label, i)
        total_input_dimension *= t.data.shape[i]

    column_labels = [x for x in t.labels if x not in row_labels]

    old_shape = t.data.shape
    total_output_dimension = int(np.product(t.data.shape) / total_input_dimension)
    data_matrix = np.reshape(t.data, (total_input_dimension,
                                      total_output_dimension))

    try:
        u, s, v = np.linalg.svd(data_matrix, full_matrices=False)
    except (np.linalg.LinAlgError, ValueError):
        # Try with different lapack driver
        warnings.warn(('numpy.linalg.svd failed, trying scipy.linalg.svd with' +
                       ' lapack_driver="gesvd"'))
        try:
            u, s, v = sp.linalg.svd(data_matrix, full_matrices=False,
                                    lapack_driver='gesvd')
        except ValueError:
            # Check for inf's and nan's:
            print("tensor_svd failed. Matrix contains inf's: "
                  + str(np.isinf(data_matrix).any())
                  + ". Matrix contains nan's: "
                  + str(np.isnan(data_matrix).any()))
            raise  # re-raise the exception

    # New shape original index labels as well as svd index
    U_shape = list(old_shape[0:len(row_labels)])
    U_shape.append(u.shape[1])
    U = Tensor(data=np.reshape(u, U_shape), labels=row_labels + [svd_label + "in"])
    V_shape = list(old_shape)[len(row_labels):]
    V_shape.insert(0, v.shape[0])
    V = Tensor(data=np.reshape(v, V_shape),
               labels=[svd_label + "out"] + column_labels)

    S = Tensor(data=np.diag(s), labels=[svd_label + "out", svd_label + "in"])

    # Absorb singular values S into either V or U
    # or take the square root of S and absorb into both
    if absorb_singular_values == "left":
        U_new = contract(U, S, ["svd_in"], ["svd_out"])
        V_new = V
        return U_new, V_new
    elif absorb_singular_values == "right":
        V_new = contract(S, V, ["svd_in"], ["svd_out"])
        U_new = U
        return U_new, V_new
    elif absorb_singular_values == "both":
        sqrtS = S.copy()
        sqrtS.data = np.sqrt(sqrtS.data)
        U_new = contract(U, sqrtS, ["svd_in"], ["svd_out"])
        V_new = contract(sqrtS, V, ["svd_in"], ["svd_out"])
        return U_new, V_new
    else:
        return U, S, V


def tensor_qr(tensor, row_labels, qr_label="qr_"):
    """
    Compute the QR decomposition of `tensor` after reshaping it into a matrix.
    Indices with labels in `row_labels` are fused to form a single index
    corresponding to the rows of the matrix (typically the left index of a
    matrix). The remaining indices are fused to form the column index. A QR
    decomposition is performed on this matrix, yielding two matrices q,r, where
    q and is a rectangular matrix with orthonormal columns and r is upper
    triangular. These two matrices are then reshaped into tensors Q and R.
    Contracting Q and R along the indices labelled `qr_label` will yeild the
    original input tensor `tensor`.

    Parameters
    ----------
    tensor : Tensor
        The tensor on which the QR decomposition will be performed.
    row_labels : list
        List of labels specifying the indices of `tensor` which will form the
        rows of the matrix on which the QR will be performed.
    qr_label : str
        Base label for the indices that are contracted between `Q` and `R`.

    Returns
    -------
    Q : Tensor
        Tensor obtained by reshaping the matrix q obtained from QR
        decomposition.  Has indices labelled by `row_labels` corresponding to
        the indices labelled `row_labels` of `tensor` and has one index
        labelled `qr_label`+"in" which connects to `R`.
    R : Tensor
        Tensor obtained by reshaping the matrix r obtained by QR decomposition.
        Indices correspond to the indices of `tensor` that aren't in
        `row_labels`. Has one index labelled `qr_label`+"out" which connects
        to `Q`.

    Examples
    --------

    >>> from tncontract.tensor import *
    >>> t=random_tensor(2,3,4)
    >>> print(t)
    Tensor object: shape = (2, 3, 4), labels = ['i0', 'i1', 'i2']
    >>> Q,R = tensor_qr(t, ["i0", "i2"])
    >>> print(Q)
    Tensor object: shape = (2, 4, 3), labels = ['i0', 'i2', 'qr_in']
    >>> print(R)
    Tensor object: shape = (3, 3), labels = ['qr_out', 'i1']

    Recombining the two tensors obtained from `tensor_qr`, yeilds a tensor very
    close to the original

    >>> x = contract(Q, R, "qr_in", "qr_out")
    >>> print(x)
    Tensor object: shape = (2, 4, 3), labels = ['i0', 'i2', 'i1']
    >>> distance(x,t)
    9.7619164946377426e-16
    """
    t = tensor.copy()

    if not isinstance(row_labels, list):
        # If row_labels is not a list, convert to list with a single entry
        # "row_labels"
        row_labels = [row_labels]

    # Move labels in row_labels to the beginning of list, and reshape data
    # accordingly
    t.move_indices(row_labels, 0)

    # Compute the combined dimension of the row indices
    row_dimension = 1
    for i, label in enumerate(t.labels):
        if label not in row_labels:
            break
        row_dimension *= t.data.shape[i]

    column_labels = [x for x in t.labels if x not in row_labels]

    old_shape = t.data.shape
    total_output_dimension = int(np.product(t.data.shape) / row_dimension)
    data_matrix = np.reshape(t.data, (row_dimension,
                                      total_output_dimension))

    q, r = np.linalg.qr(data_matrix, mode="reduced")

    # New shape original index labels as well as svd index
    Q_shape = list(old_shape[0:len(row_labels)])
    Q_shape.append(q.shape[1])
    Q = Tensor(data=np.reshape(q, Q_shape), labels=row_labels + [qr_label + "in"])
    R_shape = list(old_shape)[len(row_labels):]
    R_shape.insert(0, r.shape[0])
    R = Tensor(data=np.reshape(r, R_shape), labels=[qr_label + "out"] +
                                                   column_labels)

    return Q, R


def tensor_lq(tensor, row_labels, lq_label="lq_"):
    """
    Compute the LQ decomposition of `tensor` after reshaping it into a matrix.
    Indices with labels in `row_labels` are fused to form a single index
    corresponding to the rows of the matrix (typically the left index of a
    matrix). The remaining indices are fused to form the column index. An LR
    decomposition is performed on this matrix, yielding two matrices l,q, where
    q and is a rectangular matrix with orthonormal rows and l is upper
    triangular. These two matrices are then reshaped into tensors L and Q.
    Contracting L and Q along the indices labelled `lq_label` will yeild the
    original input `tensor`. Note that the LQ decomposition is actually
    identical to the QR decomposition after a relabelling of indices. 

    Parameters
    ----------
    tensor : Tensor
        The tensor on which the LQ decomposition will be performed.
    row_labels : list
        List of labels specifying the indices of `tensor` which will form the
        rows of the matrix on which the LQ decomposition will be performed.
    lq_label : str
        Base label for the indices that are contracted between `L` and `Q`.

    Returns
    -------
    Q : Tensor
        Tensor obtained by reshaping the matrix q obtained by LQ decomposition.
        Indices correspond to the indices of `tensor` that aren't in
        `row_labels`. Has one index labelled `lq_label`+"out" which connects
        to `L`.
    L : Tensor
        Tensor obtained by reshaping the matrix l obtained from LQ
        decomposition.  Has indices labelled by `row_labels` corresponding to
        the indices labelled `row_labels` of `tensor` and has one index
        labelled `lq_label`+"in" which connects to `Q`.

    See Also
    --------
    tensor_qr
    
    """

    col_labels = [x for x in tensor.labels if x not in row_labels]

    temp_label = lbl.unique_label()
    # Note the LQ is essentially equivalent to a QR decomposition, only labels
    # are renamed
    Q, L = tensor_qr(tensor, col_labels, qr_label=temp_label)
    Q.replace_label(temp_label + "in", lq_label + "out")
    L.replace_label(temp_label + "out", lq_label + "in")

    return L, Q


def truncated_svd(tensor, row_labels, chi=0, threshold=1e-15,
                  absorb_singular_values="right", absolute = True):
    """
    Will perform svd of a tensor, as in tensor_svd, and provide approximate
    decomposition by truncating all but the largest k singular values then
    absorbing S into U, V or both. Truncation is performedby specifying the
    parameter chi (number of singular values to keep).

    Parameters
    ----------
    chi : int, optional
        Maximum number of singular values of each tensor to keep after
        performing singular-value decomposition.
    threshold : float
        Threshold for the magnitude of singular values to keep.
        If absolute then singular values which are less than threshold will be truncated.
        If relative then singular values which are less than max(singular_values)*threshold will be truncated
    """

    U, S, V = tensor_svd(tensor, row_labels)

    singular_values = np.diag(S.data)

    # Truncate to maximum number of singular values

    if chi:
        singular_values_to_keep = singular_values[:chi]
        truncated_evals_1 = singular_values[chi:]
    else:
        singular_values_to_keep = singular_values
        truncated_evals_1 = np.array([])

    # Thresholding

    if absolute:
        truncated_evals_2 = singular_values_to_keep[singular_values_to_keep <= threshold]
        singular_values_to_keep = singular_values_to_keep[singular_values_to_keep > threshold]
    else:
        rel_thresh = singular_values[0]*threshold
        truncated_evals_2 = singular_values_to_keep[singular_values_to_keep <= rel_thresh]
        singular_values_to_keep = singular_values_to_keep[singular_values_to_keep > rel_thresh]

    truncated_evals = np.concatenate((truncated_evals_2, truncated_evals_1), axis=0)

    # Reconstitute and truncate corresponding singular index of U and V

    S.data = np.diag(singular_values_to_keep)

    U.move_index("svd_in", 0)
    U.data = U.data[0:len(singular_values_to_keep)]
    U.move_index("svd_in", (np.size(U.labels) - 1))
    V.data = V.data[0:len(singular_values_to_keep)]


    if absorb_singular_values is None:
        return U, S, V
    # Absorb singular values S into either V or U
    # or take the square root of S and absorb into both (default)
    if absorb_singular_values == "left":
        U_new = contract(U, S, ["svd_in"], ["svd_out"])
        V_new = V
    elif absorb_singular_values == "right":
        V_new = contract(S, V, ["svd_in"], ["svd_out"])
        U_new = U
    else:
        sqrtS = S.copy()
        sqrtS.data = np.sqrt(sqrtS.data)
        U_new = contract(U, sqrtS, ["svd_in"], ["svd_out"])
        V_new = contract(sqrtS, V, ["svd_in"], ["svd_out"])

    return U_new, V_new, truncated_evals


def conjugate(tensor):
    """Return complex conjugate of `tensor`"""
    t = tensor.copy()
    t.conjugate()
    return t

