import tncontract as tn
import numpy as np

def con(*args):
    """
    Contract a network of tensors. Similar purpose to NCON, described in
    arxiv.org/abs/1402.0939, but designed to work with the Tensor objects of
    tncontract.

    Examples
    --------

    >>> import tncontract as tn

    For the examples below, we define three tensors

    >>> A = tn.Tensor(np.random.rand(3,2,4), labels=["a", "b", "c"])
    >>> B = tn.Tensor(np.random.rand(3,4), labels=["d", "e"])
    >>> C = tn.Tensor(np.random.rand(5,5,2), labels=["f", "g", "h"])

    Contract a pair indices between two tensors 
    -------------------------------------------
    The following contracts  pairs of indices "a","d" and "c","e" of tensors
    `A` and `B`. It is identical to A["a", "c"]*B["d", "e"]

    >>> tn.con(A, B, ("a", "d" ), ("c", "e")) 
    Tensor object: shape = (2), labels = ["b"]

    Contract a pair of indices beloning to one tensor (internal edges)
    ------------------------------------------------------------------
    The following contracts the "f" and "g" indices of tensor `C`

    >>> t.con(C, ("f", "g"))
    Tensor object: shape = (2), labels = ["h"]

    Return the tensor product of a pair of tensors
    ----------------------------------------------
    After all indices have been contracted, `con` will return the tensor
    product of the disconnected components of the tensor contraction. The
    following example returns the tensor product of `A` and `B`. 

    >>> tn.con(A, B) 
    Tensor object: shape = (3, 2, 4, 3, 4), labels = ["a", "b", "c", "d", "e"]

    Contract a network of several tensors
    -------------------------------------
    It is possible to contract a network of several tensors. Internal edges are
    contracted first then edges connecting separate tensors, and then the
    tensor product is taken of the disconnected components resulting from the
    contraction. Edges between separate tensors are contracted in the order
    they appear in the argument list. The result of the example below is a
    scalar (since all indices will be contracted). 

    >>> tn.con(A, B, C, ("a", "d" ), ("c", "e"), ("f", "g"), ("h", "b"))  

    Notes
    -----
    Lists of tensors and index pairs for contraction may be used as arguments. 
    The following example contracts 100 rank 2 tensors in a ring with periodic
    boundary conditions. 
    >>> N=100
    >>> A = tn.Tensor(np.random.rand(2,2), labels=["left","right"])
    >>> tensor_list = [A.suf(str(i)) for i in range(N)]
    >>> idx_pairs = [("right"+str(j), "left"+str(j+1)) for j in range(N-1)]
    >>> tn.con(tensor_list, idx_pairs, ("right"+str(N-1), "left0"))
    """

    tensor_list = []
    contract_list = []
    for x in args:
        #Can take lists of tensors/contraction pairs as arguments
        if isinstance(x, list):
            if isinstance(x[0], tn.Tensor):
                tensor_list.extend(x)
            else:
                contract_list.extend(x)

        elif isinstance(x, tn.Tensor):
            tensor_list.append(x)
        else:
            contract_list.append(x)

    tensor_list = [t.copy() for t in tensor_list] #Unlink from memory
    all_tensor_indices = [t.labels for t in tensor_list]

    #Check that all no index is specified in more than one contraction
    contracted_indices = [item for pair in contract_list for item in pair]
    if len(set(contracted_indices)) != len(contracted_indices):
        raise ValueError("Index found in more than one contraction pair.")

    index_lookup = {}
    for i,labels in enumerate(all_tensor_indices):
        for lab in labels:
            if lab in index_lookup.keys():
                raise ValueError("Index label "+lab+" found in two tensors."+
                        " Tensors must have unique index labelling.")
            index_lookup[lab] = i

    internal_contract = [] #Indicies contracted within the same tensor
    pairwise_contract = [] #Indicies contracted between different tensors
    tensor_pairs = [] 
    tensors_involved = set()
    for c in contract_list:
        if index_lookup[c[0]] == index_lookup[c[1]]:
            internal_contract.append(c)
        else:
            #Takes into account case where multiple indices from a pair of
            #tensors are contracted (will contract in one call to np.dot)
            #TODO: Better to flatten first?
            if (tuple(np.sort((index_lookup[c[0]],index_lookup[c[1]]))) 
                    in tensor_pairs):
                idx = tensor_pairs.index((index_lookup[c[0]],
                    index_lookup[c[1]]))
                if not isinstance(pairwise_contract[idx][0], list):
                    pairwise_contract[idx][0] = [pairwise_contract[idx][0]]
                    pairwise_contract[idx][1] = [pairwise_contract[idx][1]]
                pairwise_contract[idx][0].append(c[0])
                pairwise_contract[idx][1].append(c[1])
            else:
                pairwise_contract.append(list(c))
                tensor_pairs.append(tuple(np.sort((index_lookup[c[0]],index_lookup[c[1]]))))
                tensors_involved.add(index_lookup[c[0]])
                tensors_involved.add(index_lookup[c[1]])

    #Contract all internal indices
    for c in internal_contract:
        tensor_list[index_lookup[c[0]]].trace(c[0], c[1])

    #Contract pairs of tensors 
    connected_component = [i for i in range(len(tensor_list))]
    for c in pairwise_contract:

        if isinstance(c[0], list): 
            #Case where multiple indices of two tensors contracted
            d=index_lookup[c[0][0]] 
            e=index_lookup[c[1][0]]
        else:
            d=index_lookup[c[0]] 
            e=index_lookup[c[1]]

        if d==e:
            tensor_list[d].trace(c[0],c[1])
        else:
            if d<e:
                tensor_list[d]=tn.contract(tensor_list[d], tensor_list[e],
                        c[0], c[1])
                connected_component[e]=d
            else:
                tensor_list[e]=tn.contract(tensor_list[e], tensor_list[d],
                        c[1], c[0])
                connected_component[d]=e
            #Tensor in index_lookup refer to the first tensor 
            #in which the label appers in the list 
            for lab in tensor_list[min(d,e)].labels: 
                index_lookup[lab]=min(d,e) 

    #Take the tensor product of all the disconnected components
    return tn.tensor_product(*[tensor_list[connected_component.index(x)] 
        for x in set(connected_component)])
