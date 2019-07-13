# tncontract
**Note from developer: tncontract has reached a usable state, however is not currently being actively developed. Maintenance/support may occasionally be provided if time allows, however should not be expected.**

**tncontract** is an open-source tensor-network library for Python. The goal of tncontract is to provide a simple and intuitive framework for writing tensor-network algorithms. The tncontract library uses the powerful NumPy library as a numerical backend. It can easily interface with many other Python libraries, and has built-in conversions for the popular quantum library: QuTiP. Currently, tncontract includes many algorithms for one-dimensional and two-dimensional tensor networks. 

## Installation

tncontract requires recent versions of Python, NumPy, and SciPy (note there appear to be problems with the Python2 support, so Python3 is recommended). To install tncontract, download the source code using the link above, then in the root directory of the package run

```shell
$ python setup.py install
```

## Code Examples

Here are some simple examples showing how to define and contract tensors in tncontract. To define a Tensor object, the user provides an array-like object and a label for each index (axis) of the array. These labels are persistent, i.e. they will refer to the same indices after the tensor has been contracted with other tensors. Here we define a 2x2 tensor and assign labels "spam" and "eggs" to, respectively, the first and second indices of the tensor.
```python
>>> A = Tensor([[1, 2], [3, 4]], labels = ["spam", "eggs"])
>>> print(A)
Tensor object: shape = (2, 2), labels = ['spam', 'eggs']
```
The data is stored as a numpy array.
```python
>>> A.data
array([[1, 2],
       [3, 4]])
```

Here we define a 2x3x2x4 tensor with random entries with index labels given, respectively, by "i0", "i1", "i2" and "i3".
```python 
>>> B = random_tensor(2, 3, 2, 4, labels = ['i0', 'i1', 'i2', 'i3'])
>>> print(B)
Tensor object: shape = (2, 3, 2, 4), labels = ['i0', 'i1', 'i2', 'i3']
```

To perform a simple, pairwise tensor contraction we specify a pair of tensors and an index to contract from each tensor. Given A and B, defined above, we contract the "spam" index of tensor A with the "i2" index of tensor B.

```python
>>> C = contract(A, B, "spam", "i2")
>>> print(C)
Tensor object: shape = (2, 2, 3, 4), labels = ['eggs', 'i0', 'i1', 'i3']
```
The indices of the resulting tensor C are the uncontracted indices of tensors A and B. You can see that their labels have been preserved. 

We can simultaneously contract multiple indices. For instance, to contract the "spam" index of A with the "i0" index of B and at the same time contract the "eggs" index of A with the "i2" index of B we would use
```python
>>> D = contract(A, B, ["spam", "eggs"], ["i0", "i2"])
>>> print(D)
Tensor object: shape = (3, 4), labels = ['i1', 'i3']
```
The following shorthand can be used to perform the same operation.
```python
>>> D = A["spam", "eggs"]*B["i0", "i2"]
>>> print(D)
Tensor object: shape = (3, 4), labels = ['i1', 'i3']
```
The following contracts a pair of indices within the same tensor.
```python
>>> B.trace("i0", "i2")
>>> print(B)
Tensor object: shape = (3, 4), labels = ['i1', 'i3']
```
### Contract multiple tensors

tncontract contains the function, con, to perform general contractions of multiple Tensor objects. It is similar in purpose to NCON, described in [arxiv.org/abs/1402.0939](arxiv.org/abs/1402.0939), but is designed to work with the Tensor objects of tncontract.

For the examples below, we define three tensors

```python
>>> A = tn.Tensor(np.random.rand(3,2,4), labels=["a", "b", "c"])
>>> B = tn.Tensor(np.random.rand(3,4), labels=["d", "e"])
>>> C = tn.Tensor(np.random.rand(5,5,2), labels=["f", "g", "h"])
```

#### Contract a pair indices between two tensors 
The following contracts  pairs of indices "a","d" and "c","e" of tensors
`A` and `B`. It is identical to `A["a", "c"]*B["d", "e"]`

```python
>>> tn.con(A, B, ("a", "d" ), ("c", "e")) 
Tensor object: shape = (2), labels = ["b"]
```

#### Contract a pair of indices beloning to one tensor (internal edges)
The following contracts the "f" and "g" indices of tensor `C`

```python
>>> t.con(C, ("f", "g"))
Tensor object: shape = (2), labels = ["h"]
```

#### Return the tensor product of a pair of tensors
After all indices have been contracted, con will return the tensor
product of the disconnected components of the tensor contraction. The
following example returns the tensor product of `A` and `B`. 

```python
>>> tn.con(A, B) 
Tensor object: shape = (3, 2, 4, 3, 4), labels = ["a", "b", "c", "d", "e"]
```

#### Contract a network of several tensors

It is possible to contract a network of several tensors. Internal edges are
contracted first then edges connecting separate tensors, and then the
tensor product is taken of the disconnected components resulting from the
contraction. Edges between separate tensors are contracted in the order
they appear in the argument list. The result of the example below is a
scalar (since all indices will be contracted). 

```python
>>> tn.con(A, B, C, ("a", "d" ), ("c", "e"), ("f", "g"), ("h", "b"))  
```

#### Notes

Lists of tensors and index pairs for contraction may be used as arguments. 
The following example contracts 100 rank 2 tensors in a ring with periodic
boundary conditions. 

```python
>>> N=100
>>> A = tn.Tensor(np.random.rand(2,2), labels=["left","right"])
>>> tensor_list = [A.suf(str(i)) for i in range(N)]
>>> idx_pairs = [("right"+str(j), "left"+str(j+1)) for j in range(N-1)]
>>> tn.con(tensor_list, idx_pairs, ("right"+str(N-1), "left0"))
```

## Contributors

[Andrew Darmawan](https://github.com/andrewdarmawan) Yukawa Institute for Theoretical Physics (YITP) — Kyoto University

[Arne Grimsmo](https://github.com/arnelg) The University of Sydney
