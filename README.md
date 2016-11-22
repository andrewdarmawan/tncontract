# tncontract
**tncontract** is an open-source tensor-network library for Python. The goal of tncontract is to provide a simple and intuitive framework for writing tensor-network algorithms. The tncontract library uses the powerful NumPy library as a numerical backend. It can easily interface with many other Python libraries, and has built-in conversions for the popular quantum library: QuTiP. Currently, tncontract includes many algorithms for one-dimensional and two-dimensional tensor networks. While far from complete, it is under active development with new features being constantly added.

##Installation

tncontract requires recent versions of Python, NumPy, and SciPy. To install tncontract, download the source code using the link above, then in the root directory of the package run

```shell
$ python setup.py install
```

##Code Examples

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
>>> B.contract_internal("i0", "i2")
>>> print(B)
Tensor object: shape = (3, 4), labels = ['i1', 'i3']
```

##Contributors

[Andrew Darmawan](https://github.com/andrewdarmawan) Université de Sherbrooke

[Arne Grimsmo](https://github.com/arnelg) Université de Sherbrooke
