# tncontract
tncontract is an open-source tensor-network library for Python. The goal of tncontract is to provide a simple and intuitive framework for implementing tensor-network algorithms. The tncontract library uses the high-performance Numpy library as a numerical backend. tncontract can easily interface with many other Python libraries, and has built-in conversions for the popular quantum information library: QuTiP. Currently, tncontract includes many algorithms for one-dimensional and two-dimensional tensor networks. It is under active development with new features being constantly added. 

##Code Example

```python
#Define a 2x2 tensor corresponding to the identity matrix. Assign labels "spam"
#and "eggs" to, respectively, the first and second indices of the tensor

>>> A = Tensor([[1, 0], [0, 1]], labels = ["spam", "eggs"])
>>> print(A)
Tensor object: shape = (2, 2), labels = ['spam', 'eggs']

#Define a 2x3x2x4 tensor with random entries without specifying labels. 
#As labels are not specified, they will be assigned automatically using the
#convention "i0", "i1", "i2", ...

>>> B = random_tensor(2, 3, 2, 4)
>>> print(B)
Tensor object: shape = (2, 3, 2, 4), labels = ['i0', 'i1', 'i2', 'i3']

#Contract the "spam" index of tensor A with the "i2" index of tensor B.

>>> C = contract(A, B, "spam", "i2")
>>> print(C)
Tensor object: shape = (2, 2, 3, 4), labels = ['eggs', 'i0', 'i1', 'i3']

#Contract, respectively,  the "spam" and "eggs" indices of tensor A with the 
#"i0" and "i2" indices of tensor B.

>>> D = contract(A, B, ["spam", "eggs"], ["i0", "i2"])
>>> print(D)
Tensor object: shape = (3, 4), labels = ['i1', 'i3']

#Contract "i0" and "i2" of tensor B internally. 

>>> B.contract_internal("i0", "i2")
>>> print(B)
Tensor object: shape = (3, 4), labels = ['i1', 'i3']
```
