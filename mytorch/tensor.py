"this code is inspired by https://github.com/amirrezarajabi/rs-dl-framework/blob/main/rsdl/tensors.py"

import numpy as np
from typing import List, NamedTuple, Callable, Optional

class Dependency(NamedTuple):
    tensor: 'Tensor'
    grad_fn: Callable[[np.ndarray], np.ndarray]

class Tensor:
    def __init__(self, data, requires_grad=False, depends_on: List[Dependency] = None):
        # store as numpy array
        self._data = np.array(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.depends_on = depends_on or []
        self.shape = self._data.shape
        self.grad: Optional['Tensor'] = None

        if self.requires_grad:
            self.zero_grad()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, new_data: np.ndarray) -> None:
        self._data = np.array(new_data, dtype=np.float64)
        self.shape = self._data.shape
        self.grad = None

    def zero_grad(self):
        self.grad = np.zeros_like(self.data, dtype=np.float64)

    def backward(self, grad=None):
        # ensure grad is numpy array
        if grad is None:
            grad = np.ones_like(self.data, dtype=np.float64)
        else:
            grad = np.array(grad, dtype=np.float64)

        # initialize grad storage if needed
        if self.grad is None:
            self.zero_grad()

        # accumulate (try-safe)
        try:
            self.grad = self.grad + grad
        except Exception:
            self.grad = grad

        # propagate
        for dependency in self.depends_on:
            try:
                backward_grad = dependency.grad_fn(grad)
            except Exception:
                backward_grad = None

            if backward_grad is None:
                continue

            backward_grad = np.array(backward_grad, dtype=np.float64)

            if dependency.tensor is None:
                continue
            dependency.tensor.backward(backward_grad)

    def __repr__(self) -> 'Tensor':
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"

    # ops
    def __add__(self, other) -> 'Tensor':
        return _add(self, ensure_tensor(other))

    def __mul__(self, other) -> 'Tensor':
        return _mul(self, ensure_tensor(other))

    def __truediv__(self, other) -> 'Tensor':
        return _div(self, ensure_tensor(other))

    def __sub__(self, other):
        return _add(self, _neg(ensure_tensor(other)))

    def __matmul__(self, other) -> 'Tensor':
        return _matmul(self, ensure_tensor(other))

    def __pow__(self, power: float) -> 'Tensor':
        return _pow(self, power)

    def __neg__(self) -> 'Tensor':
        return _neg(self)

    def sum(self):
        return _sum(self)

    def log(self):
        return _log(self)

    def exp(self):
        return _exp(self)

# helper
def ensure_tensor(t):
    if isinstance(t, Tensor):
        return t
    return Tensor(t)

# implementations

def _add(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data + t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            grad = np.array(grad, dtype=np.float64)
            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad
        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            grad = np.array(grad, dtype=np.float64)
            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad

        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(
        data=data,
        requires_grad=requires_grad,
        depends_on=depends_on
    )

def _mul(t1, t2):
    data = t1.data * t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            grad = np.array(grad, dtype=np.float64)
            grad = grad * t2.data
            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad

        depends_on.append(Dependency(t1, grad_fn1))
    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            grad = np.array(grad, dtype=np.float64)
            grad = grad * t1.data
            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad

        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data, requires_grad, depends_on)

def _div(t1, t2):
    data = t1.data / t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    depends_on = []

    if t1.requires_grad:
        def grad_fn1(grad):
            g = np.array(grad, dtype=np.float64)
            return g / t2.data
        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad):
            g = np.array(grad, dtype=np.float64)
            return -g * t1.data / (t2.data ** 2)
        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(
        data=data,
        requires_grad=requires_grad,
        depends_on=depends_on
    )

def _matmul(t1: Tensor, t2: Tensor) -> Tensor:
    "TODO: implement matrix multiplication"
    data = t1.data @ t2.data
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            g = np.array(grad, dtype=np.float64)
            return g @ t2.data.T
        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            g = np.array(grad, dtype=np.float64)
            return t1.data.T @ g
        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data,
                  requires_grad,
                  depends_on)

def _neg(t):
    data = -t.data
    requires_grad = t.requires_grad
    depends_on = []
    if requires_grad:
        def grad_fn(grad):
            return -np.array(grad, dtype=np.float64)
        depends_on.append(Dependency(t, grad_fn))
    return Tensor(data, requires_grad, depends_on)

def _sum(t):
    data = np.sum(t.data)
    requires_grad = t.requires_grad
    depends_on = []
    if requires_grad:
        def grad_fn(grad):
            g = np.array(grad, dtype=np.float64)
            return g * np.ones_like(t.data)
        depends_on.append(Dependency(t, grad_fn))
    return Tensor(data, requires_grad, depends_on)

def _log(t):
    data = np.log(t.data)
    requires_grad = t.requires_grad
    depends_on = []
    if requires_grad:
        def grad_fn(grad):
            g = np.array(grad, dtype=np.float64)
            return g / t.data
        depends_on.append(Dependency(t, grad_fn))
    return Tensor(data, requires_grad, depends_on)

def _exp(t):
    data = np.exp(t.data)
    requires_grad = t.requires_grad
    depends_on = []
    if requires_grad:
        def grad_fn(grad):
            g = np.array(grad, dtype=np.float64)
            return g * data
        depends_on.append(Dependency(t, grad_fn))
    return Tensor(data, requires_grad, depends_on)

def _pow(t, power):
    data = t.data ** power
    requires_grad = t.requires_grad
    depends_on = []
    if requires_grad:
        def grad_fn(grad):
            g = np.array(grad, dtype=np.float64)
            return g * power * (t.data ** (power - 1))
        depends_on.append(Dependency(t, grad_fn))
    return Tensor(data, requires_grad, depends_on)
