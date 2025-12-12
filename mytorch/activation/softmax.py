import numpy as np
from mytorch import Tensor, Dependency


def softmax(x: Tensor) -> Tensor:
    """
    TODO: implement softmax function
    hint: you can do it using function you've implemented (not directly define grad func)
    hint: you can't use sum because it has not axis argument so there are 2 ways:
        1. implement sum by axis
        2. using matrix mul to do it :) (recommended)
    hint: a/b = a*(b^-1)
    """
    data = x.data - np.max(x.data, axis=1, keepdims=True)
    exp_data = np.exp(data)
    data = exp_data / np.sum(exp_data, axis=1, keepdims=True)
    requires_grad = x.requires_grad
    
    if requires_grad:
        def grad_fn(grad: np.ndarray):
            s = data
            jacobian = np.einsum('ij,ik->ijk', s, s)
            diag = np.einsum('ij,jk->ijk', s, np.eye(s.shape[1]))
            softmax_grad = diag - jacobian
            return np.einsum('ijk,ik->ij', softmax_grad, grad)
        depends_on = [Dependency(x, grad_fn)]
    else:
        depends_on = []
    
    return Tensor(data, requires_grad, depends_on)