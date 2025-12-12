import numpy as np
from mytorch import Tensor, Dependency

def relu(x: Tensor) -> Tensor:
    "TODO: implement relu function"
    data = np.maximum(0, x.data)
    requires_grad = x.requires_grad
    
    if requires_grad:
        def grad_fn(grad: np.ndarray):
            return grad * (x.data > 0)
        depends_on = [Dependency(x, grad_fn)]
    else:
        depends_on = []
    return Tensor(data=data, requires_grad=requires_grad, depends_on=depends_on)