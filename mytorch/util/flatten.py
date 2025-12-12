import numpy as np
from mytorch import Tensor
from mytorch.tensor import Dependency

def flatten(x: Tensor) -> Tensor:
    """
    TODO: implement flatten. 
    this methods transforms a n dimensional array into a flat array
    hint: use numpy flatten
    """
    data = x.data.flatten()
    req_grad = x.requires_grad
    
    if req_grad:
        def grad_fn(grad: np.ndarray):
            return grad.reshape(x.shape)
        
        depends_on = [Dependency(x, grad_fn)]
    else:
        depends_on = []
        
    return Tensor(data=data, requires_grad=req_grad, depends_on=depends_on)
