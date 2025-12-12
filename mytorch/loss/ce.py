import numpy as np
from mytorch import Tensor
from mytorch.tensor import Dependency

def CategoricalCrossEntropy(preds: Tensor, label: Tensor):
    "TODO: implement Categorical Cross Entropy loss"
    """
    preds: Tensor (after softmax) with shape (batch, num_classes)
    label: Tensor either shape (batch,) with integer labels or (batch, num_classes) one-hot
    returns: Tensor scalar (mean cross-entropy) with proper dependency on preds
    """

    # Convert integer labels to one-hot if needed
    if len(label.data.shape) == 1:
        num_classes = preds.data.shape[1]
        label_one_hot = np.eye(num_classes)[label.data.astype(int)]
        label = Tensor(label_one_hot, requires_grad=False)

    # numerical stability epsilon
    eps = 1e-12

    # Create log_probs as a Tensor but keep dependency on preds so backward flows.
    clipped = np.clip(preds.data, eps, 1.0 - eps)
    log_data = np.log(clipped)

    # grad_fn for log: d/dx log(x) = 1/x
    if preds.requires_grad:
        def log_grad_fn(grad: np.ndarray):
            # grad has shape (batch, num_classes)
            return grad / preds.data
        log_probs = Tensor(log_data, requires_grad=True, depends_on=[Dependency(preds, log_grad_fn)])
    else:
        log_probs = Tensor(log_data, requires_grad=False)

    # product = label * log_probs  (uses Tensor __mul__ so graph continues)
    product = label * log_probs
    sum_product = product.sum()   # scalar Tensor

    # loss = - sum_product / N
    N = preds.data.shape[0]
    neg_sum = -sum_product
    loss = neg_sum / Tensor(N)

    return loss
