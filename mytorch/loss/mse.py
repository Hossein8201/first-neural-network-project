from mytorch import Tensor

def MeanSquaredError(preds: Tensor, actual: Tensor):
    "TODO: implement Mean Squared Error loss"
    # MSE = 1/N * Σ(y_pred - y_true)^2
    diff = preds - actual
    squared = diff ** 2
    return squared.sum() / Tensor(preds.data.shape[0])