from typing import List
from mytorch.layer import Layer
from mytorch.optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, layers: List[Layer], learning_rate=0.01):
        super().__init__(layers)
        self.learning_rate = learning_rate

    def step(self):
        "TODO: implement SGD algorithm"
        for layer in self.layers:
            params = layer.parameters()
            for param in params:
                if param.requires_grad and param.grad is not None:
                    param.data -= self.learning_rate * param.grad
