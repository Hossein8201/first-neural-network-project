from mytorch.optimizer import Optimizer
import numpy as np

"TODO: (optional) implement Momentum optimizer"
class Momentum(Optimizer):
    def __init__(self, layers, learning_rate=0.01, momentum=0.9):
        super().__init__(layers)
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        # Initialize velocities
        self.velocities = {}
        for layer in layers:
            params = layer.parameters()
            for param in params:
                if param.requires_grad:
                    self.velocities[id(param)] = np.zeros_like(param.data)

    def step(self):
        for layer in self.layers:
            params = layer.parameters()
            for param in params:
                if param.requires_grad and param.grad is not None:
                    param_id = id(param)
                    
                    # Update velocity
                    self.velocities[param_id] = (self.momentum * self.velocities[param_id] - 
                                               self.learning_rate * param.grad.data)
                    
                    # Update parameters
                    param.data += self.velocities[param_id]
