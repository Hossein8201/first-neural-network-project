from mytorch.optimizer import Optimizer
import numpy as np

"TODO: (optional) implement RMSprop optimizer"
class RMSprop(Optimizer):
    def __init__(self, layers, learning_rate=0.001, beta=0.9, epsilon=1e-8):
        super().__init__(layers)
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        
        # Initialize squared gradients
        self.squared_grads = {}
        for layer in layers:
            params = layer.parameters()
            for param in params:
                if param.requires_grad:
                    self.squared_grads[id(param)] = np.zeros_like(param.data)

    def step(self):
        for layer in self.layers:
            params = layer.parameters()
            for param in params:
                if param.requires_grad and param.grad is not None:
                    param_id = id(param)
                    
                    # Update squared gradients
                    self.squared_grads[param_id] = (self.beta * self.squared_grads[param_id] + 
                                                  (1 - self.beta) * (param.grad.data ** 2))
                    
                    # Update parameters
                    param.data -= (self.learning_rate * param.grad.data / 
                                 (np.sqrt(self.squared_grads[param_id]) + self.epsilon))
