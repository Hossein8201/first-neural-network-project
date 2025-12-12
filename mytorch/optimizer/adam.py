from mytorch.optimizer import Optimizer
import numpy as np

"TODO: (optional) implement Adam optimizer"
class Adam(Optimizer):
    def __init__(self, layers, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(layers)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0

        self.m = {}
        self.v = {}

        # init moment buffers
        for layer in layers:
            params = layer.parameters()
            for param in params:
                if param.requires_grad:
                    self.m[id(param)] = np.zeros_like(param.data, dtype=np.float64)
                    self.v[id(param)] = np.zeros_like(param.data, dtype=np.float64)

    def step(self):
        self.t += 1

        for layer in self.layers:
            params = layer.parameters()

            for param in params:
                if param.requires_grad and param.grad is not None:

                    g = param.grad
                    pid = id(param)

                    # Update biased first moment
                    self.m[pid] = self.beta1 * self.m[pid] + (1 - self.beta1) * g

                    # Update biased second raw moment
                    self.v[pid] = self.beta2 * self.v[pid] + (1 - self.beta2) * (g * g)

                    # Bias correction
                    m_hat = self.m[pid] / (1 - self.beta1 ** self.t)
                    v_hat = self.v[pid] / (1 - self.beta2 ** self.t)

                    # Update parameters
                    param.data -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
