from mytorch import Tensor
from mytorch.layer import Layer
from mytorch.util import initializer
from mytorch.tensor import Dependency
import numpy as np

class Conv2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1), need_bias: bool = False, mode="xavier") -> None:
        super().__init__(need_bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.need_bias = need_bias
        self.weight: Tensor = None
        self.bias: Tensor = None
        self.initialize_mode = mode

        self.initialize()

    def forward(self, x: Tensor) -> Tensor:
        "TODO: implement forward pass"
        batch_size, in_channels, height, width = x.data.shape
        
        # Calculate output dimensions
        out_height = (height + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        out_width = (width + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        
        # Apply padding
        if self.padding != (0, 0):
            x_padded = np.pad(x.data, 
                            ((0, 0), (0, 0), 
                             (self.padding[0], self.padding[0]), 
                             (self.padding[1], self.padding[1])), 
                            mode='constant')
        else:
            x_padded = x.data
        
        # Perform convolution
        output = np.zeros((batch_size, self.out_channels, out_height, out_width))
        
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride[0]
                h_end = h_start + self.kernel_size[0]
                w_start = j * self.stride[1]
                w_end = w_start + self.kernel_size[1]
                
                window = x_padded[:, :, h_start:h_end, w_start:w_end]
                for k in range(self.out_channels):
                    output[:, k, i, j] = np.sum(
                        window * self.weight.data[k, :, :, :], 
                        axis=(1, 2, 3)
                    )
        
        # Add bias if needed
        if self.need_bias:
            output += self.bias.data[:, :, None, None]
        
        requires_grad = x.requires_grad or self.weight.requires_grad or (self.need_bias and self.bias.requires_grad)
        
        if requires_grad:
            def grad_fn(grad: np.ndarray):
                # Gradient for input
                input_grad = np.zeros_like(x_padded)
                weight_grad = np.zeros_like(self.weight.data)
                
                if self.need_bias:
                    bias_grad = np.zeros_like(self.bias.data)
                
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride[0]
                        h_end = h_start + self.kernel_size[0]
                        w_start = j * self.stride[1]
                        w_end = w_start + self.kernel_size[1]
                        
                        window = x_padded[:, :, h_start:h_end, w_start:w_end]
                        
                        for k in range(self.out_channels):
                            # Gradient for weights
                            weight_grad[k] += np.sum(
                                window * grad[:, k, i, j][:, None, None, None], 
                                axis=0
                            )
                            
                            # Gradient for input
                            input_grad[:, :, h_start:h_end, w_start:w_end] += \
                                self.weight.data[k, :, :, :] * grad[:, k, i, j][:, None, None, None]
                        
                        # Gradient for bias
                        if self.need_bias:
                            bias_grad += np.sum(grad[:, :, i, j], axis=0)
                
                # Remove padding from input gradient
                if self.padding != (0, 0):
                    input_grad = input_grad[:, :, 
                                          self.padding[0]:-self.padding[0], 
                                          self.padding[1]:-self.padding[1]]
                
                # Store gradients
                if x.requires_grad:
                    x_grad = input_grad
                else:
                    x_grad = None
                
                if self.weight.requires_grad:
                    self.weight.grad.data += weight_grad
                
                if self.need_bias and self.bias.requires_grad:
                    self.bias.grad.data += bias_grad
                
                return x_grad
            
            depends_on = [Dependency(x, grad_fn)]
        else:
            depends_on = []
        
        return Tensor(data=output, requires_grad=requires_grad, depends_on=depends_on)
    
    def initialize(self):
        "TODO: initialize weights"
        weight_shape = (self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        self.weight = Tensor(
            data=initializer(weight_shape, self.initialize_mode),
            requires_grad=True
        )
        
        if self.need_bias:
            self.bias = Tensor(
                data=initializer((self.out_channels,), "zero"),
                requires_grad=True
            )

    def zero_grad(self):
        "TODO: implement zero grad"
        if self.weight.requires_grad:
            self.weight.zero_grad()
        if self.need_bias and self.bias.requires_grad:
            self.bias.zero_grad()

    def parameters(self):
        "TODO: return weights and bias"
        if self.need_bias:
            return [self.weight, self.bias]
        else:
            return [self.weight]
    
    def __str__(self) -> str:
        return "conv 2d - total params: {} - kernel: {}, stride: {}, padding: {}".format(
                                                                                    self.kernel_size[0] * self.kernel_size[1],
                                                                                    self.kernel_size,
                                                                                    self.stride, self.padding)
