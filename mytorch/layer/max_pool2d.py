from mytorch import Tensor
from mytorch.layer import Layer
from mytorch.tensor import Dependency
import numpy as np

class MaxPool2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1)) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        "TODO: implement forward pass"
        batch_size, channels, height, width = x.data.shape
        
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
        
        # Perform max pooling
        output = np.zeros((batch_size, channels, out_height, out_width))
        max_indices = np.zeros((batch_size, channels, out_height, out_width, 2), dtype=int)
        
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride[0]
                h_end = h_start + self.kernel_size[0]
                w_start = j * self.stride[1]
                w_end = w_start + self.kernel_size[1]
                
                window = x_padded[:, :, h_start:h_end, w_start:w_end]
                output[:, :, i, j] = np.max(window, axis=(2, 3))
                
                # Store indices for backward pass
                for b in range(batch_size):
                    for c in range(channels):
                        flat_idx = np.argmax(window[b, c])
                        h_idx = flat_idx // self.kernel_size[1]
                        w_idx = flat_idx % self.kernel_size[1]
                        max_indices[b, c, i, j] = [h_start + h_idx, w_start + w_idx]
        
        requires_grad = x.requires_grad
        
        if requires_grad:
            def grad_fn(grad: np.ndarray):
                input_grad = np.zeros_like(x_padded)
                
                for i in range(out_height):
                    for j in range(out_width):
                        for b in range(batch_size):
                            for c in range(channels):
                                h_idx, w_idx = max_indices[b, c, i, j]
                                input_grad[b, c, h_idx, w_idx] += grad[b, c, i, j]
                
                # Remove padding from gradient
                if self.padding != (0, 0):
                    input_grad = input_grad[:, :, 
                                          self.padding[0]:-self.padding[0], 
                                          self.padding[1]:-self.padding[1]]
                
                return input_grad
            
            depends_on = [Dependency(x, grad_fn)]
        else:
            depends_on = []
        
        return Tensor(data=output, requires_grad=requires_grad, depends_on=depends_on)
    
    def __str__(self) -> str:
        return "max pool 2d - kernel: {}, stride: {}, padding: {}".format(self.kernel_size, self.stride, self.padding)
