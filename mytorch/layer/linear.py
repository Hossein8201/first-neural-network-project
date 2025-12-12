from mytorch import Tensor
from mytorch.layer import Layer
from mytorch.util import initializer

import numpy as np


class Linear(Layer):
    def __init__(self, inputs: int, outputs: int, need_bias: bool = False, mode="xavier") -> None:
        super().__init__(need_bias)
        self.inputs = inputs
        self.outputs = outputs
        self.need_bias = need_bias
        
        self.initialize(mode)

    def forward(self, x: Tensor) -> Tensor:
        "TODO: implement forward pass"
        output = x @ self.weight
        if self.need_bias:
            output = output + self.bias
        return output

    def initialize(self, mode="xavier"):
        "TODO: initialize weight by initializer function (mode)"
        self.weight = Tensor(
            data=initializer((self.inputs, self.outputs), mode),
            requires_grad=True
        )

        "TODO: initialize bias by initializer function (zero mode)"
        if self.need_bias:
            self.bias = Tensor(
                data=initializer((1, self.outputs), "zero"),
                requires_grad=True
            )

    def zero_grad(self):
        "TODO: implement zero grad"
        self.weight.zero_grad()
        if self.need_bias:
            self.bias.zero_grad()

    def parameters(self):
        "TODO: return weights and bias"
        if self.need_bias:
            return [self.weight, self.bias]
        return [self.weight]

    def __str__(self) -> str:
        return "linear - total param: {} - in: {}, out: {}".format(self.inputs * self.outputs, self.inputs,
                                                                   self.outputs)
