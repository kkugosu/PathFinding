import stat
from typing import Any
import torch

class MyCustomFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input*2
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input



x = torch.tensor([1.0, 2.0, -1.0], requires_grad=True)
y = MyCustomFunction.apply(x)
y.backward(torch.tensor([1.0, 1.0, 1.0]))
print(x.grad())