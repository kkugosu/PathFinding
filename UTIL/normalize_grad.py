import torch

def NormalizeGradients(parameters):
    total_norm = 0
    for param in parameters:
        
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            scale = 1/(param_norm + 1e-6)
            #print("scale = ",scale)
            param.grad.data.mul_(scale**0.5)


