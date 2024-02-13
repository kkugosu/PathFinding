a = -3.44
from math import cos, sin
print(cos(3.14))
import torch
from NeuralNet import NN
a = torch.rand(1, 6)
testcod = NN.SelfAttention(6,256,2)
print(testcod(a))

weightlist = torch.tensor([[1,0,0], 
                           [-1,0,0], 
                           [0,1,0], 
                           [0,-1,0], 
                           [0,0,1], 
                           [-1/(3**(1/2)), 1/(3**(1/2)), 1/(3**(1/2))], 
                           [1/(3**(1/2)), -1/(3**(1/2)), 1/(3**(1/2))], 
                           [1/(3**(1/2)), 1/(3**(1/2)), 1/(3**(1/2))], 
                           [-1/(3**(1/2)), -1/(3**(1/2)), 1/(3**(1/2))], 
                           ])
print(weightlist)