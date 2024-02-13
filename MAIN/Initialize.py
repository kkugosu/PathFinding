import torch
from torch import nn
import copy
import numpy as np
import math
from math import cos, sin, pi
import sympy
import copy
import random
import matplotlib.pyplot as plt
from torch import nn
import torch
import torch.distributions as dist
from torchvision.transforms import Lambda
import math
import time
import gc
from NeuralNet import NN
from mpl_toolkits.mplot3d import Axes3D
device = 'cuda' if torch.cuda.is_available() else 'cpu'

load = 0
skill_num = 9999
el = 10
var = 4
mapEmbeddingSpace = 100

vaecnn = NN.VAEcnn().to(device)
customVae = NN.VAE().to(device)
#globalNet = NN.SelfAttention(6,256,2).to(device)
globalTransformer = NN.Transformer(6 + mapEmbeddingSpace ,256,2).to(device)
globalTransformerQ = NN.TransformerQ(8 + mapEmbeddingSpace ,256,1).to(device)

optimizer_vae = torch.optim.SGD(customVae.parameters(), lr = 1e-9, weight_decay= 0.00001 )
optimizer_vaec = torch.optim.SGD(vaecnn.parameters(), lr = 1e-8, weight_decay= 0.00001 )
#optimizer_p = torch.optim.SGD(globalNet.parameters(), lr = 1e-6, weight_decay= 0.00001 )
optimizer_gp = torch.optim.SGD(globalTransformer.parameters(), lr = 1e-11, weight_decay= 0.00001 )
optimizer_gpQ = torch.optim.SGD(globalTransformerQ.parameters(), lr = 1e-8, weight_decay= 0.00001 )

if load == 1:
    print("load")
    #globalNet.load_state_dict(torch.load('PARAM/bigpolicy.pth'))
    globalTransformer.load_state_dict(torch.load('PARAM/mappolicy.pth'))
    globalTransformerQ.load_state_dict(torch.load('PARAM/mapqueue.pth'))
    customVae.load_state_dict(torch.load('vae.pth'))
    vaecnn.load_state_dict(torch.load('vaecnn.pth'))
    
if load == 2:
    print("load")
    #globalNet.load_state_dict(torch.load( str(skill_num) + "n" + str(el) + 'bigpolicy.pth'))
    globalTransformer.load_state_dict(torch.load( 'PARAM/' + str(skill_num) + "n" + str(el) + '/mappolicy_tmp.pth'))
    globalTransformerQ.load_state_dict(torch.load( 'PARAM/' + str(skill_num) + "n" + str(el) + '/mapqueue_tmp.pth'))
    customVae.load_state_dict(torch.load('vae_tmp.pth'))
    vaecnn.load_state_dict(torch.load('vaecnn_tmp.pth'))
    
