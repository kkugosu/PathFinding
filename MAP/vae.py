# -*- coding: utf-8 -*-
from enum import NAMED_FLAGS
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
from MAIN import Initialize
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
import torch.nn.functional as F
from UTIL import normalize_grad
import mapgen


device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.autograd.set_detect_anomaly(True)


vae = Initialize.customVae

optimizer_p = Initialize.optimizer_vae

temporaryStateLine = mapgen.gen(9999)*2-1

load = 1
if load == 1:
    print("load")
    
    vae.load_state_dict(torch.load('vae.pth'))

i = 0
while i < 100000:
    
    output, mu = vae(temporaryStateLine)

    loss = torch.sum(torch.square(temporaryStateLine - output*1))
    
    loss = loss + torch.sum(torch.square(mu))*0.001
    optimizer_p.zero_grad()
    loss.backward()
    #normalize_grad.NormalizeGradients(vae.parameters())

    optimizer_p.step()
    if torch.isnan(loss).any():
        print("nan")
        break
    
    if i%100 == 0:
        
        #temporaryStateLine = torch.rand(999, 100, 100).to(device)*2 - 1
        #print(temporaryStateLine)
        #print(output)
        #print(mu)
        print("muloss",torch.sum(torch.square(mu)))
        print("totalloss",loss)

        '''
        for name, param in vae.named_parameters():
             print(name)
             print("grad = ",param.grad)
        '''
        #temporaryStateLine = torch.rand(100, 1, 100, 100).to(device)*2 - 1
        torch.save(vae.state_dict(),'vae.pth')
    if i%101 == 0:
        torch.save(vae.state_dict(),'1vae.pth')

    i = i + 1

