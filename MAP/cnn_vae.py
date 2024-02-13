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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.autograd.set_detect_anomaly(True)

temporaryStateLine = torch.rand(1000, 1, 100, 100).to(device)*2 - 1
load = 0
if load == 1:
    print("load")
    Initialize.vaecnn.load_state_dict(torch.load('vae.pth'))

i = 0
while i < 10000:
    output, mu = Initialize.vaecnn(temporaryStateLine)
    loss = torch.sum(torch.square(temporaryStateLine - output*1))
    loss = loss #+ torch.sum(torch.square(mu))
    Initialize.optimizer_vaec.zero_grad()
    loss.backward()
    normalize_grad.NormalizeGradients(Initialize.vaecnn.parameters(), target_norm=1.0)

    Initialize.optimizer_vaec.step()
    if torch.isnan(loss).any():
        print("nan")
        break
    
    if i%100 == 0:
        temporaryStateLine = torch.rand(1000, 1, 100, 100).to(device)*2 - 1
        print("output", output.size())
        print("tmps",temporaryStateLine.size())
        print("sizecompare")
        print(torch.sum(torch.square(mu)))
        print(loss)

        '''
        for name, param in vae.named_parameters():
             print(name)
             print("grad = ",param.grad)
        '''
        #temporaryStateLine = torch.rand(100, 1, 100, 100).to(device)*2 - 1
        torch.save(Initialize.vaecnn.state_dict(),'vae.pth')
    if i%101 == 0:
        torch.save(Initialize.vaecnn.state_dict(),'1vae.pth')

    i = i + 1

