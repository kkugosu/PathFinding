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
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.autograd.set_detect_anomaly(True)

def gen(sample_num):
    total_map = torch.zeros([sample_num, 100, 100])

    for s in range(sample_num):
        
        num = torch.randint(1, 5, (1,))
        a = torch.zeros(1,2)
        b = torch.zeros(1,1)
        for i in range(num):
            tmp = torch.rand(1,2)*20 - 10
            a = torch.cat((a, tmp), dim = 0)
    
        hillList = a[1:].to(device)

        for i in range(num):
            tmp = torch.rand(1,1)*3 + 2
            b = torch.cat((b, tmp), dim = 0)

        varList = b[1:].to(device)

        rows = torch.arange(0, 100)
        cols = torch.arange(0, 100)

        row_indices, col_indices = torch.meshgrid(rows, cols, indexing="ij")
        indices_tensor = torch.stack((row_indices, col_indices), dim = 2)
        basemap = (indices_tensor.float()*2 - 99)/10
        basemap = basemap.to(device)

        sumweight = torch.zeros_like(basemap[:, :, 0]).unsqueeze(-1)

        for i in range(len(hillList)):
            sumweight = sumweight + torch.exp(- torch.sum(torch.square(basemap - hillList[i])/varList[i], -1)).unsqueeze(-1) # mapping to 0 ~ 1

        sumweight = torch.clamp(sumweight*2, max = 1)
        total_map[s] = sumweight.squeeze()
    return total_map.to(device)

'''

row_indices, col_indices = torch.meshgrid(rows, cols, indexing="ij")
indices_tensor = torch.stack((row_indices, col_indices), dim = 2)

sumweight = gen()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(row_indices.cpu().numpy(), col_indices.cpu().numpy(), sumweight.squeeze().cpu().numpy())

ax.set_zlim(0, 5)
plt.show()
'''