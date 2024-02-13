# -*- coding: utf-8 -*-
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

x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X,Y = np.meshgrid(x, y)


def render(map, x, y, z): #5maps, 3view
    episodeL = len(x[0])
    fig, axs = plt.subplots(5, 4, figsize = (10, 9))
    for i in range(5): #num of map
        contour = axs[i, 0].contourf(X, Y, map[i].cpu().numpy(), 20, cmap = 'viridis')
        for j in range(len(x[i])):
            axs[i, 1].plot(x[i][j], y[i][j], 'tab:red') # size = epl
            axs[i, 1].set_xlim([-episodeL, episodeL])
            axs[i, 1].set_ylim([-episodeL, episodeL])
            axs[i, 2].plot(y[i][j], z[i][j], 'tab:orange')
            axs[i, 2].set_xlim([-episodeL, episodeL])
            axs[i, 2].set_ylim([-episodeL, episodeL])
            axs[i, 3].plot(x[i][j], z[i][j], 'tab:green')
            axs[i, 3].set_xlim([-episodeL, episodeL])
            axs[i, 3].set_ylim([-episodeL, episodeL])
        
    plt.tight_layout()
    for ax in axs.flat:
        ax.set(xlabel='x-label', ylabel='y-label')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.show()
    
