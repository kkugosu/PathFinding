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


def render(x, y, z):
    episodeL = len(x[0])
    fig, axs = plt.subplots(2, 2)
    for i in range(len(x)):
        axs[0, 0].plot(x[i].cpu().detach().numpy(), y[i].cpu().detach().numpy(), 'tab:red')
        axs[0, 0].set_xlim([-episodeL, episodeL])
        axs[0, 0].set_ylim([-episodeL, episodeL])
        axs[1, 0].plot(y[i].cpu().detach().numpy(), z[i].cpu().detach().numpy(), 'tab:orange')
        axs[1, 0].set_xlim([-episodeL, episodeL])
        axs[1, 0].set_ylim([-episodeL, episodeL])
        axs[1, 1].plot(x[i].cpu().detach().numpy(), z[i].cpu().detach().numpy(), 'tab:green')
        axs[1, 1].set_xlim([-episodeL, episodeL])
        axs[1, 1].set_ylim([-episodeL, episodeL])
            
    for ax in axs.flat:
        ax.set(xlabel='x-label', ylabel='y-label')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.show()