
# -*- coding: utf-8 -*-
from enum import NAMED_FLAGS
import torch
from torch import embedding, nn
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
from UTIL import render

import json

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.autograd.set_detect_anomaly(True)
torch.set_printoptions(threshold=1000000)
thisvae = Initialize.customVae
thisvae.load_state_dict(torch.load('PARAM/VAE/vae.pth'))
for param in thisvae.parameters():
    param.requires_grad = False

# Define the variable
var = Initialize.var # 
skillNum = Initialize.skill_num
epLength = Initialize.el
embeddingSpaceM = Initialize.mapEmbeddingSpace
policy = Initialize.globalTransformer
optimizerP = Initialize.optimizer_gp

dist_by_coordinate = lambda i, j : (torch.sum((i-j)**2, -1))**(1/2)
angle_to_coord = lambda i: torch.stack((torch.cos(i[:,0])*torch.sin(i[:,1]), torch.sin(i[:,0])*torch.sin(i[:,1]), torch.cos(i[:,1])), dim=1)

cusk = 0 # current updat0ng skill
training_recur = 1

# load model
load = 1
if load == 1:
    print("load")
    policy.load_state_dict(torch.load( 'PARAM/POLICY/pretrainedPolicytest2.pth'))

output, mu = thisvae(torch.zeros(10000).to(device) -1)
originalinput = mu.repeat(skillNum,1).clone()
#print("mu = ", mu)

rep = 0
while rep < training_recur:
    if rep%100 == 0:
        print("reset")
        phi = math.pi * (torch.rand(skillNum)*2-1)# pi ~ -pi
        theta = math.pi * (torch.rand(skillNum)*2 - 1) #pi ~ -pi
        angle_ary = torch.stack([phi, theta], dim = 1).to(device)
        #print(angle_ary.size())
        batchedWeightList = angle_to_coord(angle_ary)
        #print("bsize",batchedWeightList.size())
        #almost uniform distribution of weight parameter 9999,3

        #batchedMap = torch.rand(skillNum, embeddingSpaceM).to(device)*0.01 - 0.005
        batchedMap = originalinput
        #9999,100

        batchedInitialState = torch.zeros(skillNum, 3, requires_grad = True).to(device) #+ batchedWeightList*0.01
        #9999,3

    tmpInput = torch.cat((batchedWeightList, batchedMap, batchedInitialState), dim = 1) # 9999, 106
    temporaryStateLine = torch.zeros(skillNum, 1, 106, requires_grad = True).to(device)# batchsize = skillnum, el, featurenum
    stateList = torch.zeros(skillNum, epLength + 1, 3, requires_grad = True).to(device)
    pre_output1 = phi
    pre_output2 = theta
    pre_output = torch.stack([pre_output1, pre_output2], dim = 1).to(device)
    #9999, el, 106
    step = 0
    while step < epLength:
        #print("tmpInput = ",tmpInput)
        output = policy(tmpInput.detach().clone())
        #print("step = ", step)
        #print(loss)

        output = torch.atan(output) # -pi/2 ~ pi/2
        #output = torch.tensor([[-0.00, -0.00],[-0.00,  0.00]], device='cuda:0')
        #print("ouptut = ", output)
        #print("pre_ouptut = ", pre_output)
        tmpState = angle_to_coord(output + pre_output.detach().clone())
        pre_output = output + pre_output
        stateList[:, step+1] = stateList[:, step] + tmpState
        tmpInput = torch.cat((batchedWeightList, batchedMap, tmpState ), dim = 1) + temporaryStateLine[:, -1]
        temporaryStateLine = torch.cat([temporaryStateLine,  tmpInput.unsqueeze(1)], dim = 1)
        
        step = step + 1
 
    result = policy(temporaryStateLine[:, 2:-1].reshape(-1, 106).detach().clone())
    loss = torch.sum(torch.square(result))
    p_loss = loss
    #p_loss = p_loss -reward
    optimizerP.zero_grad()
    p_loss.backward(retain_graph = True)
    # print("stlist",stateList[0:3 , :, 1])
    optimizerP.step()
    if rep%10 == 0:
        torch.save(policy.state_dict(),'PARAM/POLICY/pretrainedPolicytest_.pth') #successful_pretrained
        torch.save(policy.state_dict(),'PARAM/POLICY/pretrainedPolicytest2.pth') #successful_pretrained
    #if rep %10001 == 10000:
    #    torch.save(policy.state_dict(),  'PARAM/POLICY/' + str(skillNum) + "n" + str(epLength) + 'pretrainedPolicy.pth')
    
    print("loss", loss)

    del loss
    del p_loss
    rep = rep + 1
    if rep%1 == 0:
        graphinput = torch.stack((batchedInitialState, batchedWeightList), dim = 1)  # 
        #render.render(graphinput[:, :,-3], graphinput[:, :, -2], graphinput[:, :, -1])
        torch.set_printoptions(threshold=100000)

            
        render.render(temporaryStateLine[:100, :, -3], temporaryStateLine[:100, :, -2], temporaryStateLine[:100, :, -1])
        
