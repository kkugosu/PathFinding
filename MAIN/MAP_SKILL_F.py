# -*- coding: utf-8 -*-
from enum import NAMED_FLAGS
from turtle import update
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
from UTIL import render, render_multi
from MAP import mapgen
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
skillNum = 2# Initialize.trainingSkillNum
mapNum = 97
epLength = 3 #Initialize.el
embeddingSpaceM = Initialize.mapEmbeddingSpace
policy = Initialize.globalTransformer
optimizerP = Initialize.optimizer_gp

dist_by_coordinate = lambda i, j : (torch.sum((i-j)**2, -1))**(1/2)
angle_to_coord = lambda i: torch.stack((torch.cos(i[:,0])*torch.sin(i[:,1]), torch.sin(i[:,0])*torch.sin(i[:,1]), torch.cos(i[:,1])), dim=1)

cusk = 0 # current updat0ng skill
training_recur = 100000

# load model
load = 0
if load == 1:
    print("load")
    policy.load_state_dict(torch.load( 'PARAM/POLICY/Policytest_.pth'))

rep = 0
while rep < training_recur:
    if rep%1000 == 0:
        print("reset")
        indices = torch.arange(0, skillNum, dtype=torch.float)
        phi = torch.tensor([np.pi/2, np.pi/2])#torch.arccos(1 - 2 * indices / skillNum)
        theta = torch.tensor([np.pi, 2*np.pi])#np.pi * (1 + 5**0.5) * indices
        x = torch.cos(theta) * torch.sin(phi)
        y = torch.sin(theta) * torch.sin(phi)
        z = torch.cos(phi)
        print(x)
        print(y)
        print("z=",z)
        myPoints = torch.stack((x, y, z), dim=1).to(device)
        
        myPoints = torch.tensor([[ -1.0000e+00,  0.0000e+00,  0.0000e+00],
        #[0e-01, 0e-01,  -1e-00],
        #[1e-00,  0e-01,  0e-01],
        #[ -1e-00, 0e-01, 0e-01],
        #[0e-01,  1e-00, 0e-01],
        [1e-00,  0e-00, 0e-00]], device='cuda:0')

        batchedWeightList = myPoints.unsqueeze(0).repeat(mapNum, 1, 1)
        #19,3

        generated_map = mapgen.gen(mapNum)# fix the map
        output, mu = thisvae(generated_map*2-1) #0~1 -> -1 ~ 1 # 97,100
        originalinput = mu.unsqueeze(1).repeat(1, skillNum, 1).clone()
        batchedMap = originalinput # 97, 19, 100
        #19,100

        batchedInitialState = torch.zeros(skillNum, 3, requires_grad = True).unsqueeze(0).repeat(mapNum, 1, 1).to(device) #+ batchedWeightList*0.01
        #19,3

    tmpInput = torch.cat((batchedWeightList, batchedMap, batchedInitialState), dim = -1).reshape(-1, 106) # 19, 106 ##
    temporaryStateLine = torch.zeros(skillNum * mapNum, 1, 106, requires_grad = True).to(device) # batchsize = skillnum, el, featurenum
    stateList = torch.zeros(skillNum * mapNum, epLength + 1, 3, requires_grad = True).to(device)
    pre_output1 = theta
    pre_output2 = phi
    pre_output = torch.stack([pre_output1, pre_output2], dim = 1).to(device).repeat(mapNum, 1)
    angle_ary = pre_output.unsqueeze(1)
    print("presize", pre_output.size())
    #9999, el, 106
    step = 0
    while step < epLength:

        output = policy(tmpInput.detach().clone())
        output = torch.atan(output) # -pi/2 ~ pi/2
        tmpState = angle_to_coord(output + pre_output.detach().clone())
        pre_output = output + pre_output
        angle_ary = torch.cat((angle_ary, pre_output.unsqueeze(1)), dim = 1)
        stateList[:, step+1] = stateList[:, step] + tmpState
        tmpInput = torch.cat((batchedWeightList.reshape(-1, 3), batchedMap.reshape(-1, 100), tmpState ), dim = 1) + temporaryStateLine[:, -1]
        temporaryStateLine = torch.cat([temporaryStateLine,  tmpInput.unsqueeze(1)], dim = 1)
        
        step = step + 1
    angle_ary = angle_ary[:, :-1].detach().clone()
    
    pg_input = temporaryStateLine[:, :-1].reshape(-1, 106).detach().clone() # 101*10, 106
    pg_output = policy(pg_input)
    input_ary = angle_ary.reshape(-1, 2)
    print(input_ary.size())
    print("addition", pg_output.size(), pg_output[:6])
    updateRate = angle_to_coord(pg_output + input_ary) # 1010, 3
    
    updateRate = updateRate.reshape(mapNum, skillNum, epLength, 3) # 
    updateResult = stateList[:, :-1].reshape(mapNum, skillNum, epLength, 3) + updateRate # 1010,3 
    updateResult = torch.cat((stateList[:, 0].reshape(mapNum, skillNum, 1, 3), updateResult), dim = 2)
    updateResult = updateResult.reshape(mapNum, -1, 3) ##

    #print("upresult", updateResult.size(), updateResult[0])
    updateMatrix = torch.sum(torch.square(updateResult[:, None, :, :] - updateResult[:, :, None, :]), -1)
    torch.set_printoptions(threshold=100000000)
    #print("updmat",updateMatrix.size(), updateMatrix[0])
    ##
    for i in range(skillNum):
        updateMatrix[:, i*(epLength+1) : (i+1)*(epLength+1), i*(epLength+1) : (i+1)*(epLength+1)] = 0
    #print("updmat2",updateMatrix.size(), updateMatrix[0])
    reward = torch.sum(torch.square(1-torch.exp(-updateMatrix/var)))
    p_loss = -reward
    print(p_loss)
    
    torch.set_printoptions(threshold=100000000)
    
    optimizerP.zero_grad()
    p_loss.backward(retain_graph = True)
    # print("stlist",stateList[0:3 , :, 1])
    optimizerP.step()
    torch.save(policy.state_dict(),'PARAM/POLICY/Policytest_.pth') #successful_pretrained
    torch.save(policy.state_dict(),'PARAM/POLICY/Policytest.pth') #successful_pretrained
    if rep %10001 == 10000:
        torch.save(policy.state_dict(),  'PARAM/POLICY/' + str(skillNum) + "n" + str(epLength) + 'pretrainedPolicy.pth')

    del p_loss
    rep = rep + 1
    if rep%100 == 0:
        graphinput = torch.stack((batchedInitialState, batchedWeightList), dim = 1)  # 
        #render.render(graphinput[:, :,-3], graphinput[:, :, -2], graphinput[:, :, -1])
        torch.set_printoptions(threshold=100000)
        inputline = updateResult.reshape(mapNum, skillNum, epLength+1, 3).cpu().detach().clone()#pg_input.reshape(mapNum, skillNum, epLength, 106)
        #print("inputline size",updateResult[0])
        #print("inputline size2",inputline[0])
        render_multi.render(generated_map, inputline[:, :, :, -3], inputline[:, :, :, -2], inputline[:, :, :, -1])
        

