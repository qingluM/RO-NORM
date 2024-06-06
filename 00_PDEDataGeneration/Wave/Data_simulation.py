# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 15:30:25 2023

@author: Mxm
"""

import scipy.io as sio
import pandas as pd 
import numpy as np
import matplotlib
import os
import matplotlib.pyplot as plt
import random
import time
import warnings
import mph
import math
from random_fields1 import GaussianRF
import torch
import matplotlib.tri as mtri

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# In[]

FEM = 2
save = 1

Itr = 2000
Nx = 506 
Nt = 100 

U_source = np.ones((Itr, Nt))
U_field = np.ones((Itr, Nt, Nx))

if FEM:
    
    Time1 = time.time()    
    for i in range(Itr):
        
        s = 100
        x = np.linspace(0, 5, 100)


        Input_u = np.zeros((s, 1))
        device = torch.device('cpu')
        GRF = GaussianRF(1, s, alpha=3.5, tau=5, device=device)
        U = GRF.sample(1) 
        
        Input_u = U[0].numpy()
        
        min_u = np.min(Input_u)
        max_u = np.max(Input_u)
        mean = 0.5*(max_u+min_u)
        
        U_source[i] = Input_u-mean
        

        Ini_data_u = np.concatenate((x.reshape(-1,1),(Input_u-mean).reshape(-1,1)),axis = 1)
        np.savetxt("Input.csv", Ini_data_u)
        print("New initial field generated!")
        
        print(" FEM  calculating  ...")
        client = mph.start(cores=6,version='6.0')
        model = client.load('WaveEquation.mph')
        
        Time2 = time.time()
        model.build()
        model.mesh()
        model.solve('研究 1')
        Time3 = time.time()
        
        print('### %d in %d ###' %(i, Itr))
        print('仿真时间:%.2f秒' % ((Time3 - Time2)))
        print('总计时间:%.2f秒' % ((Time3 - Time1)))
        
        model.export('数据 1', 'tempdata/U_'+str(i+1)+'.csv')
        
        client.remove(model)
        client.clear()

if save:
    
    probe3 = pd.read_table('coordinates.csv', header=None, sep=',')       
    nodes = probe3.to_numpy()
    
    probe4 = pd.read_table('elements.csv', header=None, sep=',')       
    elements = probe4.to_numpy()
    
    for itr in range(Itr):
        
        print('Epoch:',itr)
        data1 = pd.read_csv('tempdata/U_'+str(itr+1)+'.csv', header=None).values
    
        U_field[itr] = data1[:,2:].T
        

    sio.savemat('data/Wave.mat', 
                 { 'nodes':      nodes,
                   'elements':   elements,
                   'U_source':  U_source,
                   'U_field': U_field,
                   'x': nodes[:,0],
                   'y': nodes[:,1]})              
             









