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

FEM = 1
save = 1

Itr = 4000
Nx = 415 # 
Nt = 100 # 


U_initial = np.ones((Itr, Nx))
U_field   = np.ones((Itr, Nx, Nt))

if FEM:
    
    Time1 = time.time()    
    for i in range(Itr):
        
        s = 96
        x = np.linspace(0, 1, s)
        X, Y = np.meshgrid(x,x)

        # regular initial condition field
        Ini_field_u = np.zeros((s, s))
        device = torch.device('cpu')
        GRF = GaussianRF(2, s, alpha=3, tau=3, device=device)
        U = GRF.sample(1) 
        Ini_field_u = U[0].numpy()
        

        Ini_data_u = np.concatenate((X.reshape(-1,1),Y.reshape(-1,1),Ini_field_u.reshape(-1,1)),axis = 1)
        np.savetxt("Ini_data_u.csv", Ini_data_u)
        print("New initial field generated!")
        
        print(" FEM  calculating  ...")
        client = mph.start(cores=4,version='6.0')
        model = client.load('Burgers.mph')
        
        Time2 = time.time()
        model.build()
        model.mesh()
        model.solve('研究 1')
        Time3 = time.time()
        
        print('### %d in %d ###' %(i, Itr))
        print('Time:%.2f秒' % ((Time3 - Time2)))
        print('Total time:%.2f秒' % ((Time3 - Time1)))
        
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
    
        U_field[itr] = data1[:,3:]
        U_initial[itr] = data1[:,2]
        

    sio.savemat('data/Burgers.mat', 
                 { 'nodes':      nodes,
                   'elements':   elements,
                   'U_initial':  U_initial,
                   'U_field': U_field,
                   'x': nodes[:,0],
                   'y': nodes[:,1]})              
             









