# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 16:17:28 2023

@author: Mxm
"""

#用于生成各种类型的基，提前保存好

import numpy as np
import scipy.io as sio
from lapy import TriaMesh,Solver,TetMesh
import pandas as pd

class Z_score(object):
    def __init__(self, x, eps=0.00001):
        super(Z_score, self).__init__()
        
        self.mean = np.mean(x, 0)
        self.std = np.std(x, 0)
        self.eps = eps

    def encode(self, x):
        
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x):
        
        std = self.std + self.eps 
        mean = self.mean

        x = (x * std) + mean
        return x    
    

    
Data = sio.loadmat("Wave.mat")

dataIn = Data['U_field']
ntrain = 1500
ntest  = 500
kmodes = 256

POD_norm = 0
LBO = 0


# In[]

nodes = pd.read_csv('../coordinates.csv',header=None).values
elements = pd.read_csv('../elements.csv',header=None).values

num = dataIn.shape[0]
nt  = dataIn.shape[1]
nx  = dataIn.shape[2]

data_train = dataIn[:ntrain,:]
data_test = dataIn[-ntest:,:]

if LBO:
    
    if (nodes.shape[1]==2):
        
        s = nodes.shape[0]
        Points = np.vstack((nodes.T, np.zeros(s).reshape(1,-1)))
        mesh = TriaMesh(Points.T,elements.T-1)
        fem = Solver(mesh)
        evals, LBO_MATRIX = fem.eigs(k=kmodes)
        
        print("The shape of LBO basis: ", LBO_MATRIX.shape)
        evDict = {'LBO_basis' : LBO_MATRIX}
        sio.savemat('LBO_basis.mat', evDict)
    
    if (nodes.shape[1]==3):
        
        s = nodes.shape[0]
        tetra = TetMesh(nodes,elements)
        fem = Solver(tetra)
        evals, evecs = fem.eigs(k=kmodes)
        
        print("The shape of LBO basis: ", LBO_MATRIX.shape)
        evDict = {'LBO_basis' : evecs}
        sio.savemat('LBO_basis.mat', evDict)

    
    
if POD_norm:
    
    norm_y  = Z_score(data_train)
    data_train_norm = norm_y.encode(data_train)
    
    U = np.reshape(data_train_norm, (-1, nx))
    R = 1./(ntrain*nt-1)*np.matmul(U.T, U)
    eig_value, eig_vector = np.linalg.eigh(R)
    
    eig_vector = np.fliplr(eig_vector) 
    t_eig_value  = eig_value[::-1].reshape(-1,1)
    t_eig_vector = eig_vector[:,0:kmodes]
    
    print("The shape of POD basis: ", t_eig_vector.shape)
    
    evDict = {'POD_value' : t_eig_value,
              'POD_basis': t_eig_vector}
    
    sio.savemat('POD_basis_norm.mat', evDict)
    

