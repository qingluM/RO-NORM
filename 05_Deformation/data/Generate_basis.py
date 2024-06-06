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

    
class MinMax(object):
    def __init__(self, x, low=0.0, high=1.0):
        super(MinMax, self).__init__()

        mymin = np.min(x)
        mymax = np.max(x)

        self.a = (high - low) / (mymax - mymin)
        self.b = -self.a * mymax + high

    def encode(self, x):
        s = x.shape
        x = x.reshape(s[0], -1)
        x = self.a * x + self.b
        x = x.reshape(s)
        return x

    def decode(self, x):
        x.astype(np.float32)
        s = x.shape
        x = x.reshape(s[0], -1)
        x = (x - self.b) / self.a
        x = x.reshape(s)
        return x
    

Data = sio.loadmat("Qianyuan.mat")

dataIn = Data['T_field']
ntrain = 500
ntest  = 100
kmodes = 128


POD_norm = 1
LBO = 1

dataIn = dataIn.transpose(0,2,1)


# In[]

nodes = pd.read_csv('../coordinates.csv',header=None).values
elements = pd.read_csv('../elements.csv',header=None).values

num = dataIn.shape[0]
nx  = dataIn.shape[1]
nt  = dataIn.shape[2]

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
        tetra = TetMesh(nodes,elements-1)
        fem = Solver(tetra)
        evals, evecs = fem.eigs(k=kmodes)
        
        print("The shape of LBO basis: ", evecs.shape)
        evDict = {'LBO_basis' : evecs}
        sio.savemat('LBO_basis.mat', evDict)


    
if POD_norm:
    
    norm_y  = MinMax(data_train)
    data_train_norm = norm_y.encode(data_train)
    
    U = np.reshape(data_train_norm, (-1, nt))
    R = 1./(ntrain*nx-1)*np.matmul(U.T, U)
    eig_value, eig_vector = np.linalg.eigh(R)
    
    eig_vector = np.fliplr(eig_vector) 
    t_eig_value  = eig_value[::-1].reshape(-1,1)
    t_eig_vector = eig_vector[:,0:kmodes]
    
    print("The shape of POD basis: ", t_eig_vector.shape)
    
    evDict = {'POD_value' : t_eig_value,
              'POD_basis': t_eig_vector}
    
    sio.savemat('POD_basis_norm.mat', evDict)
    

