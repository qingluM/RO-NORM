# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 08:50:20 2023

@author: Mxm
"""

import numpy as np
import scipy.io as sio
import torch

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
class L2Loss(object):
    
    def __init__(self, d=2, size_average=True):
        super(L2Loss, self).__init__()
        
        # Dimension and Lp-norm type are postive
        assert d > 0
        self.d = d
        self.p = 2
        self.size_average = size_average


    def rel(self, x, y):
        
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.size_average:
            return torch.mean(diff_norms / y_norms)
        else:
            return torch.sum(diff_norms / y_norms)


    def __call__(self, x, y):
        return self.rel(x, y)
    
def loss_func(y_true, y_pre):
    
    MAX_Error  = np.max(np.abs(y_true - y_pre))
    Mean_MAX   = np.mean(np.max(np.abs(y_true - y_pre), axis = (1,2)))
    Mean_Error = np.mean(np.abs(y_true - y_pre))
    
    myloss = L2Loss(size_average=False)
    lploss = myloss(torch.Tensor(y_true), torch.Tensor(y_pre)).item()/y_true.shape[0]
    result = np.array([MAX_Error, Mean_MAX, Mean_Error, lploss*100])

    return result

if __name__ == '__main__':
    
    n = 5
    k = 1
    BASIS = 'POD'
    casenamelist = ['Layout']
    
    for j in range(len(casenamelist)):
        
        casename = casenamelist[j]
        Results = np.zeros((n,4))
        
        print('\n')
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print(casename)
        for i in range (n):
            
            CaseName = casename+"_"+str(i+1)
            data = sio.loadmat('logs/'+BASIS + '_'+CaseName+'/STFNO_pre_'+CaseName+'.mat')

            y_test    = data['y_test']
            pre_test  = data['pre_test']
            
            Results[i, :] = loss_func(pre_test,y_test)
            
            print('Step: %d, Max error: %.3f, MeanMax error: %.3f, Mean error: %.3f, ReL2 error: %.3f'%(i, Results[i, 0], Results[i, 1], Results[i, 2],Results[i, 3]))
            
        
        std  = np.std (Results, axis=0).reshape((1,-1))
        mean = np.mean(Results, axis=0).reshape((1,-1))
        
        print('-------------------')
        print('Mean:  , Max error: %.3f, MeanMax error: %.3f, Mean error: %.3f, ReL2 error: %.3f'%(mean[:,0], mean[:,1], mean[:,2],mean[:,3]))
        print('Std:   , Max error: %.3f, MeanMax error: %.3f, Mean error: %.3f, ReL2 error: %.3f'%(std[:,0], std[:,1], std[:,2], std[:,3]))
        

        