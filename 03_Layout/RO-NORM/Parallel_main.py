# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 14:11:57 2023

@author: Mxm
"""
import os
import torch
import time
import scipy.io as sio
import numpy as np
import pandas as pd
from utils import NORM_Net,Z_score, L2Loss, get_parameter_number
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def main(args):
    
    print("\n=============================")
    print("torch.cuda.is_available(): " + str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        print("torch.cuda.get_device_name(0): " + str(torch.cuda.get_device_name(0)))
    print("=============================\n")
    
    path = args.data_dir
    
    CaseName = args.casename
    
    learning_rate = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    
    ntrain = args.num_train
    ntest = args.num_test
    
    modes  = args.modes     
    Lmodes = args.LBOmodes  
    width  =  args.width
    
    BASIS = args.BASIS
    
    step_size = 100
    gamma = 0.5
    
    Train_Data = sio.loadmat(path)
    
    y_dataIn = Train_Data['T_field']
    x_dataIn = Train_Data['layout']


    nt  = y_dataIn.shape[2]
    nx  = y_dataIn.shape[1]

    POD_BASIS = sio.loadmat('../data/POD_basis_norm.mat') 
    LBO_BASIS = sio.loadmat('../data/LBO_basis.mat')

    LBO_basis = LBO_BASIS['LBO_basis']
    POD_basis = POD_BASIS['POD_basis']
    POD_value = POD_BASIS['POD_value']
    
    LBO_MATRIX = torch.Tensor(LBO_basis[:,:Lmodes]).cuda()
    LBO_INVERSE = (LBO_MATRIX.T@LBO_MATRIX).inverse()@LBO_MATRIX.T 
    
    print("\n======The data have been imported=======\n")
    
    spatial_basis = np.zeros((modes,nt))
    if BASIS == 'POD':
        
        power = np.sum(POD_value[:modes])/np.sum(POD_value[:])
        print('Energy proportion:',np.round(power,8),'%')
        
        for j in range(modes):
            spatial_basis[j] = POD_basis[:,j]
       
        spatial_basis = torch.Tensor(spatial_basis).cuda()
        
        BASE_MATRIX = POD_basis[:,:modes].copy()
        BASE_MATRIX = torch.Tensor(BASE_MATRIX).cuda()
        BASE_INVERSE = torch.linalg.pinv(BASE_MATRIX)
        print("\n===== POD basis calculation complete=====\n")
    
    if BASIS == 'LBO':
        for j in range(modes):
            spatial_basis[j] = LBO_MATRIX[:,j]
        
        spatial_basis = torch.Tensor(spatial_basis).cuda()
        
        BASE_MATRIX = LBO_MATRIX[:,:modes].copy()
        BASE_MATRIX = torch.Tensor(BASE_MATRIX).cuda()
        BASE_INVERSE = (BASE_MATRIX.T@BASE_MATRIX).inverse()@BASE_MATRIX.T 

        print("\n======LBO basis calculation complete====\n")
    
    x_train = torch.Tensor(x_dataIn[:ntrain,:].reshape(ntrain,-1,1))
    y_train = torch.Tensor(y_dataIn[:ntrain,:,:])
    x_test  = torch.Tensor(x_dataIn[-ntest:,:].reshape(ntest,-1,1))
    y_test  = torch.Tensor(y_dataIn[-ntest:,:,:])
    
    norm_x  = Z_score(x_train)
    x_train = norm_x.encode(x_train)
    x_test  = norm_x.encode(x_test)
    
    norm_y  = Z_score(y_train)
    y_train = norm_y.encode(y_train)
    y_test  = norm_y.encode(y_test)
    

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), 
                                               batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), 
                                              batch_size=batch_size, shuffle=False)


    weight_model = NORM_Net(Lmodes,width,modes,LBO_MATRIX,LBO_INVERSE).cuda()
    ################################################################
    # training and evaluation
    ################################################################
    optimizer = torch.optim.Adam(weight_model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    myloss = L2Loss(size_average=False)
    
    time_start = time.perf_counter()
    time_step = time.perf_counter()
    
    train_error = np.zeros((epochs))
    test_error = np.zeros((epochs))
    ET_list = np.zeros((epochs))
    
    for ep in range(epochs):
        
        weight_model.train()

        train_l2 = 0
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
    
            optimizer.zero_grad()
            out_weight = weight_model(x)
            out  = torch.einsum('ntm,mx->ntx',out_weight,spatial_basis).cuda()
            l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
            l2.backward() 
        
            
            out_real = norm_y.decode(out.cpu())
            y_real = norm_y.decode(y.cpu())
            
            train_l2 += myloss(out_real, y_real).item()   
    
            optimizer.step()

        scheduler.step()
        weight_model.eval()
        test_l2 = 0.0
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()
    
                out = weight_model(x)
                out  = torch.einsum('ntm,mx->ntx',out,spatial_basis).cuda()
                out_real = norm_y.decode(out.cpu())
                y_real = norm_y.decode(y.cpu())
                
                test_l2 += myloss(out_real, y_real).item()                
                loss_max_test = (abs(out.view(batch_size, -1)- y.view(batch_size, -1))).max(axis=1).values.mean()
    
        train_l2 /= ntrain
        test_l2 /= ntest
        train_error[ep] = train_l2
        test_error[ep] = test_l2
        
        ET_list[ep] = loss_max_test
        time_step_end = time.perf_counter()
        T = time_step_end - time_step

        print('Step: %d, Train L2: %.5f, Test L2: %.5f, Emax_test: %.5f, Time: %.3fs'%(ep, train_l2, test_l2, loss_max_test, T))
        time_step = time.perf_counter()
    
    print("\n=============================")
    print("Training done...")
    print("=============================\n")
    
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), 
                                              batch_size=1, shuffle=False)
    pre_test = torch.zeros(y_test.shape)
    y_test   = torch.zeros(y_test.shape)
    x_test   = torch.zeros(x_test.shape)
    
    index = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            out = weight_model(x)
            out  = torch.einsum('ntm,mx->ntx',out,spatial_basis)
            
            out_real = norm_y.decode(out.cpu())
            y_real   = norm_y.decode(y.cpu())
            x_real   = norm_x.decode(x.cpu())
            pre_test[index,:] = out_real
            y_test[index,:] = y_real
            x_test[index,:] = x_real
            
            index = index + 1
            
    Test_error  = np.abs(pre_test.cpu().detach().numpy() - y_test.cpu().detach().numpy())
    T_max_t = np.max(Test_error)
    T_max = np.max(Test_error[:, :, :], axis=(1, 2))
    T_mean = np.mean(Test_error[:, :, :], axis=(1, 2))
    
    print('\nMaxMax:', np.round(T_max_t, 3))
    print('\nMeanMax:', np.round(np.mean(T_max), 3), 'Std:', np.round(np.std(T_max),3))
    print('MeanError:', np.round(np.mean(T_mean), 3), 'Std:', np.round(np.std(T_mean), 3))
    
    # ================ Save Data ====================
    current_directory = os.getcwd()
    sava_path = current_directory + "/logs/" + BASIS +"_"+ CaseName + "/"
    if not os.path.exists(sava_path):
        os.makedirs(sava_path)
    
    dataframe = pd.DataFrame({'Test_loss': [test_l2],
                              'num_paras': [get_parameter_number(weight_model)],
                              'train_time':[time_step_end - time_start]})
    
    dataframe.to_csv(sava_path + 'log.csv', index = False, sep = ',')
    
    loss_dict = {'train_error' :train_error,
                 'test_error'  :test_error}
    
    pred_dict = {'pre_test' : pre_test.cpu().detach().numpy(),
                    'x_test'   : x_test.cpu().detach().numpy(),
                    'y_test'   : y_test.cpu().detach().numpy(),
                    }
    
    sio.savemat(sava_path +'STFNO_loss_' + CaseName + '.mat', mdict = loss_dict)                                                     
    sio.savemat(sava_path +'STFNO_pre_'  + CaseName + '.mat', mdict = pred_dict)
    
    print('\nTesting error: %.3e'%(test_l2))
    print('Training time: %.3f'%(time_step_end - time_start))
    print('Num of paras : %d'%(get_parameter_number(weight_model)))
        


if __name__ == "__main__":
    
    class objectview(object):
        def __init__(self, d):
            self.__dict__ = d
            
    for i in range(5):
        
        i = i + 1
        for args in [
                
                { 'modes': 8, 
                  'LBOmodes' :128,
                  'width': 16,
                  'batch_size': 50, 
                  'epochs': 500,
                  'data_dir': '../data/Layout', 
                  'num_train': 1000, 
                  'num_test': 200,
                  'casename': 'Layout_'+str(i),
                   'BASIS':'POD',
                  'lr' : 0.01},
             ]:
            
            args = objectview(args)
                    
        main(args)