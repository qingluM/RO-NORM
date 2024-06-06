
import time
from timeit import default_timer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from functools import reduce
import operator

#################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

################################################################
#  1d fourier layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  
        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    def compl_mul1d(self, input, weights):

        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x) 
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNO1d(nn.Module):
    def __init__(self, Fmodes,width,modes):
        super(FNO1d, self).__init__()
        
        self.modes1 = Fmodes
        self.width = width
        self.channel = modes
        self.padding = 2 
        self.fc0 = nn.Linear(6, self.width) 

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.width)
        self.fc3 = nn.Linear(self.width,self.channel)

    def forward(self, x):
        
        x = self.fc0(x) 

        x = x.permute(0, 2, 1) 
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = torch.relu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = torch.relu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = torch.relu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2 

        x = x.permute(0, 2, 1) 
        x = self.fc1(x)
        x = torch.relu(x)
        
        x = self.fc2(x) 
        x = self.fc3(x) 
        
        return x

class Z_score(object):
    def __init__(self, x, eps=0.00001):
        super(Z_score, self).__init__()
        
        # x could be in shape of ntrain*time or ntrain*time*nodes
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x):
        
        std = self.std + self.eps 
        mean = self.mean

        x = (x * std) + mean
        return x

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

class MinMax(object):
    def __init__(self, x, low=0.0, high=1.0):
        super(MinMax, self).__init__()

        # mymin = torch.min(x,0).view(-1)
        # mymax = torch.max(x,0).view(-1)
        
        mymin = torch.min(x)
        mymax = torch.max(x)

        self.a = (high - low) / (mymax - mymin)
        self.b = -self.a * mymax + high

    def encode(self, x):
        s = x.shape
        x = x.reshape(s[0], -1)
        x = self.a * x + self.b
        x = x.reshape(s)
        return x

    def decode(self, x):

        s = x.shape
        x = x.reshape(s[0], -1)
        x = (x - self.b) / self.a
        x = x.reshape(s)
        return x
    
def get_parameter_number(net):
    
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return  trainable_num