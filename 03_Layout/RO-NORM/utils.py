
import time
from timeit import default_timer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.optim import Adam
from functools import reduce
import operator
from Adam import Adam

#################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Approximation_block(nn.Module):
    def __init__ (self, in_channels, out_channels, modes, LBO_MATRIX, LBO_INVERSE):
        
        super(Approximation_block, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = LBO_MATRIX.shape[1]
        self.LBO_MATRIX = LBO_MATRIX
        self.LBO_INVERSE = LBO_INVERSE

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.float))

    def forward(self, x):
                        
        ################################################################
        # Encode
        ################################################################
        x = x = x.permute(0, 2, 1)
        x = self.LBO_INVERSE @ x  
        x = x.permute(0, 2, 1)
        
        ################################################################
        # Approximator
        ################################################################
        x = torch.einsum("bix,iox->box", x[:, :], self.weights1)
        
        ################################################################
        # Decode
        ################################################################
        x =  x @ self.LBO_MATRIX.T
        
        return x
    
class NORM_Net(nn.Module):
    def __init__(self, modes, width, channel, LBO_MATRIX, LBO_INVERSE):
        super(NORM_Net, self).__init__()

        self.modes1 = modes
        self.width = width
        self.channel = channel
        self.padding = 2 
        self.fc0 = nn.Linear(1, self.width) 
        self.LBO_MATRIX = LBO_MATRIX
        self.LBO_INVERSE = LBO_INVERSE

        self.conv0 = Approximation_block(self.width, self.width, self.modes1, self.LBO_MATRIX, self.LBO_INVERSE )
        self.conv1 = Approximation_block(self.width, self.width, self.modes1, self.LBO_MATRIX, self.LBO_INVERSE )
        self.conv2 = Approximation_block(self.width, self.width, self.modes1, self.LBO_MATRIX, self.LBO_INVERSE )
        self.conv3 = Approximation_block(self.width, self.width, self.modes1, self.LBO_MATRIX, self.LBO_INVERSE )

        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 32)
        self.fc2 = nn.Linear(32, self.channel)

    def forward(self, x):

        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
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
        mymin = torch.min(x, 0)[0].view(-1)
        print('Min',mymin.shape)
        mymax = torch.max(x, 0)[0].view(-1)
        print('Max',mymax.shape)

        self.a = (high - low)/(mymax - mymin)
        self.b = -self.a*mymax + high

    def encode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = self.a*x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.b)/self.a
        x = x.view(s)
        return x
    
def get_parameter_number(net):
    
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return  trainable_num
