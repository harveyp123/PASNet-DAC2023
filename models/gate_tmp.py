import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import logging


def x2act(input, weight, bias, weight_cale):
    '''
    Applies the x^2 Unit (x2act) function element-wise:
        x2act(x) = a*x^2+b*x+c
    '''
    return weight_cale * weight[0] * torch.mul(input, input) + weight[1] * input + bias

class PolyAct(nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.poly_weight = nn.Parameter(data=torch.Tensor([0.005, 1]), requires_grad=True)
        self.poly_bias = nn.Parameter(data=torch.Tensor([0]), requires_grad=bias)
    def forward(self, input, weight_scale = 1):
        return x2act(input, self.poly_weight, self.poly_bias, weight_scale) 
    
class GateAct(nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.alpha_act = nn.Parameter(data=torch.Tensor([0.0, 0.0]), requires_grad=True)
        self.relu = nn.ReLU(inplace=True)
        self.poly = PolyAct(bias)
        self.lat_act = nn.Parameter(data=torch.Tensor([0.0, 0.0]), requires_grad=False)
        self.gate = True
        self.sftmx = torch.nn.Softmax(dim=0)
        self.weight_scale = 1

    def forward(self, input):

        self.lat_act.data[0]  = 1360 * 0.01 * input.nelement()/input.size()[0] + 0.4
        self.lat_act.data[1] = 3 * 0.01 * input.nelement()/input.size()[0] + 0.2

        if self.gate: 
            c = self.sftmx(self.alpha_act)
            r1 = self.relu(input)
            r2 = self.poly(input, self.weight_scale)
            pass
        else: 
            with torch.no_grad():
                c = torch.Tensor([0.0, 0.0])
                sel = torch.argmax(self.alpha_act)
                c[sel] = 1
            r1 = self.relu(input)
            r2 = self.poly(input, self.weight_scale)
            
            # if sel.item() == 0:
            #     print("!!!!!!!!!!!!!!!!!", input.nelement()/input.size()[0])

        return c[0] * r1 + c[1] * r2

class GatePool(nn.Module):
    def __init__(self, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.alpha_pool = nn.Parameter(data=torch.Tensor([0.0, 0.0]), requires_grad=True)
        self.MaxPool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        self.AvgPool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        self.lat_pool = nn.Parameter(data=torch.Tensor([0.0, 0.0]), requires_grad=False)
        self.gate = True
        self.sftmx = torch.nn.Softmax(dim=0)

    def forward(self, input):

        self.lat_pool.data[0] = (1360 * 0.01 * input.nelement()/input.size()[0] + 0.4)* 0.75
        self.lat_pool.data[1] = 0.01 * input.nelement()/input.size()[0]

        if self.gate: 
            c = self.sftmx(self.alpha_pool)
            r1 = self.MaxPool(input)
            r2 = self.AvgPool(input)
            pass
        else: 
            with torch.no_grad():
                c = torch.Tensor([0.0, 0.0])
                sel = torch.argmax(self.alpha_pool)
                c[sel] = 1
            r1 = self.MaxPool(input)
            r2 = self.AvgPool(input)
        
        return c[0] * r1 + c[1] * r2
