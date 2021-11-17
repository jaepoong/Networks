import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def conv3(in_channels,out_channels,stride=1):
    return nn.Conv2d(in_channels=in_channels,out_channels=out_channels,stride=stride,kernel_size=3)

def conv1(in_channels,out_channels,stride=1):
    return nn.Conv2d(in_channels=in_channels,out_channels=out_channels,stride=1,kernel_size=1)

class BasicBlock(nn.Module):
    
    def __init__(in_channels,out_channels, stride=1,downsample=None):
        super().__init__()
        self.stride=stride
        self.conv1=conv3*3(in_channels,out_channels,stride=stride)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU(inplace=True)
        self.conv2=conv3*3(out_channels,out_channels,stride=stride)
        self.bn2=nn.BatchNorm2d(out_channels)
        self.downsample=downsample
        
    def forward(self,x):
        identity=x
        
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.conv2(x)
        out=self.bn2(out)
        
        if self.downsample is not None: #  다운셈플링 진행
            identity=self.downsample(x)
        out += identity # shortcut
        out=self.relu(out)
        return out

class BottleNeck(nn.Module):
    def __init__(in_channels,out_channels,stride=1,downsample=None):
        super().__init()
        self.conv1=conv1(in_channels,out_channles,stride=stride)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.conv2=conv3(out_channels,out_channels,stride=stride)
        self.bn2=nn.BatchNorm2d(out_channels)
        self.conv3=conv1(out_channels,out_channels,stride=stride)
        self.bn3=nn.BatchNorm2d(out_channels)
        self.downsamplt=downsample
        self.stride=stride
    
    def forward(self,x):
        identity=x
        
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)
        
        out=self.conv2(out)
        out=self.bn2(out)
        out=self.relu(out)
        
        out=self.conv3(out)
        out=self.bn3(out)
        
        if self.downsample is not None:
            identity=self.downsample(x)
        
        out+=identity
        out=self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self,block=None,layers=None,num_classes=1000):
        super().__init__()
        if block==None:
            self.block=BasicBlock
        self.norm=nn.BatchNorm2d
        
        conv1=nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3)
        self.bn1=norm(64)
        self.relu=nn.ReLU()
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        
        
    def make_layer(self,block,out_channels,in_channels,stride=1):
        norm_layer=self.norm
        downsample=None
        if stride !=1 or self.
