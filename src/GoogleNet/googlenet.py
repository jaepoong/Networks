%cd /content/drive/MyDrive/데이터사이언스/Networks
!pip install import_ipynb

import import_ipynb
from src.GoogleNet.util_ import *
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms,datasets
import torchvision
import os

class BasicConv2d(nn.Module):
  def __init__(self,in_channels,out_channels,**kwarks):
    super(BasicConv2d,self).__init__()
    self.conv=nn.Conv2d(inchannels,out_channels,bias=False,**kwargs)
    self.bn=nn.BatchNorm2d(out_channels)

  def forward(self,x):
    x=self.conv(x)
    x=self.bn(x)
    return nn.ReLU(True)(x)

class Inception(nn.Module):
  def __init__(self,in_channels,ch1,ch3red,ch3,ch5red,ch5,pool_proj,conv_block=None):
    super(Inception,self).__init__()
    if conv_block is None:
      conv_block=BasicConv2d()
    self.branch1=conv_block(in_channels,ch1)
    self.branch2=nn.Sequential(
        conv_block(in_channels,ch3red,kernel_size=1),
        conv_block(ch3red,ch3,kernel_size=3,padding=1)
    )
    self.branch3=nn.Sequential(
        conv_block(in_channels,ch5red,kernel_size=1),
        conv_block(ch5red,ch5,kernel_size=3,padding=1)
    )
    self.branch4=nn.Sequential(
        nn.MaxPool2d(kernel_size=3,stride=1,padding=1,ceil_mode=True),
        conv_block(in_channels,pool_proj,kernel_size=1)
    )
  def forward(self,x):
    branch1=self.branch1(x)
    branch2=self.branch2(x)
    branch3=self.branch3(x)
    branch4=self.branch4(x)
    return torch.cat([branch1,branch2,branch3,branch4],1)


class InceptionAux(nn.Module):
  def __init__(self,in_channels,num_classes,dropout_rate=0.5,conv_block=None):
    super(InceptionAux,self).__init_()
    if conv_block is None:
      conv_block=BasicConv2d
    self.conv=conv_block(in_channels,128,kernel_size=1)

    self.fc1=nn.Linear(2048,1024)
    self.fc2=nn.Linear(1024,num_classes)
  
  def forward(self,x):
    x=nn.AdaptiveAvgPool2d((4,4))(x)
    x=self.conv(x)
    x=torch.flatten(x,1)
    x=self.fc1(x)
    x=nn.ReLU(True)(x)
    x=nn.Dropout(0.5)(x)
    x=self.fc2(x)
    return x

class GoogleNet(nn.Module):
  def __init__(self,input_size=224,aux=True,num_classes=1000,dropout_rate=0.5):
    super(GoogleNet,self).__init__()
    conv_block=BasicConv2d
    inception_block=Inception
    inception_aux_block=InceptionAux
    self.dropout_rate=dropout_rate
    self.input_size=input_size
    self.aux=None
    self.num_classes=num_classes

    self.conv1=conv_block(3,64,kernel_size=7,stride=2,padding=3)
    self.maxpool1=nn.MaxPool2d(3,stride=2,ceil_mode=True)
    self.conv2=conv_block(64,64,kernel_size=1,stride=1)
    self.conv3=nn.conv_block(64,192,kernel_size=3,stride=1,padding=1)
    self.max_pool2=nn.MaxPool2d(3,stride=2,ceil_mode=True)
    self.inception3a=inception_block(192,64,96,128,16,32,32)
    self.inception3b=inception_block(256,128,128,192,32,96,64)
    self.max_pool3=nn.Maxpool2d(3,stride=2,ceil_mode=True)
    self.inception4a=inception_block(480,192,96,208,16,48,64)
    self.inception4b=inception_block(512,160,112,224,24,64,64)
    self.inception4c=inception_block(512,128,128,256,24,64,64)
    self.inception4d=inception_block(512,112,144,288,32,64,64)
    self.inception4e=inception_block(528,256,160,320,32,128,128)
    self.max_pool4=nn.Maxpool2d(3,stride=2,ceil_mode=True)
    self.inception5a=inception_block(832,256,160,320,32,128,128)
    self.inception5b=inception_block(832,384,192,384,48,128,128)
    
    self.avgpool=nn.AdaptiveAvgPool2d((1,1))
    self.dropout=nn.Dropout(p=self.dropout_rate)
    self.fc=nn.Linear(1024,selfnum_classes)

    def _forward(self,x):
      x=self.conv1(x)
      x=self.maxpool1(x)
      x=self.conv2(x)
      x=self.conv3(x)
      x=self.max_pool2(x)
      x=self.inception3a(x)
      x=self.inception3b(x)
      if aux:
        aux1=self.inception_aux_block(480,num_classes=self.num_classes,dropout_rate=self.dropout_rate,cnov_block=self.conv_block)(x)
      else:
        aux1=None
      x=self.inception4a(x)
      x=self.inception4b(x)
      x=self.inception4c(x)
      x=self.inception4d(x)
      x=self.inception4e(x)
      if aux:
        aux2=self.inception_aux_block(832,num_classes=self.num_classes,dropout_rate=self.dropout_rate,cnov_block=self.conv_block)(x)
      else:
        aux2=None
      x=self.inception5a(x)
      x=self.inception5b(x)
      x=self.avgpool(x)
      x=self.dropout(x)
      x=self.fc(x)
      x=nn.Softmax(x)
      return x,aux1,aux2
    
    def forward(self,x):
      x,aux1,aux2=self._forward(x)
      return x



