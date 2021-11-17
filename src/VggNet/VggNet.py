from src.VggNet.util_ import vgg_params 
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms,datasets
import torchvision
import os


configures = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def make_layers(configures,is_batch=True):
  layers=[]
  in_channels=3
  for v in configures:
    if v=="M":
      layers+=[nn.MaxPool2d(kernel_size=2,stride=2)]
    else:
      conv2d=nn.Conv2d(in_channels,v,kernel_size=3,padding=1)
      if is_batch:
        layers+=[conv2d,nn.BatchNorm2d(v),nn.ReLU(True)]
      else:
        layers+=[conv2d,nn.ReLU(True)]
      in_channels=v
  return nn.Sequential(*layers)




class VGG(nn.Module):
  def __init__(self,model,dropout=0.5,num_classes=1000):
    '''
    <args>
      models : 모델의 종류 ex)"vgg11"

    '''
    super(VGG,self).__init__()
    self.model=vgg_params(model)
    self.dropout=dropout # 드랍아웃 비율
    self.num_classes=num_classes # 출력 class 개수
    self.configures=configures[self.model[0]] # A,B,D,E
    self.input_size=self.model[1] # 입력 이미지 크기 224
    self.is_batch=self.model[2] # 배치정규화 여부 True

    self.conv_layer=make_layers(self.configures,self.is_batch)

    self.classifier=nn.Sequential(
        nn.Linear(512*7*7,4096),
        nn.ReLU(True),
        nn.Dropout(self.dropout),
        nn.Linear(4096,4096),
        nn.ReLU(True),
        nn.Dropout(self.dropout),
        nn.Linear(4096,self.num_classes),
        nn.Softmax()
    )
  def forward(self,x):
    x=self.conv_layer(x)
    x=torch.flatten(x)
    return self.classifier(x)


Vgg=VGG("vgg19_bn")

Vgg


