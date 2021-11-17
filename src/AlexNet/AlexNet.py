import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms,datasets
import torchvision
import os

class AlexNet(nn.Module):
  def __init__(self,num_classes=10):
    super(AlexNet,self).__init__()
    self.num_classes=num_classes

    self.net=nn.Sequential(
        nn.Conv2d(3,96,kernel_size=11,stride=4,padding=0),#1
        nn.ReLU(inpalce=True),
        nn.MaxPool2d(kernel_size=3,stride=2),
        nn.BatchNorm2d(96),
        nn.Conv2d(96,256,kernel_size=5,stride=1,padding=2),#2
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3,stride=2),
        nn.BatchNorm2d(256),
        nn.Conv2d(256,384,kernel_size=3,stride=1,padding=1),#3
        nn.ReLU(inplace=True),
        nn.Conv2d(384,384,kernel_size=3,stride=1,padding=1),#4
        nn.ReLU(inplace=True),
        nn.Conv2d(384,256,kernel_size=3,stride=1,padding=1),#5
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3,stride=2)
    )
    self.fc=nn.Sequential(
        nn.Dropout(p=0.5,inplace=True),
        nn.Linear(256*6*6,4096),
        nn.ReLU(),
        nn.Dropout(p=0.5,inplace=True),
        nn.Linear(4096,4096),
        nn.ReLU(),
        nn.Linear(4096,self.num_classes)
    )
    def forward(self,input):
      output=self.net(input)
      output=output.view(-1,256*6*6)
      return self.fc(output)


