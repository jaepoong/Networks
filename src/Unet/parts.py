import torch
import torch.nn as nn
import torch.nn.functional as F
class Conv(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,kernel_size=3,padding=None):
        super().__init__()
        
    def forward(self,x):
        x=nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=None)(x)
        x=nn.BatchNorm2d(out_channels)(x)
        x=nn.ReLU(inplace=True)(x)
        return x

class Downsampling(nn.Module):
    def __init__(self,in_channels,out_channels,mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels=out_channels
        
    def forward(self,x):
        x=nn.MaxPool2d(kernel_size=2,stride=2)(x)
        x=Conv(in_channels,mid_channels)(x)
        x=Conv(mid_channels,out_channels)
            
class Upsampling(nn.Module):
    def __init__(self,in_channels,out_channels,mid_channels=None):
        super().__init__()
        self.up=nn.ConvTranspose2d(in_channels,in_channels//2,kernel_size=2,stride=2)
        if not mid_channels:
            mid_channels=out_channels 
        conv=nn.Sequential(Conv(in_channels,mid_channels),
                           Conv(mid_channels,out_channels))
    def forward(self,x1,x2):
        x1=self.up(x1)
        diffy=x2.size()[2]-x1.size()[2]
        diffx=x2.size()[3]-x1.size()[3]
        
        x1=F.pad(x1,[diffx//2,diffx-diffx//2,
                     diffy//2,diffy-diffy//2])
        x=torch.cat([x2,x1],dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(OutConv,self).__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size=1)
    
    def forward(self,x):
        return self.conv(x)