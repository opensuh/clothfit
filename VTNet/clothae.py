
from lib2to3.pgen2.pgen import NFAState
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .ae_parts import *

class ClothEncoder(nn.Module):
    def __init__(self, image_size=512, kernel_size=3, n_filters=32, dropout=False, drop_prob=0.2):
        super(ClothEncoder, self).__init__()
        self.image_size = image_size
        self.n_filters = n_filters

        self.conv1 = nn.Sequential(
                nn.Conv2d(1, n_filters, kernel_size=(kernel_size,kernel_size), padding=(kernel_size//2,kernel_size//2)),
                nn.BatchNorm2d(n_filters, track_running_stats=False),
                nn.ReLU(inplace = True),
        )
        self.pool = nn.MaxPool2d((2,2),(2,2))
        self.conv2 = nn.Sequential(
                nn.Conv2d(n_filters, n_filters*2, kernel_size=(kernel_size,kernel_size), padding=(kernel_size//2,kernel_size//2)),
                nn.BatchNorm2d(n_filters*2, track_running_stats=False),
                nn.ReLU(inplace = True),
        )
        self.conv3 = nn.Sequential(
                nn.Conv2d(n_filters*2, n_filters*4, kernel_size=(kernel_size,kernel_size), padding=(kernel_size//2,kernel_size//2)),
                nn.BatchNorm2d(n_filters*4, track_running_stats=False),
                nn.ReLU(inplace = True),
                nn.Dropout(p=0.5)
        )
        self.conv4 = nn.Sequential(
                nn.Conv2d(n_filters*4, n_filters*8, kernel_size=(kernel_size,kernel_size), padding=(kernel_size//2,kernel_size//2)),
                nn.BatchNorm2d(n_filters*8, track_running_stats=False),
                nn.ReLU(inplace = True),
                nn.Dropout(p=0.5)
        )

    def forward(self, x):   #512 x 512 x 1
        x1 = self.conv1(x)  
        x1 = self.pool(x1)  #256 x 256 x 32
        x2 = self.conv2(x1)
        x2 = self.pool(x2)  #128 x 128 x 64
        x3 = self.conv3(x2)
        x3 = self.pool(x3)  # 64 x 64 x 128
        x4 = self.conv4(x3)
        x4 = self.pool(x4)  # 32 x 32 x 256

        return x4

class ClothDecoder(nn.Module):
    def __init__(self, image_size, kernel_size=3, bilinear=True, n_filters=32, dropout=False, drop_prob=0.2):
        super(ClothDecoder, self).__init__()
        
        self.image_size = image_size
        self.n_filters = n_filters

        self.up= Up(n_filters*8, n_filters*4, bilinear, kernel_size=kernel_size, dropout=dropout, drop_prob=drop_prob, scale_factor=2)
        self.up1= Up(n_filters*4, n_filters*2, bilinear, kernel_size=kernel_size, dropout=dropout, drop_prob=drop_prob, scale_factor=2)
        self.up2= Up(n_filters*2, n_filters, bilinear, kernel_size=kernel_size, dropout=dropout, drop_prob=drop_prob, scale_factor=2)
        self.up3= Up(n_filters, n_filters, bilinear, kernel_size=kernel_size, dropout=dropout, drop_prob=drop_prob, scale_factor=2)
        self.outc = OutConv(n_filters, 1)

    def forward(self, x): 
        x = self.up(x)  # 64 x 64 x 256 
        x = self.up1(x) # 128 x 128 x128
        x = self.up2(x) # 256 x 256 x64
        x = self.up3(x) # 512 x 512 x32
        
        out = self.outc(x)

        return out